"""PyTorch Geometric implementation of M3GNet.

Uses ``edge_index`` / scatter-based message passing and a tensor-bundle
line graph from ``matgl.graph._compute.create_line_graph``.
"""
#1. Structure -> graph
#    由 graph converter 做。
#    得到 node_type / z, edge_index, frac_coords, lattice, pbc_offset 等。
#    在 predict_structure 里还会得到 pos 和 pbc_offshift。

# 2. compute_pair_vector_and_distance
#    根据 pos, edge_index, pbc_offshift
#    计算每条边的 bond_vec 和 bond_dist，也就是 r_ij。

# 3. BondExpansion
#    把 bond_dist 展开成二体径向 basis / expanded_dists。
#    这是普通边特征的距离基展开。

# 4. create_line_graph
#    根据 edge_index 和 bond 信息，
#    找到哪些 bond-bond pair 能组成三体关系。
#    也就是构造 line_edge_index:
#    bond -> bond，对应 j-i-k 角度关系。

# 5. compute_theta_and_phi
#    根据 line_edge_index、bond_vec、bond_dist
#    计算三体角度信息，主要是 cos(theta_jik)。

# 6. SphericalBesselWithHarmonics
#    把三体中的距离和角度展开成 three_body_basis。
#    这是给三体相互作用用的 basis。

# 7. EmbeddingBlock
#    把 node_types、expanded_dists、state_attr
#    变成初始 node_feat、edge_feat、state_feat。

# 8. 多层 M3GNet 循环
#    for i in range(n_blocks):

#        8.1 ThreeBodyInteractions
#            用 three_body_basis 更新 edge_feat，
#            让边特征带上三体/角度环境信息。

#        8.2 M3GNetBlock
#            用 edge_index、edge_feat、node_feat、state_feat、expanded_dists
#            做 graph convolution：
#                edge update
#                node update
#                optional state update

# 9. Readout

#    如果是 interatomic potential / extensive energy:
#        node_feat
#        -> WeightedReadOut
#        -> 每个原子的能量贡献 E_i
#        -> sum_i E_i
#        -> total energy E

#    如果是 general intensive property:
#        node_feat 或 edge_feat
#        -> weighted_atom / reduce_atom / set2set readout
#        -> graph-level vector
#        -> final MLP
#        -> property
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn
from pymatgen.core import Element  # 根据元素符号查询原子序数 Z

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute import (
    compute_pair_vector_and_distance,#计算边向量和距离
    compute_theta_and_phi,#计算theta角度和phi，两条边夹角确实只靠 bond_vec 和 bond_dist 就能算出来。代码后面把这个 cos_theta 拿去做角向基展开，给三体 interaction 使用。
    create_line_graph,#把边当成节点，边之间的关系当成边，构建一个 line graph（返回是一个 dict 包含 line graph 的 edge_index 和一些预计算的张量）
    ensure_line_graph_compatibility, #辅助函数确保兼容
)
from matgl.layers import (
    MLP,  # 普通多层感知机，用在输出层、三体 atom update 等位置
    ActivationFunction,  # 根据字符串选择激活函数，比如 swish、softplus2
    DIEPGrid,  # 根据范围和间距，创建 DIEP 的二维采样网格
    EmbeddingBlock,#把原子类型和距离信息编码成特征向量，作为后续 message passing 的输入
    GatedMLP, #gMLP。论文里定义了
    M3GNetBlock,  # M3GNet 的主图卷积层，负责 edge/node/state 消息传递
    Set2SetReadOut,#对MEGNET是核心但对M3GNET只是一个可选参数
    SphericalBesselWithHarmonics,#对应公式2，把三体几何距离+角度变成三体特征向量
    ThreeBodyInteractions,#三体信息聚合回边e_ij
    compute_bond_features,  # 键长 + 两端原子序数 + 网格 -> 二体 DIEP basis
    compute_triplet_features,  # 三体坐标 + 原子序数 Z + 网格 -> 三体 DIEP basis
)
from matgl.layers._readout_torch import ReduceReadOut, TransformerAtomicReadOut, WeightedAtomReadOut, WeightedReadOut  # Reduce/Weighted readout，用于普通性质或势能分支
from matgl.utils.cutoff import polynomial_cutoff  # 三体 cutoff，让远距离三体作用平滑衰减到 0

from ._core import MatGLModel, _warn_feature_dict_kwarg

if TYPE_CHECKING:
    from matgl.graph._converters import GraphConverter
logger = logging.getLogger(__name__)


class M3GNet(MatGLModel):
    """PyG implementation of the M3GNet model."""

    __version__ = 2

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,  # 默认元素表，决定模型认识哪些原子类型
        dim_node_embedding: int = 64,  # 节点/原子隐藏向量维度，也就是 embedding 之后 node_feat 的维度
        dim_edge_embedding: int = 64,  # 边/bond 隐藏向量维度
        dim_state_embedding: int = 0,  # 全局 state/u 维度，默认 0 表示通常不用 state
        ntypes_state: int | None = None,  # 离散 state 类别数，如果 state 是整数类别才用
        dim_state_feats: int | None = None,  # 连续 state 特征维度，如果 state 是 T/P 等连续值才用
        max_n: int = 3,  # 径向 basis 阶数，控制距离展开大小 对应公式2中z_ln，径向 radial basis 数量，对应球贝塞尔函数里和距离 r 有关的展开通道
        max_l: int = 3,  # 角向 basis 阶数，控制三体角度展开大小，对应公式2中Y_l，角向 angular basis 数量，对应球谐函数里和角度 θ 有关的展开通道
        nblocks: int = 3,  # M3GNet 层数，每层先 three-body 再 graph conv
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "SphericalBessel",  # 保留原有参数接口；当前二体 basis 已改用 DIEP
        is_intensive: bool = True,  # True 做普通性质预测；False 做总能量/势函数 sum
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom", "transformer"] = "weighted_atom",  # 普通性质预测时的聚合方式,我加入了 transformer 方式
        transformer_nhead: int = 8,  # Transformer readout 的多头 attention 头数
        transformer_num_layers: int = 3,  # TransformerEncoderLayer 堆叠层数
        transformer_dim_ff: int = 256,  # Transformer 内部 FFN/MLP 隐藏维度
        transformer_dropout: float = 0.1,  # Transformer 内部 dropout
        task_type: Literal["classification", "regression"] = "regression",  # 任务类型，默认回归
        cutoff: float = 5.0,  # 二体边截断半径
        threebody_cutoff: float = 4.0,  # 三体相互作用截断半径
        units: int = 64,  # block 内部 MLP 隐藏维度
        ntargets: int = 1,  # 输出目标数量，单目标回归就是 1
        use_smooth: bool = False,  # 是否使用 smooth 版本径向 basis
        use_phi: bool = False,  # 是否使用完整方位角 phi，默认主要用 bond-bond 夹角
        niters_set2set: int = 3,  # Set2Set 迭代次数，只在 set2set readout 用
        nlayers_set2set: int = 3,  # Set2Set 内部 LSTM 层数，只在 set2set readout 用
        field: Literal["node_feat", "edge_feat"] = "node_feat",  # readout 聚合 node_feat 还是 edge_feat，默认节点特征
        include_state: bool = False,  # 是否把全局 state/u 加入消息传递
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",  # 激活函数类型
        dropout: float | None = None,  # dropout 比例，默认不用

        # DIEP 网格设置：x、y 两个方向都覆盖 [-L, L]。
        diep_grid_half_length: float = 5.0,  # L：网格范围的一半
        diep_grid_spacing: float = 1.0,  # 相邻采样点的目标间距

        **kwargs,
    ):
        """Initialize the M3GNet model."""
        super().__init__()

        self.save_args(locals(), kwargs)  # 保存初始化参数，方便模型保存/加载

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()  # 把字符串激活函数名变成真正的 nn.Module
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types or DEFAULT_ELEMENTS  # 保存元素表，后面 graph converter 和 embedding 都要对齐
        # DIEP：建立“元素类别编号 → 真实原子序数 Z”的固定查找表。
        # 顺序与 self.element_types 相同。
        # 例如元素表为 ("Li", "O", "H")，这里保存 [3, 8, 1]。
        self.register_buffer(
            "atomic_number_table",
            torch.tensor(
                [Element(symbol).Z for symbol in self.element_types],
                dtype=torch.long,
            ),
            persistent=False,
        )

        # DIEP：保存网格设置，后面计算 basis 时据此创建网格。
        self.diep_grid_half_length = diep_grid_half_length
        self.diep_grid_spacing = diep_grid_spacing

        # DIEP 二体 basis 的维度：grid 模式下，每个采样点贡献一个分量。
        # 点数算法与 DIEPGrid 一致；默认每轴 11 点，共 121 个分量。
        if self.diep_grid_half_length <= 0 or self.diep_grid_spacing <= 0:
            raise ValueError("DIEP grid half length and spacing must be positive.")
        num_diep_axis_points = int(round(2 * self.diep_grid_half_length / self.diep_grid_spacing)) + 1
        if num_diep_axis_points < 2:
            raise ValueError("At least two DIEP grid points per axis are required.")
        degree_rbf = num_diep_axis_points * num_diep_axis_points
        # 下方 EmbeddingBlock 和 M3GNetBlock 都使用 degree_rbf 接收二体 basis。

        # 原 M3GNet 写法：根据贝塞尔和球谐展开参数计算三体 basis 的维度。
        # 保留作对照，当前不执行。
        # degree = max_n * max_l * max_l if use_phi else max_n * max_l

        # DIEP 写法：三体也采用同一张网格的 grid 输出，因此 basis 长度等于网格点数。
        degree = degree_rbf

        self.embedding = EmbeddingBlock(  # 创建初始 embedding 模块，生成 node/edge/state 初始特征
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=len(element_types),
            ntypes_state=ntypes_state,
            dim_state_feats=dim_state_feats,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.basis_expansion = SphericalBesselWithHarmonics(  # 创建三体 basis 展开模块，距离+角度 -> three_body_basis
            max_n=max_n,
            max_l=max_l,
            cutoff=cutoff,
            use_phi=use_phi,
            use_smooth=use_smooth,
        )
#         这段代码是在创建 nblocks 个“三体边更新器”。
# 每个更新器里有两个网络：
# 1. update_network_atom：用 v_k 生成公式 (2) 里的 sigmoid 权重
# 2. update_network_bond：把 ẽ_ij 转成公式 (3) 里的边修正量
# 真正的 three_body_basis 和 edge_feat 是 forward 里才传进去计算的。
#      
        self.three_body_interactions = nn.ModuleList(  # 每个 block 一个三体更新层，先把角度信息加到 edge_feat;在模型里保存一个三体更新层列表
            [
                ThreeBodyInteractions(
                    update_network_atom=MLP(  # 三体公式里和原子特征有关的更新网络，对应公式2的σ(W_v v_k + b_v)
                        dims=[dim_node_embedding, degree],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[dim_edge_embedding], use_bias=False),  # 三体公式里更新边特征的 GatedMLP，对应公式3# 对应公式(3)里的 g(W2 ẽ_ij) ⊙ σ(W1 ẽ_ij)，
# GatedMLP 内部有 value/gate 两条不同参数分支，逐元素相乘被封装在 forward 里。
                )
                for _ in range(nblocks)
            ]
        )

        dim_state_feats_used = dim_state_embedding

        self.graph_layers = nn.ModuleList(  # M3GNetBlock 列表，真正做 edge/node/state graph conv
            [
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_feats_used,
                    include_state=include_state,
                    dropout=dropout,
                )
                for _ in range(nblocks)
            ]
        )

        if is_intensive:  # 普通 intensive 性质预测分支，输出不随原子数简单相加
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding  # 根据 field 决定用节点特征readout还是用边特征readout
            if readout_type == "set2set":  # 如果选择 Set2Set 聚合，势函数主线一般不走这里
                if field != "node_feat": #只能对节点特征做 Set2Set 聚合，边特征暂时不支持
                    raise NotImplementedError("Set2Set readout on edge features is not implemented for PyG yet.")
                self.readout = Set2SetReadOut(  # type: ignore[call-arg]#Set2Set 是一种比较复杂的集合聚合方法，内部有 LSTM attention。它把所有节点看成一个无序集合，然后学一个 graph representation。
                    in_feats=input_feats, n_iters=niters_set2set, n_layers=nlayers_set2set
                )
                readout_feats = 2 * input_feats + dim_state_feats_used if include_state else 2 * input_feats  #set2set输出维度是 2 * input_feats，如果有 state 就加上 state 的维度
            elif readout_type == "weighted_atom":  # 默认 weighted atom 聚合分支
                self.readout = WeightedAtomReadOut(  # type: ignore[assignment]  #对原子特征做带权重的聚合 #这是默认分支。#它不是简单平均，而是给每个原子一个可学习权重：

                    in_feats=input_feats, dims=[units, units], activation=activation
                )
                readout_feats = units + dim_state_feats_used if include_state else units
            elif readout_type == "transformer":#新加的transformer readout 分支，参考 DiepFormer 论文,但是在此处intensive默认为不可用
                raise ValueError( "Transformer atomic-energy readout requires is_intensive=False."
                )
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore[assignment] #这里是最简单的readout，求所有节点/边特征求平均
                readout_feats = input_feats + dim_state_feats_used if include_state else input_feats

            dims_final_layer = [readout_feats, units, units, ntargets]  # 最终 MLP 的维度：graph vector -> 输出
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)  # 普通性质预测的输出 MLP
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()  # 分类任务最后接 sigmoid
        else:#如果是 extensive 能量预测分支，输出随原子数简单相加
            if task_type == "classification":#在此情况下，分类任务不适用，因为 extensive 能量预测是回归问题
                raise ValueError("Classification task cannot be extensive.")
            if readout_type == "transformer":
                self.final_layer = TransformerAtomicReadOut(
                    in_feats = dim_node_embedding,
                    num_targets = ntargets,
                    nhead = transformer_nhead,
                    num_layers = transformer_num_layers,
                    dim_ff = transformer_dim_ff,
                    dropout = transformer_dropout,
                )
            else:
                self.final_layer = WeightedReadOut(  # type: ignore[assignment]
                    in_feats=dim_node_embedding,
                    dims=[units, units],
                    num_targets=ntargets,
                )

        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = nblocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_state = include_state
        self.task_type = task_type
        self.is_intensive = is_intensive
        self.field = field
        self.readout_type = readout_type

    def _readout(self, node_feat: torch.Tensor, edge_feat: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:#根据设置，决定最后 readout 时用 node_feat 还是 edge_feat。就这个一个功能
        """Dispatch the configured readout on the right field tensor."""
        x = node_feat if self.field == "node_feat" else edge_feat  # 根据 field 选择读出节点特征或边特征
        if isinstance(self.readout, ReduceReadOut):#isinstance(self.readout, ReduceReadOut) 是在判断：self.readout 这个对象是不是 ReduceReadOut 类的实例，反正有点冗余
            return self.readout(x, batch)
        return self.readout(x, batch)

    def forward(
        self,
        g: Any,
        state_attr: torch.Tensor | None = None,
        l_g: dict[str, torch.Tensor] | None = None,
        return_all_layer_output: bool = False,
    ):
        """Forward pass of M3GNet (PyG).

        Intermediate layer features are always stored on ``self.feature_dict`` after
        every call (overwritten on each forward).

        Args:
            g: PyG ``Data`` (or ``Data``-like) object with attributes
                ``node_type`` (or ``z``), ``pos``, ``edge_index``, optionally
                ``pbc_offshift`` and ``batch`` / ``num_graphs``.
            state_attr: Per-graph state features (optional).
            l_g: Cached line-graph bundle from
                :func:`matgl.graph._compute.create_line_graph`. If ``None``,
                a fresh one is built from ``g``.
            return_all_layer_output: **Deprecated.** Use ``model.feature_dict`` after
                the forward call instead. Will be removed in matgl v5. When ``True``
                the feature dict is still returned for backwards compatibility.
        """
        if return_all_layer_output: 
            _warn_feature_dict_kwarg("return_all_layer_output")
        #读入图信息
        node_types = getattr(g, "node_type", getattr(g, "z", None))  # 读取原子类型，node_type 或 z
        pos = g.pos  # 从图g拿原子笛卡尔坐标
        # DIEP：根据类别编号查表，得到每颗原子的真实原子序数。
        # node_types 仍然保留，继续供原来的节点 Embedding 使用。

        # 查找表与类别编号放在相同设备上，方便索引。
        atomic_table = self.atomic_number_table.to(
            device=node_types.device
        )

        # 按类别编号取出 Z，再转换成与坐标相同的浮点类型。
        atomic_numbers = atomic_table[node_types.long()].to(
            dtype=pos.dtype
        )

        edge_index = g.edge_index  # 原子图的边连接关系，shape 是 (2, num_edges)

        # DIEP 二体输入：根据每条边的两个端点，取得对应原子序数。
        # 注意区分：
        # atom_ids 保存“原子在当前图中的编号”；
        # z 保存这些原子的真实原子序数。

        # 1. 每条边的起点、终点原子编号，形状都是 [边数]。
        bond_src_atom_ids = edge_index[0]
        bond_dst_atom_ids = edge_index[1]

        # 2. 根据原子编号，从 atomic_numbers 中取出真实 Z。
        bond_src_z = atomic_numbers[bond_src_atom_ids]
        bond_dst_z = atomic_numbers[bond_dst_atom_ids]

        # 3. 将同一条边的两个 Z 配在一起。
        # 结果形状：[边数, 2]，每一行都是一条边的 [Z_i, Z_j]。
        bond_atomic_numbers = torch.stack(
            (bond_src_z, bond_dst_z),
            dim=1,
        )

        pbc_offshift = getattr(g, "pbc_offshift", None)  # 周期性边界下的真实空间偏移
        batch = getattr(g, "batch", None)  # batch 中每个节点属于哪个结构
        num_graphs = getattr(g, "num_graphs", None)
        num_nodes = pos.size(0)  # 当前 batch 里的总原子数
        num_bonds = edge_index.size(1)  # 当前 batch 里的总边数
        if num_graphs is None:
            num_graphs = 1 if batch is None else int(batch.max().item()) + 1
        edge_batch = None if batch is None else batch[edge_index[0]].to(torch.long)  # 每条边属于 batch 中哪个结构

# #pos + edge_index
# -> 每条边的向量 bond_vec
# -> 每条边的距离 bond_dist
# -> 结合两端原子序数和网格，计算二体 DIEP basis expanded_dists
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)  # 根据坐标和 edge_index 计算每条边的向量和距离

        # DIEP：用模型保存的设置创建二维网格，本次计算的所有二体、三体片段共用。
        # 网格与键长使用相同的设备和浮点类型，方便后续一起计算。
        diep_grid = DIEPGrid(
            half_length=self.diep_grid_half_length,  # x、y 都从 -L 到 L
            spacing=self.diep_grid_spacing,  # 相邻采样点的目标间距
            device=bond_dist.device,  # 跟随键长放在 CPU 或 GPU 上
            dtype=bond_dist.dtype,  # 跟随键长使用 float32 或 float64 等类型
        )

        # DIEP 二体：批量计算每条边在各网格点上的 rho * V * delta_area。
        # 保留 expanded_dists 变量名，继续送入初始边 Embedding 和各 M3GNet block。
        # grid 模式输出 [边数, 网格点数]；默认是 [边数, 121]。
        expanded_dists = compute_bond_features(
            bond_dist=bond_dist,  # [边数]：原代码算出的键长
            atomic_numbers=bond_atomic_numbers,  # [边数, 2]：每条边两端的真实 Z
            grid=diep_grid,  # 这一批边共用的二维采样网格
            mode="grid",  # 保留每个采样点的贡献，形成 basis 向量
        )

        # DIEP 二体 cutoff 平滑：原始网格向量不会在建图 cutoff 处自动归零。
        # 每条边先根据自己的键长计算一个平滑因子，使用二体 self.cutoff。
        # pair_cutoff: [边数]；unsqueeze(-1) 后为 [边数, 1]。
        pair_cutoff = polynomial_cutoff(bond_dist, self.cutoff)

        # 这一条边的全部网格分量，乘同一个因子；basis 形状保持不变。
        # 后面的初始边 Embedding 和所有 M3GNet block 都使用这份平滑后的 basis。
        expanded_dists = expanded_dists * pair_cutoff.unsqueeze(-1)

        if l_g is None:
            l_g = create_line_graph(edge_index, bond_dist, bond_vec, pbc_offshift, num_nodes, self.threebody_cutoff)  # 构造 line graph，找 bond-bond pair 形成三体角度
        else:
            l_g = ensure_line_graph_compatibility(l_g, bond_dist, bond_vec, pbc_offshift, self.threebody_cutoff)  # 复用已有 line graph，只刷新距离和向量

        # 三体输入准备：将筛选后的局部边编号，转换成原图边编号。

        # 1. 读取三体对应的两条边，使用筛选后的局部编号。
        # 形状：[2, 三体数]，每一列是一组三体的两条边。
        triplet_local_edge_ids = l_g["line_edge_index"]

        # 2. 读取编号对应表。
        # kept_edge_ids[局部边编号] = 原图边编号。
        kept_edge_ids = l_g["kept_edge_ids"]

        # 3. 查表，得到每个三体对应的两条原图边。
        # 形状仍然是：[2, 三体数]。
        triplet_edge_ids = kept_edge_ids[triplet_local_edge_ids]

        # M3GNet 已经找好了三体关系；这里查询每个三体涉及哪三颗原子。
        # 以下变量保存的都是边或原子的“编号”，还不是原子序数 Z。

        # 1. 取出每个三体的两条原图边编号，形状都是 [三体数]。
        triplet_first_edge_ids = triplet_edge_ids[0]
        triplet_second_edge_ids = triplet_edge_ids[1]

        # 2. 第一条边的起点，就是三体的中心原子。
        triplet_center_atom_ids = edge_index[0, triplet_first_edge_ids]

        # 3. 第一条边的终点，就是第一个邻居原子 j。
        triplet_neighbor_j_atom_ids = edge_index[1, triplet_first_edge_ids]

        # 4. 第二条边的终点，就是第二个邻居原子 k。
        triplet_neighbor_k_atom_ids = edge_index[1, triplet_second_edge_ids]

        # 三体输入准备：根据三颗原子的原图编号，查询它们的原子序数 Z。

        # 1. 把每个三体涉及的三个原子编号排成一行。
        # 顺序与 DGL 实现一致：[邻居 j，中心原子，邻居 k]。
        # 形状：[三体数, 3]。
        triplet_atom_ids = torch.stack(
            (
                triplet_neighbor_j_atom_ids,
                triplet_center_atom_ids,
                triplet_neighbor_k_atom_ids,
            ),
            dim=1,
        )

        # 2. atomic_numbers 已经保存了图中每颗原子的真实 Z。
        # 用上面的原子编号查表，得到每个三体的三个 Z。
        # 形状：[三体数, 3]，顺序仍为 [邻居 j，中心原子，邻居 k]。
        triplet_atomic_numbers = atomic_numbers[triplet_atom_ids]

        # 为每个三体准备原始三维坐标，之后交给 DIEP 函数转换成二维坐标。

        # 1. 用中心原子的原图编号，查询它的三维坐标。
        triplet_center_pos = pos[triplet_center_atom_ids]

        # 2. 中心坐标 + 指向邻居 j 的边向量，得到邻居 j 的坐标。
        triplet_neighbor_j_pos = (
            triplet_center_pos + bond_vec[triplet_first_edge_ids]
        )

        # 3. 中心坐标 + 指向邻居 k 的边向量，得到邻居 k 的坐标。
        triplet_neighbor_k_pos = (
            triplet_center_pos + bond_vec[triplet_second_edge_ids]
        )

        # 4. 将每个三体的三颗原子的坐标配成一组。
        # 顺序与 Z 一致：[邻居 j，中心，邻居 k]。
        # 形状：[三体数, 3颗原子, xyz三个坐标分量]。
        triplet_coords = torch.stack(
            (
                triplet_neighbor_j_pos,
                triplet_center_pos,
                triplet_neighbor_k_pos,
            ),
            dim=1,
        )

        # 调用已经写好的 DIEP 三体函数，批量计算每个三体的 basis。
        # 函数内部完成：二维转换 -> 网格距离 -> 密度和势因子 -> 网格贡献。
        three_body_basis = compute_triplet_features(
            coords=triplet_coords,  # 每个三体的三颗原子的三维坐标
            atomic_numbers=triplet_atomic_numbers,  # 对应三颗原子的 Z
            grid=diep_grid,  # 二维网格的采样点和面积权重
            mode="grid",  # 保留各网格点的贡献，输出 [三体数, 网格点数]
        )
        three_body_cutoff = polynomial_cutoff(bond_dist, self.threebody_cutoff)  # 三体 cutoff 权重，远距离平滑衰减到 0

        node_feat, edge_feat, state_feat = self.embedding(node_types, expanded_dists, state_attr)  # 生成初始 node/edge/state hidden features；这里把 expanded_dists/e0ij 编码或投影成 edge_feat，不是再做距离展开
        if self.include_state and state_feat is not None and state_feat.dim() == 1:
            state_feat = state_feat.unsqueeze(0)  # 单图 state 补一个 batch 维度

        fea_dict: dict[str, Any] = {
            "bond_expansion": expanded_dists,
            "three_body_basis": three_body_basis,
            "embedding": {"node_feat": node_feat, "edge_feat": edge_feat, "state_feat": state_feat},
        }

        edge_dst_atom = edge_index[1]  # 每条边的终点原子 index，三体更新里要用
        # 旧写法：线图使用筛选后的小表编号，不能直接拿来索引原图的边特征。
        # line_edge_index = l_g["line_edge_index"]
        # n_triple_ij = l_g["n_triple_ij"]

        # 新写法：与前面计算三体坐标、Z 时使用同一套原图边编号。
        # 第一行指定三体贡献加回哪条边，第二行指定提供邻居 k 的另一条边。
        line_edge_index = triplet_edge_ids

        # 按原图边编号统计三体数；未参与三体的原图边对应 0。
        # 保留原更新层的参数接口，形状为 [原图边数]。
        n_triple_ij = torch.bincount(line_edge_index[0].long(), minlength=num_bonds)

        for i in range(self.n_blocks):  # 逐层执行 M3GNet：每层先三体更新边，再 graph conv
            edge_feat = self.three_body_interactions[i](  # 用三体 basis 更新 edge_feat，把角度信息写进边特征
                edge_dst_atom,
                line_edge_index,
                n_triple_ij,
                num_bonds,
                three_body_basis,
                three_body_cutoff,
                node_feat,
                edge_feat,
            )
            edge_feat, node_feat, state_feat = self.graph_layers[i](  # 调用 M3GNetBlock，执行 edge/node/state 消息传递
                edge_index,
                edge_feat,
                node_feat,
                state_feat,
                expanded_dists,
                batch,
                edge_batch,
                num_nodes,
                num_graphs,
            )
            fea_dict[f"gc_{i + 1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }

        if self.is_intensive:  # 如果 readout 走 intensive 性质预测分支，输出不随原子数简单相加
            field_vec = self._readout(node_feat, edge_feat, batch)  # 根据 field 聚合 node/edge 特征，得到 graph-level vector

            if self.include_state and state_feat is not None:
                state_view = state_feat.view(num_graphs, -1)  # state_feat 可能是 (num_graphs, dim_state) 或 (dim_state,) 需要 reshape
                readout_vec = torch.hstack([field_vec, state_view])  # 把 graph-level vector 和 state 拼接
            else:
                readout_vec = field_vec

            fea_dict["readout"] = readout_vec
            output = self.final_layer(readout_vec)  # 最终 MLP 输出预测结果

            if self.task_type == "classification":
                output = self.sigmoid(output)  # 分类任务最后接 sigmoid
        else:#如果readout走 extensive 能量预测分支，输出随原子数简单相加
            if self.readout_type == "transformer":
                #Transformer 需要 batch，以保证每个结构只在自己的原子之间做 attention
                atomic = self.final_layer(node_feat, batch) #这里的final layer是前面初始化的，根据输入readout_type不同而不同
            else:
                atomic = self.final_layer(node_feat)  # 原本M3GNET，逐原子GATED MLP不需要batch
            fea_dict["readout"] = atomic
            atomic = atomic.view(-1)
            if batch is None:
                output = atomic.sum().view(1)  # 单个结构：所有原子能量贡献求和得到总能量
            else: #多个结构batch时，用 index_add 按 graph 分组把原子能量求和
                output = torch.zeros(num_graphs, dtype=atomic.dtype, device=atomic.device)
                output = output.index_add(0, batch.to(torch.long), atomic)  # batch 多个结构：按 graph 分组把原子能量求和

        fea_dict["final"] = output
        self.feature_dict = fea_dict  # 保存中间特征，方便调试/查看每层输出
        if return_all_layer_output:
            return fea_dict
        return torch.squeeze(output)  # 去掉多余维度后返回预测结果，这里是能量

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
    ):
        """Convenience method to predict a property from a structure (PyG).

        Args:
            structure: An input crystal/molecule.
            state_feats: Optional state attributes.
            graph_converter: Graph converter. Defaults to ``Structure2Graph``.
            output_layers: Currently unused; kept for API symmetry with other models.
            return_features: **Deprecated.** Use ``model.feature_dict`` after calling
                ``predict_structure`` instead. Will be removed in matgl v5.
        """
        import matgl

        if return_features:
            _warn_feature_dict_kwarg("return_features")

        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # 默认用 Structure2Graph 把 pymatgen Structure 转成图
        g, lat, state_attr_default = graph_converter.get_graph(structure)  # 得到图 g、晶格 lat 和默认 state
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])  # 把周期性偏移转成笛卡尔空间偏移
        g.pos = g.frac_coords @ lat[0]  # 分数坐标乘晶格矩阵得到笛卡尔坐标
        if state_feats is None:
            state_feats = torch.tensor(state_attr_default, dtype=matgl.float_th)  # 没有手动传 state 时使用 graph converter 给的默认 state
        if return_features:
            self(g=g, state_attr=state_feats)
            return self.feature_dict
        return self(g=g, state_attr=state_feats).detach()  # 直接调用 forward 做预测，并 detach 出普通张量
# predict_structure是一个方便预测的接口，直接输入 pymatgen Structure
# -> 自动转成 graph
# -> 调用 forward()
# -> 返回预测结果


##啊啊啊测试专用哈哈哈哈
