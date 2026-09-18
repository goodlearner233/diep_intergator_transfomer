"""Architecture probe for copying into a notebook: real forward, no training."""
import os
import sys
from pathlib import Path

# 1. 明确使用哪个仓库，避免导入电脑里另一个 MatGL。
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import matgl
from pymatgen.core import Structure, Lattice
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.models._m3gnet import M3GNet
from matgl.utils.training import PotentialLightningModule, xavier_init

assert Path(matgl.__file__).resolve().is_relative_to((ROOT / "src").resolve()), \
    "导入了其他 MatGL：请重启内核，再运行本单元格。"
matgl.set_default_dtype("float", 32)
torch.manual_seed(42)
print("MatGL 来源：", matgl.__file__)

# 2. 按准备提交的配置创建完整网络；这里是随机初始化，没有加载训练权重。
model = M3GNet(
    element_types=DEFAULT_ELEMENTS, is_intensive=False,
    readout_type="transformer",
    nblocks=3, dim_node_embedding=64, dim_edge_embedding=64,
    transformer_nhead=4, transformer_num_layers=1,
    transformer_dim_ff=128, transformer_dropout=0.0,
    cutoff=5.0, threebody_cutoff=4.0,
    diep_grid_half_length=5.0, diep_grid_spacing=1.0,
)
xavier_init(model)
lit = PotentialLightningModule(
    model=model, energy_weight=1.0, force_weight=1.0, stress_weight=0.1,
    loss="huber_loss", lr=0.001, decay_steps=1000, decay_alpha=0.01,
)
lit.eval()  # 评估模式；下面不调用 Trainer.fit，也不更新参数。
potential = lit.model
assert potential.model is model
assert isinstance(model.final_layer.atomic_head, torch.nn.Linear)
print("实际封装：", type(lit).__name__, "→", type(potential).__name__, "→", type(model).__name__)
print("Readout：", type(model.final_layer).__name__)
print("参数量：", sum(p.numel() for p in model.parameters()))

# 3. 一个含3颗原子的简单结构；20 Å 晶胞避免邻近周期镜像进入cutoff。
structure = Structure(
    Lattice.cubic(20), ["Li", "O", "H"],
    [[8, 8, 8], [11.2, 8, 8], [9, 9.732, 8]], coords_are_cartesian=True,
)
graph, lattice, _ = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=5.0).get_graph(structure)

# 4. hook只是观察器：模块真正执行后，记录其输出形状，不修改计算。
events = []
def shape_of(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return tuple(shape_of(x) for x in value)
    return None

def watch(name):
    def hook(layer, inputs, output):
        events.append((name, shape_of(output)))
    return hook

watched = [("初始 embedding [原子, 边, state]", model.embedding)]
for i in range(model.n_blocks):
    watched += [
        (f"第{i+1}层：三体更新后的边", model.three_body_interactions[i]),
        (f"第{i+1}层：图卷积 [边, 原子, state]", model.graph_layers[i]),
    ]
watched += [
    ("Transformer [结构, 原子位置, 特征]", model.final_layer.transformer),
    ("Linear [原子, 能量通道]", model.final_layer.atomic_head),
]
handles = [layer.register_forward_hook(watch(name)) for name, layer in watched]
# 旧贝塞尔模块仍登记在模型里；另外观察它是否真的被调用。
legacy_calls = []
def watch_legacy(layer, inputs, output):
    legacy_calls.append(True)
handles.append(model.basis_expansion.register_forward_hook(watch_legacy))

# 5. 使用支持求导的数学 attention 后端，兼容本地旧版和新版 PyTorch。
try:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention_context = sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        attention_context = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False)
    # 力/应力需要坐标求导，因此这里不能包 torch.no_grad()。
    with attention_context, torch.enable_grad():
        energy, forces, stress, _ = potential(graph, lattice)
finally:
    for handle in handles:
        handle.remove()  # 避免重复运行单元格时，观察器越挂越多。

# 6. basis由普通函数算出：读取模型已经保存的实际中间结果。
features = model.feature_dict
print("\n原子数：", graph.num_nodes, "；有向边数：", graph.num_edges)
print("二体 DIEP basis（已乘二体平滑）：", shape_of(features["bond_expansion"]))
print("三体 DIEP basis（后续交互层再使用cutoff）：", shape_of(features["three_body_basis"]))
for name, shape in events:
    print(f"{name}: {shape}")
print("按结构求和后的总能量形状：", shape_of(features["final"]))
print("\n各原子能量：", features["readout"].detach().flatten().tolist())
print("结构总能量：", energy.detach().reshape(-1).tolist())
print("力的形状：", shape_of(forces), "；应力形状：", shape_of(stress))

# 此处无能量缩放/元素参考能，验证原子能量之和等于Potential输出。
assert torch.allclose(features["readout"].detach().sum(), energy.detach().sum(), atol=1e-6)
assert all(torch.isfinite(x).all() for x in [energy, forces, stress])
assert len(events) == 1 + 2 * model.n_blocks + 2
print("保留的旧贝塞尔模块实际调用次数：", len(legacy_calls))
assert not legacy_calls
print("\n通过：三体交互、图卷积、Transformer、Linear均实际执行；E/F/应力有限。")

# 想展开查看所有子层（Linear、激活、LayerNorm等），再单独运行：print(model)
