#此文件作用:输入片段的原子种类和集合关系，输出DIEP描述符

"""DIEP descriptors computed on a two-dimensional grid."""


"""
计算 DIEP 二体、三体描述符。

计算流程：
1. 创建二维网格，得到采样点和面积权重。
2. 将原子片段放入标准二维坐标系。
3. 计算采样点到片段原子的距离平方。
4. 计算近似电子密度与势因子。
5. 输出网格贡献向量，或对网格贡献求和。

当前进度：
已实现网格；正在用双原子例子验证距离计算。
"""



from dataclasses import dataclass

import torch
# ============================================================
# 1. 二维网格：准备采样点 r_q 和面积权重 ΔA
# ============================================================

@dataclass
class DIEPGrid:
    half_length: float
    spacing: float
    device: torch.device
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.half_length <= 0 or self.spacing <= 0:
            raise ValueError("half_length and spacing must be positive.")

        num_points = int(round(2 * self.half_length / self.spacing)) + 1 #生成一个坐标轴，num_points是坐标轴上的点数，+1是因为包含两端点
        if num_points < 2:
            raise ValueError("At least two grid points per axis are required.")

        self.axis = torch.linspace(       #根据起点，终点和点数，生成均匀分布的坐标，这只是横轴或纵轴的一维坐标，不是整张网格
            -self.half_length,
            self.half_length,
            steps=num_points,
            device=self.device,
            dtype=self.dtype,
        )

        grid_x, grid_y = torch.meshgrid(
            self.axis, self.axis, indexing="xy"
        )

        self.points = torch.stack(
            (grid_x, grid_y), dim=-1
        ).reshape(-1, 2)

        actual_spacing = self.axis[1] - self.axis[0] #算出实际的网格间距，可能会因为四舍五入而与self.spacing略有不同
        self.delta_area = actual_spacing.square()#算出每个网格点对应的面积，实际网格间距的平方




# ============================================================
# 2. 二体几何：键长 → 标准二维坐标 → 到网格点的距离平方
# ============================================================

def compute_bond_grid_geometry(
    bond_dist: torch.Tensor,
    grid_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据键长构造标准双原子片段，并计算到网格点的距离平方。

    输入：
        bond_dist:   [E]，每条边的键长。
        grid_points: [P, 2]，二维网格采样点。
        两个输入应使用相同的设备和浮点类型。

    输出：
        fragment_coords: [E, 2, 2]，每条边的两个原子坐标。
        dist_sq:         [E, 2, P]，每个原子到各网格点的距离平方。
    """
    # 2.1 键长的一半，以及所有原子的 y=0 坐标
    half_dist = bond_dist / 2
    zeros = torch.zeros_like(half_dist)

    # 2.2 两个原子分别位于 (-d/2, 0)、(d/2, 0)
    atom_i = torch.stack((-half_dist, zeros), dim=-1)
    atom_j = torch.stack((half_dist, zeros), dim=-1)

    # 2.3 每条边的两个原子合成一个片段：[E, 2, 2]，代表有一E个片段，每个片段有两个原子，每个原子有x和y两个坐标
    fragment_coords = torch.stack((atom_i, atom_j), dim=1)

    # 2.4 每个片段原子与每个网格点配对，计算坐标差
    # [E, 2, 1, 2] - [1, 1, P, 2] → [E, 2, P, 2]
    diff = (
        fragment_coords.unsqueeze(2)
        - grid_points.unsqueeze(0).unsqueeze(0)
    )

    # 2.5 d² = Δx² + Δy²：[E, 2, P, 2] → [E, 2, P]
    dist_sq = diff.square().sum(dim=-1)

    return fragment_coords, dist_sq

# ============================================================
# 3. 近似密度：距离平方 → 高斯贡献 → 对片段原子求和
# ============================================================

def compute_fragment_density(
    dist_sq: torch.Tensor,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    输入：
        dist_sq: [片段数, 原子数, 网格点数]，距离平方。
        sigma: 高斯宽度参数，必须为正。

    输出：
        rho_atoms: 每个原子在每个网格点的密度贡献。
        rho_total: 每个片段在每个网格点的总密度。
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    # 3.1 将每个距离平方代入高斯函数
    gaussian = torch.exp(-dist_sq / sigma)

    # 3.2 沿用本地 DGL 的二维高斯归一化
    rho_atoms = gaussian / (torch.pi * sigma)

    # 3.3 对片段中的原子求和，保留各网格点
    rho_total = rho_atoms.sum(dim=1)

    return rho_atoms, rho_total

# ============================================================
#到这已经有了同一个网格点的近似密度贡献，接下来可以计算势因子
# ============================================================

# ============================================================
# 4. 势因子：原子序数 + 距离平方 → 各原子贡献 → 片段总势
# ============================================================

def compute_fragment_potential(
    dist_sq: torch.Tensor,
    atomic_numbers: torch.Tensor,
    softening_epsilon: float = 0.5,
    use_effective_charge: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    输入：
        dist_sq: [片段数, 原子数, 网格点数]，距离平方。
        atomic_numbers: [片段数, 原子数]，真实原子序数。

    输出：
        potential_atoms: 各原子在各网格点的势贡献。
        potential_total: 对片段原子求和后的势因子。
    """
    if softening_epsilon < 0:
        raise ValueError("softening_epsilon must be non-negative.")

    # 4.1 按 DGL 配置选择 sqrt(Z) 或 Z
    if use_effective_charge:
        charges = atomic_numbers.sqrt()
    else:
        charges = atomic_numbers

    # 4.2 计算软化后的距离分母
    denominator = torch.sqrt(dist_sq + softening_epsilon**2)

    # 4.3 每个原子的电荷除以它到各网格点的距离分母
    potential_atoms = charges.unsqueeze(-1) / denominator

    # 4.4 对片段里的原子求和，保留各网格点
    potential_total = potential_atoms.sum(dim=1)

    return potential_atoms, potential_total


# ============================================================
# 5. 网格贡献：密度 × 势因子 × 面积权重
#    grid 保留逐点贡献；sum 对网格求和
# ============================================================

def compute_grid_features(
    rho_total: torch.Tensor,
    potential_total: torch.Tensor,
    delta_area: torch.Tensor,
    mode: str = "grid",
) -> torch.Tensor:
    """
    输入：
        rho_total:       [片段数, 网格点数]，各点的总密度。
        potential_total: [片段数, 网格点数]，各点的总势因子。
        delta_area: 面积权重。

    输出：
        grid 模式：[片段数, 网格点数]。
        sum 模式： [片段数, 1]。
    """
    # 5.1 逐点计算 I_q = rho_q * V_q * ΔA
    integrand = rho_total * potential_total * delta_area

    # 5.2 保留每个网格点的贡献
    if mode == "grid":
        return integrand

    # 5.3 对网格点求和，得到每个片段的近似积分值
    if mode == "sum":
        return integrand.sum(dim=1, keepdim=True)

    raise ValueError("mode must be 'grid' or 'sum'.")


# ============================================================
# 6. 二体描述符入口：把前面的计算步骤串起来，这个函数就是以后主模型调用二体计算的入口.
# ============================================================

def compute_bond_features(
    bond_dist: torch.Tensor,
    atomic_numbers: torch.Tensor,
    grid: DIEPGrid,
    sigma: float = 1.0,
    softening_epsilon: float = 0.5,
    use_effective_charge: bool = True,
    mode: str = "grid",
) -> torch.Tensor:
    """由键长和原子序数计算每条边的 DIEP 描述符。"""

    # 6.1 键长 → 标准坐标 → 距离平方
    _, dist_sq = compute_bond_grid_geometry(
        bond_dist,
        grid.points,
    )

    # 6.2 距离平方 → 各网格点的总密度
    _, rho_total = compute_fragment_density(
        dist_sq,
        sigma=sigma,
    )

    # 6.3 距离平方、原子序数 → 各网格点的总势因子
    _, potential_total = compute_fragment_potential(
        dist_sq,
        atomic_numbers,
        softening_epsilon=softening_epsilon,
        use_effective_charge=use_effective_charge,
    )

    # 6.4 密度 × 势因子 × 面积权重 → 描述符
    return compute_grid_features(
        rho_total,
        potential_total,
        grid.delta_area,
        mode=mode,
    )

# ============================================================
# 7. 三体几何：计算三条边长，找到最长边的两个端点
# ============================================================

def find_triplet_longest_edge(
    coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    输入：
        coords: [T, 3, 3]，每个三体中三颗原子的三维坐标。

    输出：
        lengths: [T, 3]，边 0-1、1-2、0-2 的长度。
        longest_pair: [T, 2]，最长边两个端点的片段内编号。
    """
    # 7.1 三对原子的坐标分别相减，得到三条边的向量
    vec01 = coords[:, 0] - coords[:, 1]
    vec12 = coords[:, 1] - coords[:, 2]
    vec02 = coords[:, 0] - coords[:, 2]

    # 7.2 向量的长度 = sqrt(Δx² + Δy² + Δz²)
    length01 = torch.linalg.vector_norm(vec01, dim=-1)
    length12 = torch.linalg.vector_norm(vec12, dim=-1)
    length02 = torch.linalg.vector_norm(vec02, dim=-1)

    # 每个三体的三个长度，按 0-1、1-2、0-2 的顺序排列
    lengths = torch.stack(
        (length01, length12, length02),
        dim=1,
    )

    # 7.3 找到每个三体中，最大长度所在的位置
    longest_edge_id = lengths.argmax(dim=1)

    # 7.4 将“长度所在的位置”转换成“两个端点的编号”
    edge_pairs = torch.tensor(
        [[0, 1], [1, 2], [0, 2]],
        dtype=torch.long,
        device=coords.device,
    )
    longest_pair = edge_pairs[longest_edge_id]

    return lengths, longest_pair

# ============================================================
# 8. 三体排序：确定未来的左端点、右端点、第三点
#
# 注意：
# coords 中的 0、1、2 只是原始片段内的原子编号，不代表左右。
# 本函数只返回排列顺序，不旋转、不平移，也不修改原子坐标。
# ============================================================

def get_triplet_vertex_order(
    coords: torch.Tensor,
    longest_pair: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    输入：
        coords: [T, 3, 3]，三颗原子的原始三维坐标。
        longest_pair: [T, 2]，最长边两个端点的原始片段内编号。

    输出：
        order: [T, 3]，依次为 [左端点编号, 右端点编号, 第三点编号]。
        输出的数字仍然引用 coords 中原来的原子位置。

    排列规则：
        最长边作为底边。
        第三点到左端点的距离 <= 到右端点的距离，允许 eps 数值容差。
    """
    # 8.1 先把最长边的两个端点暂时叫 u、v，还没有决定左右。
    # u_idx、v_idx 保存原子编号，不是原子坐标。
    u_idx = longest_pair[:, 0]
    v_idx = longest_pair[:, 1]

    # 8.2 找剩下的第三颗原子。
    # 片段内编号只有 0、1、2，总和是 3；减掉两个端点编号即可。
    third_idx = 3 - u_idx - v_idx

    # 8.3 为每个三体生成编号：0、1、2……，用于从各自片段中取坐标。
    triplet_idx = torch.arange(coords.shape[0], device=coords.device)

    # 现在才根据编号取出三个原子的坐标，每个变量的形状都是 [T, 3]。
    pos_u = coords[triplet_idx, u_idx]
    pos_v = coords[triplet_idx, v_idx]
    pos_third = coords[triplet_idx, third_idx]

    # 8.4 分别计算两个端点到第三点的距离。
    dist_u_third = torch.linalg.vector_norm(pos_u - pos_third, dim=-1)
    dist_v_third = torch.linalg.vector_norm(pos_v - pos_third, dim=-1)

    # 8.5 默认 u 在左、v 在右。
    # 如果 v 到第三点明显更近，就交换，让 v 在左、u 在右。
    # 加 eps 是为了避免几乎等长时因很小的数值误差触发交换。
    swap = dist_v_third + eps < dist_u_third

    left_idx = torch.where(swap, v_idx, u_idx)
    right_idx = torch.where(swap, u_idx, v_idx)

    # 8.6 保存未来的摆放顺序，里面仍然是原始原子编号。
    order = torch.stack((left_idx, right_idx, third_idx), dim=1)

    return order

# ============================================================
# 9. 三体投影：三维坐标 → 局部二维坐标 → 重心移到原点
#
# A、B 是已经确定左右顺序的最长边端点，C 是第三点。
# 这里的“横轴、纵轴”属于每个三角形自己的局部坐标系。
# ============================================================

def project_triplet_to_2d(
    coords: torch.Tensor,
    order: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    输入：
        coords: [T, 3, 3]，原始三维坐标。
        order: [T, 3]，依次为 [左端点编号, 右端点编号, 第三点编号]。

    输出：
        projected: [T, 3, 2]，以 AB 中点为原点的二维坐标。
        centered:  [T, 3, 2]，进一步将三角形重心移到原点的坐标。
        两个输出的原子顺序都是 [A, B, C]。
    """
    # 9.1 根据排序结果取出 A、B、C 的三维坐标。
    triplet_idx = torch.arange(coords.shape[0], device=coords.device)
    pos_a = coords[triplet_idx, order[:, 0]]
    pos_b = coords[triplet_idx, order[:, 1]]
    pos_c = coords[triplet_idx, order[:, 2]]

    # 9.2 最长边 AB 的中点，作为临时原点。
    origin = (pos_a + pos_b) / 2

    # 9.3 横轴方向：从 A 指向 B，再除以长度，变成单位向量。
    vec_ab = pos_b - pos_a
    length_ab = torch.linalg.vector_norm(vec_ab, dim=-1, keepdim=True)
    e_x = vec_ab / length_ab.clamp_min(eps)

    # 9.4 从临时原点指向 C 的向量。
    vec_c = pos_c - origin

    # C 沿横轴方向的位置：向量与横轴单位向量做点积。
    c_along_x = (vec_c * e_x).sum(dim=-1, keepdim=True)

    # 去掉横向分量，剩下的就是朝向 C 的垂直向量。
    perpendicular = vec_c - c_along_x * e_x
    height = torch.linalg.vector_norm(
        perpendicular, dim=-1, keepdim=True
    )

    # 9.5 普通三角形的纵轴：垂直向量除以自身长度。
    e_y_regular = perpendicular / height.clamp_min(eps)

    # 三点接近共线时，高接近 0，不能靠它稳定确定纵轴。
    # 沿用 DGL 的思路：准备一个与横轴垂直的备用方向。
    ref_x = torch.tensor(
        [1.0, 0.0, 0.0], dtype=coords.dtype, device=coords.device
    ).expand_as(e_x)
    ref_y = torch.tensor(
        [0.0, 1.0, 0.0], dtype=coords.dtype, device=coords.device
    ).expand_as(e_x)

    candidate1 = torch.cross(e_x, ref_x, dim=-1)
    candidate2 = torch.cross(e_x, ref_y, dim=-1)
    norm1 = torch.linalg.vector_norm(candidate1, dim=-1, keepdim=True)
    norm2 = torch.linalg.vector_norm(candidate2, dim=-1, keepdim=True)

    e_y_fallback = torch.where(
        norm1 > eps,
        candidate1 / norm1.clamp_min(eps),
        candidate2 / norm2.clamp_min(eps),
    )
    e_y = torch.where(height < eps, e_y_fallback, e_y_regular)

    # 保证第三点朝向局部纵轴的非负方向。
    orientation = (vec_c * e_y).sum(dim=-1, keepdim=True)
    e_y = torch.where(orientation < 0, -e_y, e_y)

    # 9.6 按 [A, B, C] 整理坐标，再减去临时原点。
    ordered_coords = torch.stack((pos_a, pos_b, pos_c), dim=1)
    relative_coords = ordered_coords - origin.unsqueeze(1)

    # 分别求每颗原子沿局部横轴、纵轴的位置。
    x = (relative_coords * e_x.unsqueeze(1)).sum(dim=-1)
    y = (relative_coords * e_y.unsqueeze(1)).sum(dim=-1)
    projected = torch.stack((x, y), dim=-1)

    # 9.7 三颗原子的二维坐标取平均，得到三角形重心。
    centroid = projected.mean(dim=1, keepdim=True)
    centered = projected - centroid

    return projected, centered


# ============================================================
# 10. 网格距离：片段原子的二维坐标 → 到各网格点的距离平方
#
# 这里不再改变原子坐标，只计算原子与网格点之间的距离。
# ============================================================

def compute_fragment_grid_distances(
    fragment_coords: torch.Tensor,
    grid_points: torch.Tensor,
) -> torch.Tensor:
    """
    输入：
        fragment_coords: [片段数, 原子数, 2]，原子的二维坐标。
        grid_points: [网格点数, 2]，网格的二维采样坐标。

    输出：
        dist_sq: [片段数, 原子数, 网格点数]，距离平方。
    """
    # 10.1 每个原子分别减去每个网格点，得到 [Δx, Δy]。
    diff = (
        fragment_coords.unsqueeze(2)
        - grid_points.unsqueeze(0).unsqueeze(0)
    )

    # 10.2 两个坐标差分别平方，再相加：d² = Δx² + Δy²。
    dist_sq = diff.square().sum(dim=-1)

    return dist_sq

# ============================================================
# 11. 元素对应：原子序数使用与二维坐标相同的排列顺序
#
# order 里的数字是原始片段内的原子编号。
# 排序后，原子序数与 centered 中的 [A, B, C] 坐标逐个对应。
# ============================================================

def reorder_triplet_atomic_numbers(
    atomic_numbers: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor:
    """
    输入：
        atomic_numbers: [T, 3]，原始坐标顺序对应的原子序数。
        order: [T, 3]，[左端点, 右端点, 第三点] 的原始编号。

    输出：
        ordered_numbers: [T, 3]，与排序后坐标对应的原子序数。
    """
    ordered_numbers = torch.gather(
        atomic_numbers,
        dim=1,
        index=order,
    )

    return ordered_numbers

# ============================================================
# 12. 三体描述符入口：把三体几何和网格计算串起来
#
# 输入坐标和原子序数必须使用相同的原始排列顺序。
# 函数内部负责排序，并同时调整坐标和原子序数。
# ============================================================

def compute_triplet_features(
    coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    grid: DIEPGrid,
    sigma: float = 1.0,
    softening_epsilon: float = 0.5,
    use_effective_charge: bool = True,
    mode: str = "grid",
) -> torch.Tensor:
    """
    输入：
        coords: [T, 3, 3]，每个三体的原始三维坐标。
        atomic_numbers: [T, 3]，与原始坐标逐个对应的原子序数。
        grid: 已经创建好的二维网格。

    输出：
        grid 模式：[T, P]，每个三体一个网格贡献向量。
        sum 模式： [T, 1]，每个三体一个近似积分值。
    """
    # 12.1 找到最长边的两个端点编号。
    _, longest_pair = find_triplet_longest_edge(coords)

    # 12.2 确定 [左端点, 右端点, 第三点] 的原始编号。
    order = get_triplet_vertex_order(
        coords,
        longest_pair,
    )

    # 12.3 投影到二维，只取重心已经移到原点的最终坐标。
    _, centered = project_triplet_to_2d(
        coords,
        order,
    )

    # 12.4 三颗原子到所有网格点的距离平方。
    dist_sq = compute_fragment_grid_distances(
        centered,
        grid.points,
    )

    # 12.5 原子序数按同一个 order 排列，与上面的距离数据对应。
    ordered_numbers = reorder_triplet_atomic_numbers(
        atomic_numbers,
        order,
    )

    # 12.6 计算每个网格点上，三颗原子共同贡献的总密度。
    _, rho_total = compute_fragment_density(
        dist_sq,
        sigma=sigma,
    )

    # 12.7 计算每个网格点上，三颗原子共同贡献的总势因子。
    _, potential_total = compute_fragment_potential(
        dist_sq,
        ordered_numbers,
        softening_epsilon=softening_epsilon,
        use_effective_charge=use_effective_charge,
    )

    # 12.8 密度 × 势因子 × 面积权重，按指定模式返回描述符。
    return compute_grid_features(
        rho_total,
        potential_total,
        grid.delta_area,
        mode=mode,
    )



if __name__ == "__main__":
    grid = DIEPGrid(
        half_length=1.0,
        spacing=1.0,
        device=torch.device("cpu"),
    )

    print("axis:", grid.axis)
    print("points:\n", grid.points)
    print("shape:", grid.points.shape)
    print("delta_area:", grid.delta_area)
        # 测试二体几何：一条键长为 2 的边
    bond_dist = torch.tensor(
        [2.0],
        dtype=grid.points.dtype,
        device=grid.points.device,
    )

    # 把键长和网格坐标交给函数，接收计算结果
    fragment_coords, dist_sq = compute_bond_grid_geometry(
        bond_dist,
        grid.points,
    )

    print("\n两个原子的标准坐标：")
    print(fragment_coords[0])

    print("\n两个原子到所有网格点的距离平方：")
    print(dist_sq[0])

    print("\ndist_sq 的形状：")
    print(dist_sq.shape)

    print("\n原子 j 到第 0 个网格点的距离平方，预期为 5：")
    print(dist_sq[0, 1, 0])
        # 测试高斯密度：使用前面已经计算好的 dist_sq
    rho_atoms, rho_total = compute_fragment_density(
        dist_sq,
        sigma=1.0,
    )

    print("\n各原子密度贡献的形状：", rho_atoms.shape)
    print("片段总密度的形状：", rho_total.shape)

    # 当前网格第 4 个点是中心点 (0, 0)
    print("\n检查的网格点：", grid.points[4])
    print("原子 i 的密度贡献：", rho_atoms[0, 0, 4])
    print("原子 j 的密度贡献：", rho_atoms[0, 1, 4])
    print("两个原子相加后的密度：", rho_total[0, 4])

        # 测试：一个由两个 Li 原子组成的片段
    atomic_numbers = torch.tensor(
        [[3.0, 3.0]],
        dtype=dist_sq.dtype,
        device=dist_sq.device,
    )

    potential_atoms, potential_total = compute_fragment_potential(
        dist_sq,
        atomic_numbers,
    )

    print("\n各原子势贡献的形状：", potential_atoms.shape)
    print("片段总势因子的形状：", potential_total.shape)

    print("中心网格点：", grid.points[4])
    print("原子 i 的势贡献：", potential_atoms[0, 0, 4])
    print("原子 j 的势贡献：", potential_atoms[0, 1, 4])
    print("中心点的总势因子：", potential_total[0, 4])
        # grid 模式：每条边保留一个网格贡献向量
    bond_features = compute_grid_features(
        rho_total,
        potential_total,
        grid.delta_area,
        mode="grid",
    )

    # sum 模式：每条边得到一个近似积分值
    bond_integral = compute_grid_features(
        rho_total,
        potential_total,
        grid.delta_area,
        mode="sum",
    )

    print("\n二体 DIEP 向量：")
    print(bond_features)

    print("grid 输出形状：", bond_features.shape)
    print("中心网格点的贡献：", bond_features[0, 4])

    print("\n对所有网格贡献求和：", bond_integral)
    print("sum 输出形状：", bond_integral.shape)
        # 检查统一入口是否得到与前面分步计算相同的结果
    features = compute_bond_features(
        bond_dist,
        atomic_numbers,
        grid,
    )

    print("\n统一入口的输出：", features)
    print(
        "与分步计算一致：",
        torch.allclose(features, bond_features),
    )
        # 测试三体：直接给出三颗原子的三维坐标
    triplet_coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 1.0, 2.0],
            ]
        ],
        dtype=grid.points.dtype,
        device=grid.points.device,
    )

    lengths, longest_pair = find_triplet_longest_edge(
        triplet_coords
    )

    print("\n三体坐标：")
    print(triplet_coords)

    print("三条边长，顺序为 0-1、1-2、0-2：")
    print(lengths)

    print("最长边的两个端点编号：")
    print(longest_pair)
        # 检查原来的三体：原子 0 到第三点更近，应放左边。
    order = get_triplet_vertex_order(
        triplet_coords,
        longest_pair,
    )

    print("\n未来的 [左端点, 右端点, 第三点] 原始编号：")
    print(order)

    # 再验证交换分支：
    # 只交换输入数组中的前两颗原子，三角形本身没有改变。
    reversed_coords = triplet_coords[:, [1, 0, 2], :]

    _, reversed_pair = find_triplet_longest_edge(reversed_coords)
    reversed_order = get_triplet_vertex_order(
        reversed_coords,
        reversed_pair,
    )

    print("交换输入顺序后，应该把新编号 1 放左边：")
    print(reversed_order)

        # 投影原来的三体，注意与原来的 order 配套使用。
    projected, centered = project_triplet_to_2d(
        triplet_coords,
        order,
    )

    print("\n以最长边中点为原点的二维坐标：")
    print(projected)

    print("\n重心移到原点后的二维坐标：")
    print(centered)

    print("最终坐标形状：", centered.shape)
    print("最终三颗原子的平均坐标：", centered.mean(dim=1))
        # 使用已经居中的三体二维坐标计算网格距离。
    triplet_dist_sq = compute_fragment_grid_distances(
        centered,
        grid.points,
    )

    print("\n三体到网格点的距离平方形状：")
    print(triplet_dist_sq.shape)

    # 第 4 个网格点为 (0, 0)，检查三个原子到它的距离平方。
    print("检查的网格点：", grid.points[4])
    print("A、B、C 到这个点的距离平方：")
    print(triplet_dist_sq[0, :, 4])

        # 三颗原子的原子序数，必须先与原始 triplet_coords 顺序对应。
    # 当前测试：原子 0 是 Li，原子 1 是 O，原子 2 是 H。
    triplet_numbers = torch.tensor(
        [[3.0, 8.0, 1.0]],
        dtype=triplet_dist_sq.dtype,
        device=triplet_dist_sq.device,
    )

    # A. 与二维坐标使用同一个 order，保证元素和坐标一一对应。
    ordered_numbers = reorder_triplet_atomic_numbers(
        triplet_numbers,
        order,
    )

    # B. 复用高斯密度函数：对三个原子的贡献求和。
    _, triplet_rho_total = compute_fragment_density(
        triplet_dist_sq,
        sigma=1.0,
    )

    # C. 复用势因子函数：使用排序后的原子序数。
    _, triplet_potential_total = compute_fragment_potential(
        triplet_dist_sq,
        ordered_numbers,
    )

    # D. 逐点计算：总密度 × 总势因子 × 面积权重。
    triplet_features = compute_grid_features(
        triplet_rho_total,
        triplet_potential_total,
        grid.delta_area,
        mode="grid",
    )

    print("\n原始原子序数：", triplet_numbers)
    print("原子排列顺序：", order)
    print("排序后原子序数：", ordered_numbers)

    print("\n三体总密度形状：", triplet_rho_total.shape)
    print("三体总势因子形状：", triplet_potential_total.shape)

    print("\n三体 DIEP 向量：")
    print(triplet_features)
    print("三体 DIEP 向量形状：", triplet_features.shape)

    print("\n检查中心网格点：", grid.points[4])
    print("这个点的密度：", triplet_rho_total[0, 4])
    print("这个点的势因子：", triplet_potential_total[0, 4])
    print("这个点的加权贡献：", triplet_features[0, 4])


        # 统一入口：从原始三维坐标开始，一次完成整个三体计算。
    triplet_features_combined = compute_triplet_features(
        triplet_coords,
        triplet_numbers,
        grid,
    )

    print("\n三体统一入口的输出：")
    print(triplet_features_combined)

    print("与之前的三体分步计算一致：")
    print(torch.allclose(
        triplet_features_combined,
        triplet_features,
    ))
