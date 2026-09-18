"""Three-body (bond-angle) interaction terms.

Implements :class:`ThreeBodyInteractions`, the M3GNet/CHGNet angular
update that consumes a line graph (whose edges represent triplets of
neighbours sharing a central atom) and produces an updated edge feature
on the original graph. The caller supplies the triplet basis (DIEP grid
features in this model) and the per-bond cutoff weights. The layer combines
each basis with its neighbour's features, multiplies the two cutoff weights,
and scatters the contribution to its explicitly identified original bond.

Line-graph construction is handled by :mod:`matgl.graph._compute` upstream.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import matgl
from matgl.utils.maths import _block_repeat, scatter_sum


class ThreeBodyInteractions(nn.Module):
    """Three-body bond update.

    ``forward`` takes only tensors. The caller is responsible for unpacking
    the parent-graph edges and the line-graph adjacency / per-bond triple
    counts before calling this layer.
    """

    def __init__(self, update_network_atom: nn.Module, update_network_bond: nn.Module, **kwargs):
        """Initialize ThreeBodyInteractions.

        Args:
            update_network_atom: MLP for node features in Eq.2
            update_network_bond: Gated-MLP for edge features in Eq.3
            **kwargs: Kwargs pass-through to nn.Module.__init__().
        """
        super().__init__(**kwargs)
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(
        self,
        edge_dst_atom: torch.Tensor,
        line_edge_index: torch.Tensor,
        n_triple_ij: torch.Tensor,
        num_bonds: int,
        three_basis: torch.Tensor,
        three_cutoff: torch.Tensor,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        """Forward function for ThreeBodyInteractions.

        Args:
            edge_dst_atom: For each bond ``b`` in the parent graph, the index
                of the destination atom of that bond. Shape ``(num_bonds,)``.
            line_edge_index: Triplet bond pairs as ``(2, num_triples)``, with
                both rows indexing bonds in the original parent graph.
                Row 0 is the receiving bond; row 1 supplies the neighbour atom.
            n_triple_ij: For each bond, the number of triples it participates
                in as the "central" bond. Shape ``(num_bonds,)``. Retained for
                API compatibility; aggregation uses the explicit receiving
                bond indices, so triplet order does not affect the result.
            num_bonds: Total number of bonds in the parent graph (i.e. the
                ``dim_size`` for the per-bond scatter).
            three_basis: three body basis expansion of shape
                ``(num_triples, basis_dim)``.
            three_cutoff: per-bond cutoff weights, shape ``(num_bonds,)``.
            node_feat: node features.
            edge_feat: edge features (one row per bond).
        """
        # Get the indices of the end atoms for each bond in the line graph
        end_atom_indices = edge_dst_atom[line_edge_index[1]].to(matgl.int_th)

        # Update node features using the atom update network
        updated_atoms = self.update_network_atom(node_feat)

        # Gather updated atom features for the end atoms
        end_atom_features = updated_atoms[end_atom_indices]

        # Compute the basis term
        basis = three_basis * end_atom_features

        # Reshape and compute weights based on the three-cutoff tensor
        three_cutoff = three_cutoff.unsqueeze(1)
        edge_indices = line_edge_index.t().contiguous()
        weights = three_cutoff[edge_indices].view(-1, 2)
        weights = weights.prod(dim=-1)

        # Compute the weighted basis
        basis = basis * weights[:, None]

        # 原写法：根据三体数量生成聚合目标，依赖计数与三体排列顺序完全一致。
        # segment_ids = get_segment_indices_from_n(n_triple_ij)

        # 新写法：每个三体直接使用自己的接收边编号。
        # 调用方已把线图的小表编号转换为原图边编号；这里按原图编号聚合。
        # 即使打乱三体排列顺序，也会加回同一条原图边。
        segment_ids = line_edge_index[0].long()
        new_bonds = scatter_sum(
            basis.to(matgl.float_th),
            segment_ids=segment_ids,
            num_segments=num_bonds,
            dim=0,
        )

        # If no new bonds are generated, return the original edge features
        if new_bonds.shape[0] == 0:
            return edge_feat

        # Update edge features using the bond update network
        return edge_feat + self.update_network_bond(new_bonds)


def combine_sbf_shf(sbf, shf, max_n: int, max_l: int, use_phi: bool):
    """Combine the spherical Bessel function and the spherical Harmonics function.

    For the spherical Bessel function, the column is ordered by
        [n=[0, ..., max_n-1], n=[0, ..., max_n-1], ...], max_l blocks,

    For the spherical Harmonics function, the column is ordered by
        [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], ...] max_l blocks, and each
        block has 2*l + 1
        if use_phi is False, then the columns become
        [m=[0], m=[0], ...] max_l columns

    Args:
        sbf: torch.Tensor spherical bessel function results
        shf: torch.Tensor spherical harmonics function results
        max_n: int, max number of n
        max_l: int, max number of l
        use_phi: whether to use phi
    Returns:
    """
    if sbf.size()[0] == 0:
        return sbf

    if not use_phi:
        repeats_sbf = torch.tensor([1] * max_l * max_n)
        block_size = [1] * max_l
    else:
        # [1, 1, 1, ..., 1, 3, 3, 3, ..., 3, ...]
        repeats_sbf = np.repeat(2 * torch.arange(max_l) + 1, repeats=max_n)  # type:ignore[assignment]
        # tf.repeat(2 * tf.range(max_l) + 1, repeats=max_n)
        block_size = 2 * torch.arange(max_l) + 1  # type: ignore
        # 2 * tf.range(max_l) + 1
    repeats_sbf = repeats_sbf.to(sbf.device)
    expanded_sbf = torch.repeat_interleave(sbf, repeats_sbf, 1)
    expanded_shf = _block_repeat(shf, block_size=block_size, repeats=[max_n] * max_l)
    shape = max_n * max_l
    if use_phi:
        shape *= max_l
    return torch.reshape(expanded_sbf * expanded_shf, [-1, shape])
