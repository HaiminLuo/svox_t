#  Copyright 2021 PlenOctree Authors.
#  Copyright 2022 Artemis Authors.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""
Sparse voxel N^3 tree
"""

import os.path as osp
import torch
import numpy as np
from torch import nn, autograd
from svox_t.helpers import N3TreeView, DataFormat, _get_c_extension
from warnings import warn

_C = _get_c_extension()

class _QueryVerticalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, tree_spec, indices):
        out, node_ids, data_ids, leaf_node = _C.query_vertical(tree_spec, indices)

        ctx.mark_non_differentiable(node_ids)
        ctx.mark_non_differentiable(data_ids)
        ctx.mark_non_differentiable(leaf_node)
        ctx.tree_spec = tree_spec
        ctx.save_for_backward(indices)
        return out, node_ids, data_ids, leaf_node

    @staticmethod
    def backward(ctx, grad_out, dummy):
        if ctx.needs_input_grad[0]:
            return _C.query_vertical_backward(ctx.tree_spec,
                         ctx.saved_tensors[0],
                         grad_out.contiguous()), None, None
        return None, None, None

class _WarpVerticalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, transformation_matrix, coordinates, skinning_weights, joint_index): # transformation_matrix, coordinates, skinning_weights, joint_index
        vertices, matrices = _C.warp_vertices(transformation_matrix, coordinates, skinning_weights, joint_index) 

        ctx.save_for_backward(transformation_matrix, coordinates, skinning_weights, joint_index)

        return vertices, matrices

    @staticmethod
    def backward(ctx, vertices_grad_out, matrices_grad_out):
        if ctx.needs_input_grad[0]:
            grad_indices, grad_matrices, grad_skinning_weights = _C.warp_vertices_backward(ctx.saved_tensors[0], 
                        ctx.saved_tensors[1], ctx.saved_tensors[2], ctx.saved_tensors[3],
                        vertices_grad_out.contiguous(),
                        matrices_grad_out.contiguous())
            return grad_matrices, grad_indices, grad_skinning_weights, None
        return None, None, None, None


class N3Tree(nn.Module):
    """
    PyTorch :math:`N^3`-tree library with CUDA acceleration.
    By :math:`N^3`-tree we mean a 3D tree with branching factor N at each interior node,
    where :math:`N=2` is the familiar octree.

.. warning::
    `nn.Parameters` can change size, which
    makes current optimizers invalid. If any refine() or
    shrink_to_fit() call returns True,
    please re-make any optimizers
    """
    def __init__(self, N=2, data_dim=4, depth_limit=10,
            init_reserve=1, init_refine=0, geom_resize_fact=1.5,
            radius=0.5, center=[0.5, 0.5, 0.5],
            data_format="RGBA",
            extra_data=None,
            map_location="cpu"):
        """
        Construct N^3 Tree

        :param N: int branching factor N
        :param data_dim: int size of data stored at each leaf
        :param depth_limit: int maximum depth of tree to stop branching/refining
        :param init_reserve: int amount of nodes to reserve initially
        :param init_refine: int number of times to refine entire tree initially
        :param geom_resize_fact: float geometric resizing factor
        :param radius: float or list, 1/2 side length of cube (possibly in each dim)
        :param center: list center of space
        :param data_format: a string to indicate the data format
        :param extra_data: extra data to include with tree
        :param map_location: str device to put data

        """
        super().__init__()
        assert N >= 2
        assert depth_limit >= 0
        self.N : int = N
        self.data_dim : int = data_dim

        if init_refine > 0:
            for i in range(1, init_refine + 1):
                init_reserve += (N ** i) ** 3
        self.register_parameter("features", nn.Parameter(
            torch.zeros(init_reserve, data_dim, device=map_location)))
        self.register_buffer("data", 
            torch.ones(init_reserve, N, N, N, 1, dtype=torch.int32, device=map_location) * int(1e10))
        self.register_buffer("child", torch.zeros(
            init_reserve, N, N, N, dtype=torch.int32, device=map_location))
        self.register_buffer("parent_depth", torch.zeros(
            init_reserve, 2, dtype=torch.int32, device=map_location))

        self.register_buffer("_n_internal", torch.tensor(1, device=map_location))
        self.register_buffer("_n_free", torch.tensor(0, device=map_location))

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        radius = torch.tensor(radius, dtype=torch.float32, device=map_location)
        center = torch.tensor(center, dtype=torch.float32, device=map_location)

        self.register_buffer("invradius", 0.5 / radius)
        self.register_buffer("offset", 0.5 * (1.0 - center / radius))

        self.depth_limit = depth_limit
        self.geom_resize_fact = geom_resize_fact
        self.data_format = DataFormat(data_format) if data_format is not None else None

        if extra_data is not None:
            assert isinstance(extra_data, torch.Tensor)
            self.register_buffer("extra_data", extra_data.to(device=map_location))
        else:
            self.extra_data = None

        self._ver = 0
        self._invalidate()
        self._lock_tree_structure = False
        self._weight_accum = None

        self.filled = self._n_internal.item()

        self.refine(repeats=init_refine)

    def construct_tree(self, indices):
        _C.construct_tree(self._spec(self.features), indices)

    # Main accesors
    def set(self, indices, values, cuda=True):
        """
        Set tree values,

        :param indices: torch.Tensor (Q, 3)
        :param values: torch.Tensor (Q, K)
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.

.. warning::
        Beware: If multiple indices point to same leaf node,
        only one of them will be taken
        """
        assert len(indices.shape) == 2
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert not values.requires_grad  # Grad wrt values not supported
        indices = indices.to(device=self.data.device)
        values = values.to(device=self.data.device)

        if not cuda or _C is None or not self.data.is_cuda:
            warn("Using slow assignment")
            indices = self.world2tree(indices)

            n_queries, _ = indices.shape
            indices.clamp_(0.0, 1.0 - 1e-10)
            ind = indices.clone()

            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            remain_mask = torch.ones(n_queries, dtype=torch.bool, device=indices.device)
            while remain_mask.any():
                ind_floor = torch.floor(ind[remain_mask] * self.N)
                ind_floor.clamp_max_(self.N - 1)
                sel = (node_ids[remain_mask], *(ind_floor.long().T),)

                deltas = self.child[sel]
                vals = self.data.data[sel]

                nonterm_partial_mask = deltas != 0
                nonterm_mask = torch.zeros(n_queries, dtype=torch.bool, device=indices.device)
                nonterm_mask[remain_mask] = nonterm_partial_mask

                node_ids[remain_mask] += deltas
                ind[remain_mask] = ind[remain_mask] * self.N - ind_floor

                term_mask = remain_mask & ~nonterm_mask
                vals[~nonterm_partial_mask] = values[term_mask]
                self.data.data[sel] = vals

                remain_mask &= nonterm_mask
        else:
            _C.assign_vertical(self._spec(self.features), indices, values)

    def forward(self, features, indices, cuda=True, want_node_ids=False, world=True, want_data_ids=False, want_leaf_node=False):
        """
        Get tree values. Differentiable.

        :param indices: :math:`(Q, 3)` the points
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.
        :param want_node_ids: if true, returns node ID for each query.
        :param world: use world space instead of :math:`[0,1]^3`, default True

        :return: (Q, data_dim), [(Q)]

        """
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert len(indices.shape) == 2

        if not cuda or _C is None or not self.data.is_cuda:
            if not want_node_ids:
                warn("Using slow query")
            if world:
                indices = self.world2tree(indices)

            indices.clamp_(0.0, 1.0 - 1e-10)

            n_queries, _ = indices.shape
            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            result = torch.empty((n_queries, self.data_dim), dtype=torch.float32,
                                  device=indices.device)
            remain_indices = torch.arange(n_queries, dtype=torch.long, device=indices.device)
            ind = indices.clone()

            if want_node_ids:
                subidx = torch.zeros((n_queries, 3), dtype=torch.long, device=indices.device)

            while remain_indices.numel():
                ind *= self.N
                ind_floor = torch.floor(ind)
                ind_floor.clamp_max_(self.N - 1)
                ind -= ind_floor

                sel = (node_ids[remain_indices], *(ind_floor.long().T),)

                deltas = self.child[sel]

                term_mask = deltas == 0
                term_indices = remain_indices[term_mask]

                vals = self.data[sel]
                result[term_indices] = vals[term_mask]
                if want_node_ids:
                    subidx[term_indices] = ind_floor.to(torch.long)[term_mask]

                node_ids[remain_indices] += deltas
                remain_indices = remain_indices[~term_mask]
                ind = ind[~term_mask]

            if want_node_ids:
                txyz = torch.cat([node_ids[:, None], subidx], axis=-1)
                return result, self._pack_index(txyz)

            return result
        else:
            result, node_ids, data_ids, leaf_node = _QueryVerticalFunction.apply(
                                            features, self._spec(features, world=world), indices)
            ret = [result, node_ids] if want_node_ids else result 
            if want_data_ids:
                ret.append(data_ids)
            if want_leaf_node:
                ret.append(leaf_node)
            return ret

    # Special features
    def snap(self, indices):
        """
        Snap indices to lowest corner of corresponding leaf voxel

        :param indices: (B, 3) indices to snap

        :return: (B, 3)

        """
        return self[indices].corners

    def partial(self, data_sel=None, device=None):
        """
        Get partial tree with some of the data dimensions (channels)
        E.g. tree.partial(-1) to get tree with data_dim 1 of last channel only

        :param data_sel: data channel selector, default is all channels
        :param device: where to put result tree

        :return: partial N3Tree (copy)
        """
        if device is None:
            device = self.data.device
        if data_sel is None:
            new_data_dim = self.data_dim
            sel_indices = None
        else:
            sel_indices = torch.arange(self.data_dim)[data_sel]
            if sel_indices.ndim == 0:
                sel_indices = sel_indices.unsqueeze(0)
            new_data_dim = sel_indices.numel()
        t2 = N3Tree(N=self.N, data_dim=new_data_dim,
                depth_limit=self.depth_limit,
                geom_resize_fact=self.geom_resize_fact,
                map_location=device)
        def copy_to_device(x):
            return torch.empty(x.shape, dtype=x.dtype, device=device).copy_(x)
        t2.invradius = copy_to_device(self.invradius)
        t2.offset = copy_to_device(self.offset)
        t2.child = copy_to_device(self.child)
        t2.parent_depth = copy_to_device(self.parent_depth)
        t2._n_internal = copy_to_device(self._n_internal)
        t2._n_free = copy_to_device(self._n_free)
        if self.extra_data is not None:
            t2.extra_data = copy_to_device(self.extra_data)
        else:
            t2.extra_data = None
        t2.data_format = self.data_format
        if data_sel is None:
            t2.data = nn.Parameter(copy_to_device(self.data.data))
        else:
            t2.data = nn.Parameter(copy_to_device(self.data.data[..., sel_indices].contiguous()))
        return t2

    def clone(self, device=None):
        """
        Deep copy the tree

        :param device: device of output tree (could e.g. copy cuda tree to cpu)

        """
        return self.partial(device=device)

    # 'Frontier' operations (node merging/pruning)
    def merge(self, frontier_sel=None, op=torch.mean):
        """
        Merge leaves into selected 'frontier' nodes
        (i.e., nodes for which all children are leaves).
        Use shrink_to_fit() to recover memory freed.

        :param frontier_sel: selector (int, mask, list of indices etc)
                             for frontier nodes. In same order as reduce_frontier().
                             Default all nodes.
                             *Typical use*: use reduce_frontier(op=...) to determine
                             conditions for merge, then pass
                             mask or indices to merge().
        :param op: reduction to combine child leaves into node.
                   E.g. torch.max, torch.mean.
                   Should take a positional argument :code:`x` (B, N, data_dim) and
                   a named parameter :code:`dim` (always 1),
                   and return a matrix of (B, your_out_dim).
                   If a tuple is returned, uses first result.
        """
        if self.n_internal - self._n_free.item() <= 1:
            raise RuntimeError("Cannot merge root node")
        nid = self._frontier[frontier_sel]
        if nid.numel() == 0:
            return False
        if nid.ndim == 0:
            nid = nid.reshape(1)
        data = self.data.data[nid]
        reduced_vals = op(data.view(-1, self.N ** 3, self.data_dim), dim=1)
        if isinstance(reduced_vals, tuple):
            # Allow torch.max, torch.min, etc
            reduced_vals = reduced_vals[0]
        parent_sel = (*self._unpack_index(self.parent_depth[nid, 0]).long().T,)
        self.data.data[parent_sel] = reduced_vals
        self.child[parent_sel] = 0
        self.parent_depth[nid] = -1
        self._n_free += nid.shape[0]
        self._invalidate()
        return True

    def reduce_frontier(self, op=torch.mean, dim=None, grad=False):
        """
        Reduce child leaf values for each 'frontier' node
        (i.e., nodes for which all children are leaves).

        :param op: reduction to combine child leaves into node.
                   E.g. torch.max, torch.mean.
                   Should take a positional argument :code:`x`
                   (B, N, in_dim <= data_dim) and
                   a named parameter :code:`dim` (always 1),
                   and return a matrix of (B, your_out_dim).
        :param dim: dimension(s) of data to return, e.g. -1 returns
                    last data dimension for all 'frontier' nodes
        :param grad: if True, returns a tensor differentiable wrt tree data.
                      Default False.

        :return: reduced tensor
        """
        nid = self._frontier
        if grad:
            data = self.data[nid]
        else:
            data = self.data.data[nid]
        data = data.view(-1, self.N ** 3, self.data_dim)
        if dim is None:
            return op(data, dim=1)
        else:
            return op(data[..., dim], dim=1)

    def max_frontier(self, dim=None, grad=False):
        """
        Takes max over child leaf values for each 'frontier' node
        (i.e., nodes for which all children are leaves).
        This is simply reduce_frontier with torch.max
        operation, taking the returned values and discarding the
        argmax part.

        :param dim: dimension(s) of data to return, e.g. -1 returns
                    last data dimension for all 'frontier' nodes
        :param grad: if True, returns a tensor differentiable wrt tree data.
                      Default False.

        :return: reduced tensor
        """
        return self.reduce_frontier(op=lambda x, dim: torch.max(x, dim=dim)[0],
                grad=grad, dim=dim)

    def diam_frontier(self, dim=None, grad=False, scale=1.0):
        """
        Takes diameter over child leaf values for each 'frontier' node
        (i.e., nodes for which all children are leaves).

        :param dim: dimension(s) of data to return, e.g. -1 returns
                    last data dimension for all 'frontier' nodes
        :param grad: if True, returns a tensor differentiable wrt tree data.
                      Default False.

        :return: reduced tensor
        """
        def diam_func(x, dim):
            # (B, N3, in_dim)
            if x.ndim == 2:
                x = x[:, :, None]
            N3 = x.shape[1]
            diam = torch.zeros(x.shape[:-2], device=x.device)
            for offset in range(N3):
                end_idx = -offset if offset > 0 else N3
                delta = (x[:, offset:] - x[:, :end_idx]) * scale
                n1 = torch.norm(delta, dim=-1).max(dim=-1)[0]
                if offset:
                    delta = (x[:, :offset] - x[:, end_idx:]) * scale
                    n2 = torch.norm(delta, dim=-1).max(dim=-1)[0]
                    n1 = torch.max(n1, n2)
                diam = torch.max(diam, n1)
            return diam

        return self.reduce_frontier(op=diam_func,
                grad=grad, dim=dim)


    @property
    def _frontier(self):
        """
        Get the nodes immediately above leaves (internal use)

        :return: node indices (first dim of self.data)
        """
        if self._last_frontier is None:
            node_selector = (self.child[ :self.n_internal] == 0).reshape(
                    self.n_internal, -1).all(dim=1)
            node_selector &= self.parent_depth[:, 0] != -1
            self._last_frontier = node_selector.nonzero(as_tuple=False).reshape(-1)
        return self._last_frontier


    # Leaf refinement & memory management methods
    #@profile
    def refine(self, repeats=1, sel=None, leaf_node=None, node_id=None):
        """
        Refine each selected leaf node, respecting depth_limit.

        :param repeats: int number of times to repeat refinement
        :param sel: (N, 4) node selector. Default selects all leaves.

        :return: True iff N3Tree.data parameter was resized, requiring
                 optimizer reinitialization if you're using an optimizer

.. warning::
    The parameter :code:`tree.data` can change due to refinement. If any refine() call returns True, please re-make any optimizers
    using tree.params().

.. warning::
    The selector :code:`sel` is assumed to contain unique leaf indices. If there are duplicates
    memory will be wasted. We do not dedup here for efficiency reasons.

        """
        if self._lock_tree_structure:
            raise RuntimeError("Tree locked")
        with torch.no_grad():
            resized = False
            for repeat_id in range(repeats):
                filled = self.filled
                if sel is None:
                    # Default all leaves
                    sel = (*self._all_leaves().T,)
                    depths = self.parent_depth[sel[0], 1]
                    # Filter by depth & leaves
                    good_mask = (depths < self.depth_limit) & (self.child[sel] == 0)
                    sel = [t[good_mask] for t in sel]

                if leaf_node is None:
                    leaf_node =  torch.stack(sel, dim=-1).to(device=self.data.device)
                    
                num_nc = len(sel[0])
                if num_nc == 0:
                    # Nothing to do
                    return False
                new_filled = filled + num_nc

                cap_needed = new_filled - self.capacity
                if cap_needed > 0:
                    self._resize_add_cap(cap_needed)
                    resized = True

                new_idxs = torch.arange(filled, filled + num_nc,
                        device=leaf_node.device, dtype=self.child.dtype) # NNC

                self.child[sel] = new_idxs - leaf_node[:, 0].to(torch.int32)
                self.data.data[filled:new_filled] = self.data.data[
                        sel][:, None, None, None]
                self.parent_depth[filled:new_filled, 0] = self._pack_index(leaf_node) if node_id is None else node_id # parent
                self.parent_depth[filled:new_filled, 1] = self.parent_depth[
                        leaf_node[:, 0], 1] + 1  # depth

                if repeat_id < repeats - 1:
                    # Infer new selector
                    t1 = torch.arange(filled, new_filled,
                            device=self.data.device).repeat_interleave(self.N ** 3)
                    rangen = torch.arange(self.N, device=self.data.device)
                    t2 = rangen.repeat_interleave(self.N ** 2).repeat(
                            new_filled - filled)
                    t3 = rangen.repeat_interleave(self.N).repeat(
                            (new_filled - filled) * self.N)
                    t4 = rangen.repeat((new_filled - filled) * self.N ** 2)
                    sel = (t1, t2, t3, t4)
                self._n_internal += num_nc
                self.filled += num_nc
        if repeats > 0:
            self._invalidate()
        return resized

    def _refine_at(self, intnode_idx, xyzi):
        """
        Advanced: refine specific leaf node. Mostly for testing purposes.

        :param intnode_idx: index of internal node for identifying leaf
        :param xyzi: tuple of size 3 with each element in {0, ... N-1}
                    in xyz orde rto identify leaf within internal node

        """
        if self._lock_tree_structure:
            raise RuntimeError("Tree locked")
        assert min(xyzi) >= 0 and max(xyzi) < self.N
        if self.parent_depth[intnode_idx, 1] >= self.depth_limit:
            return

        xi, yi, zi = xyzi
        if self.child[intnode_idx, xi, yi, zi] != 0:
            # Already has child
            return

        resized = False
        filled = self.n_internal
        if filled >= self.capacity:
            self._resize_add_cap(1)
            resized = True

        self.child[filled] = 0
        self.child[intnode_idx, xi, yi, zi] = filled - intnode_idx
        depth = self.parent_depth[intnode_idx, 1] + 1
        self.parent_depth[filled, 0] = self._pack_index(torch.tensor(
            [[intnode_idx, xi, yi, zi]], dtype=torch.int32))[0]
        self.parent_depth[filled, 1] = depth
        self.data.data[filled, :, :, :] = self.data.data[intnode_idx, xi, yi, zi]
        self.data.data[intnode_idx, xi, yi, zi] = 0
        self._n_internal += 1
        self._invalidate()
        return resized

    def shrink_to_fit(self):
        """
        Shrink data & buffers to tightly needed fit tree data,
        possibly dealing with fragmentation caused by merging.
        This is called by the save() function.

.. warning::
        Will change the nn.Parameter size (data), breaking optimizer!
        """
        if self._lock_tree_structure:
            raise RuntimeError("Tree locked")
        n_int = self.n_internal
        n_free = self._n_free.item()
        new_cap = n_int - n_free
        if new_cap >= self.capacity:
            return False
        if n_free > 0:
            # Defragment
            free = self.parent_depth[:n_int, 0] == -1
            csum = torch.cumsum(free, dim=0)

            remain_ids = torch.arange(n_int, dtype=torch.long)[~free]
            remain_parents = (*self._unpack_index(
                self.parent_depth[remain_ids, 0]).long().T,)

            # Shift data over
            par_shift = csum[remain_parents[0]]
            self.child[remain_parents] -= csum[remain_ids] - par_shift
            self.parent_depth[remain_ids, 0] -= par_shift

            # Remake the data now
            self.data =self.data.data[remain_ids]
            self.child = self.child[remain_ids]
            self.parent_depth = self.parent_depth[remain_ids]
            self._n_internal.fill_(new_cap)
            self._n_free.zero_()
        else:
            # Direct resize
            self.data = self.data.data[:new_cap]
            self.child = self.child[:new_cap]
            self.parent_depth = self.parent_depth[:new_cap]
        self._invalidate()
        return True

    # Misc
    @property
    def n_leaves(self):
        return self._all_leaves().shape[0]

    @property
    def n_internal(self):
        return self._n_internal.item()

    @property
    def capacity(self):
        return self.parent_depth.shape[0]

    @property
    def max_depth(self):
        """
        Maximum tree depth - 1
        """
        return torch.max(self.depths).item()

    def accumulate_weights(self):
        """
        Begin weight accumulation

.. code-block:: python

        with tree.accumulate_weights() as accum:
            ...

        # (n_leaves) in same order as values etc.
        accum = accum()
        """
        return WeightAccumulator(self)

    # Persistence
    def save(self, path, shrink=True, compress=True):
        """
        Save to from npz file

        :param path: npz path
        :param shrink: if True (default), applies shrink_to_fit before saving
        :param compress: whether to compress the npz; may be slow

        """
        if shrink:
            self.shrink_to_fit()
        data = {
            "data_dim" : self.data_dim,
            "child" : self.child.cpu(),
            "parent_depth" : self.parent_depth.cpu(),
            "n_internal" : self._n_internal.cpu().item(),
            "n_free" : self._n_free.cpu().item(),
            "invradius3" : self.invradius.cpu(),
            "offset" : self.offset.cpu(),
            "depth_limit": self.depth_limit,
            "geom_resize_fact": self.geom_resize_fact,
            "data": self.data.data.cpu().numpy()  # save CPU Memory
        }
        if self.data_format is not None:
            data["data_format"] = repr(self.data_format)
        if self.extra_data is not None:
            data["extra_data"] = self.extra_data.cpu()
        if compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)

    @classmethod
    def load(cls, path, map_location='cpu'):
        """
        Load from npz file

        :param path: npz path
        :param map_location: device to put data

        """
        z = np.load(path)
        extra_data = torch.from_numpy(z['extra_data']).to(map_location) if \
                          'extra_data' in z.files else None
        tree = cls(extra_data=extra_data, map_location=map_location)

        tree.data_dim = int(z["data_dim"])
        tree.child = torch.from_numpy(z["child"]).to(map_location)
        tree.N = tree.child.shape[-1]
        tree.parent_depth = torch.from_numpy(z["parent_depth"]).to(map_location)
        tree._n_internal.fill_(z["n_internal"].item())
        tree.filled = tree._n_internal.item()
        if "invradius3" in z.files:
            tree.invradius = torch.from_numpy(z["invradius3"].astype(
                                np.float32)).to(map_location)
        else:
            tree.invradius.fill_(z["invradius"].item())
        tree.offset = torch.from_numpy(z["offset"].astype(np.float32)).to(map_location)
        tree.depth_limit = int(z["depth_limit"])
        tree.geom_resize_fact = float(z["geom_resize_fact"])
        tree.data.data = torch.from_numpy(z["data"]).to(torch.int).to(map_location)
        if 'n_free' in z.files:
            tree._n_free.fill_(z["n_free"].item())
        else:
            tree._n_free.zero_()
        tree.data_format = DataFormat(z['data_format'].item()) if \
                'data_format' in z.files else None
        # tree.extra_data.data = torch.from_numpy(z['extra_data']).to(map_location) if \
        #                   'extra_data' in z.files else None
        # if 'extra_data' in z.files:
        #     tree.register_buffer("extra_data", torch.from_numpy(z['extra_data']).to(map_location))
        # else:
        #     tree.extra_data = None
        return tree

    # Magic
    def __repr__(self):
        return (f"svox.N3Tree(N={self.N}, data_dim={self.data_dim}, " +
                f"depth_limit={self.depth_limit}, " +
                f"capacity:{self.n_internal - self._n_free.item()}/{self.capacity}, " +
                f"data_format:{self.data_format or 'RGBA'})")

    def __getitem__(self, key):
        """
        Get N3TreeView
        """
        return N3TreeView(self, key)

    def __setitem__(self, key, val):
        N3TreeView(self, key).set(val)

    def __iadd__(self, val):
        self[:] += val
        return self

    def __isub__(self, val):
        self[:] -= val
        return self

    def __imul__(self, val):
        self[:] *= val
        return self

    def __idiv__(self, val):
        self[:] /= val
        return self

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return torch.Size((self.n_leaves, self.data_dim))

    def size(self, dim):
        return self.data_dim if dim == 1 else self.n_leaves

    def numel(self):
        return self.data_dim * self.n_leaves

    def __len__(self):
        return self.n_leaves

    # Internal utils
    def _calc_corners(self, nodes, cuda=True):
        if _C is not None and cuda and self.data.is_cuda:
            return _C.calc_corners(self._spec(self.features), nodes.to(self.data.device))

        Q, _ = nodes.shape
        filled = self.n_internal

        curr = nodes.clone()
        mask = torch.ones(Q, device=curr.device, dtype=torch.bool)
        output = torch.zeros(Q, 3, device=curr.device, dtype=torch.float32)

        while True:
            output[mask] += curr[:, 1:]
            output[mask] /= self.N

            good_mask = curr[:, 0] != 0
            if not good_mask.any():
                break
            mask[mask.clone()] = good_mask

            curr = self._unpack_index(self.parent_depth[curr[good_mask, 0], 0].long())

        return output


    def _pack_index(self, txyz):
        return txyz[:, 0] * (self.N ** 3) + txyz[:, 1] * (self.N ** 2) + \
               txyz[:, 2] * self.N + txyz[:, 3]

    def _unpack_index(self, flat):
        t = []
        for i in range(3):
            t.append(flat % self.N)
            flat //= self.N
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)

    #@profile
    def _resize_add_cap(self, cap_needed):
        """
        Helper for increasing capacity
        """
        cap_needed = max(cap_needed, int(self.capacity * (self.geom_resize_fact - 1.0)))
        may_oom = self.capacity + cap_needed > 1e7  # My CPU Memory is limited
        if may_oom:
            # Potential OOM prevention hack
            self.data = self.data.cpu()
        self.data = torch.cat((self.data.data,
                        torch.ones((cap_needed, *self.data.data.shape[1:]), dtype=self.data.dtype,
                                device=self.data.device) * int(1e10)), dim=0)
        if may_oom:
            self.data = self.data.to(device=self.child.device)
        torch.cuda.synchronize()
        self.child = torch.cat((self.child,
                                torch.zeros((cap_needed, *self.child.shape[1:]),
                                   dtype=self.child.dtype,
                                   device=self.data.device)))
        self.parent_depth = torch.cat((self.parent_depth,
                                torch.zeros((cap_needed, *self.parent_depth.shape[1:]),
                                   dtype=self.parent_depth.dtype,
                                   device=self.data.device)))

    def _make_val_tensor(self, val):
        val_tensor = torch.tensor(val, dtype=torch.float32,
            device=self.data.device)
        while len(val_tensor.shape) < 2:
            val_tensor = val_tensor[None]
        if val_tensor.shape[-1] == 1:
            val_tensor = val_tensor.expand(-1, self.data_dim).contiguous()
        else:
            assert val_tensor.shape[-1] == self.data_dim
        return val_tensor

    def _all_leaves(self):
        if self._last_all_leaves is None:
            self._last_all_leaves = (self.child[
                :self.n_internal] == 0).nonzero(as_tuple=False).cpu()
        return self._last_all_leaves

    def world2tree(self, indices):
        """
        Scale world points to tree (:math:`[0,1]^3`)
        """
        return torch.addcmul(self.offset, indices, self.invradius)

    def tree2world(self, indices):
        """
        Scale tree points (:math:`[0,1]^3`) to world accoording to center/radius
        """
        return (indices  - self.offset) / self.invradius

    def _invalidate(self):
        self._ver += 1
        self._last_all_leaves = None
        self._last_frontier = None

    def _spec(self, features, joint_features = None, skinning_weights=None, joint_index=None, transformation_matrices=None, world=True):
        """
        Pack tree into a TreeSpec (for passing data to C++ extension)
        """
        tree_spec = _C.TreeSpec()
        tree_spec.features = features
        tree_spec.data = self.data
        tree_spec.child = self.child
        tree_spec.parent_depth = self.parent_depth
        tree_spec.extra_data = self.extra_data if self.extra_data is not None else \
                torch.empty((0, 0), device=self.data.device)
        tree_spec.offset = self.offset if world else torch.tensor(
                  [0.0, 0.0, 0.0], device=self.data.device)
        tree_spec.scaling = self.invradius if world else torch.tensor(
                  [1.0, 1.0, 1.0], device=self.data.device)
        tree_spec.n_internal = self.filled
        if hasattr(self, '_weight_accum'):
            tree_spec._weight_accum = self._weight_accum if \
                    self._weight_accum is not None else torch.empty(
                            0, device=self.data.device)

        tree_spec.joint_features = joint_features if joint_features is not None else torch.empty((0, 0), device=self.data.device)
        tree_spec.skinning_weights = skinning_weights if skinning_weights is not None else torch.empty((0, 0), device=self.data.device)
        tree_spec.joint_index = joint_index if joint_index is not None else torch.empty((0, 0), device=self.data.device).int()
        tree_spec.transformation_matrices = transformation_matrices if transformation_matrices is not None else torch.empty((0, 0, 0), device=self.data.device)
        
        return tree_spec


# Redirect functions to N3TreeView so you can do tree.depths instead of tree[:].depths
def _redirect_to_n3view():
    redir_props = ['depths', 'lengths', 'lengths_local', 'corners', 'corners_local',
                   'values', 'values_local']
    redir_funcs = ['sample', 'sample_local', 'aux',
            'normal_', 'clamp_', 'uniform_', 'relu_', 'sigmoid_', 'nan_to_num_']
    def redirect_func(redir_func):
        def redir_impl(self, *args, **kwargs):
            return getattr(self[:], redir_func)(*args, **kwargs)
        setattr(N3Tree, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
    def redirect_prop(redir_prop):
        def redir_impl(self, *args, **kwargs):
            return getattr(self[:], redir_prop)
        setattr(N3Tree, redir_prop, property(redir_impl))
    for redir_prop in redir_props:
        redirect_prop(redir_prop)
_redirect_to_n3view()

class WeightAccumulator():
    def __init__(self, tree):
        self.tree = tree

    def __enter__(self):
        self.tree._lock_tree_structure = True
        self.tree._weight_accum = torch.zeros(
                self.tree.child.shape, dtype=torch.float32,
                device=self.tree.data.device)
        self.weight_accum = self.tree._weight_accum
        return self

    def __exit__(self, type, value, traceback):
        self.tree._weight_accum = None
        self.tree._lock_tree_structure = False

    @property
    def value(self):
        return self.weight_accum

    def __call__(self):
        return self.tree.aux(self.weight_accum)

def get_transformation_matrix(src_pose, tgt_pose):
    return torch.matmul(tgt_pose, torch.inverse(src_pose))

def warp_vertices(transformation_matrix, coordinates, skinning_weights, joint_index):
    return _WarpVerticalFunction.apply(transformation_matrix, coordinates, skinning_weights, joint_index)
    # return _C.warp_vertices(transformation_matrix, coordinates, skinning_weights, joint_index)

def blend_transformation_matrix(transformation_matrix, skinning_weights, joint_index):
    coordinates = torch.zeros((skinning_weights.size(0), 3), device=skinning_weights.device)
    _, matrices = _C.warp_vertices(transformation_matrix, coordinates, skinning_weights, joint_index)
    return matrices