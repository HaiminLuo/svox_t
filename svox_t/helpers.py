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
import torch
import numpy as np
import time

def unique(indices):
    hash_table = torch.zeros(indices.max() + 1, device=indices.device)
    hash_table[indices] = 1
    indices = torch.nonzero(hash_table).squeeze()

    return indices

class N3TreeView:
    #@profile
    def __init__(self, tree, key):
        self.tree = tree
        local = False
        self.single_key = False
        if isinstance(key, LocalIndex):
            key = key.val
            local = True
        if isinstance(key, tuple) and len(key) >= 3:
            # Handle tree[x, y, z[, c]]
            main_key = torch.tensor(key[:3], dtype=torch.float32,
                        device=tree.data.device).reshape(1, 3)
            if len(key) > 3:
                key = (main_key, *key[3:])
            else:
                key = main_key
        leaf_key = key[0] if isinstance(key, tuple) else key
        if torch.is_tensor(leaf_key) and leaf_key.ndim == 2 and leaf_key.shape[1] == 3:
            # Handle tree[P[, c]] where P is a (B, 3) matrix of 3D points
            if leaf_key.dtype != torch.float32:
                leaf_key = leaf_key.float()
            val, target, unique_leaf_node = tree.forward(self.tree.features, leaf_key, want_node_ids=True, world=not local, want_leaf_node=True)
            self._packed_ids = target.clone()
            self.unique_leaf_node = unique_leaf_node
            self.leaf_node_id = target
            # leaf_node = (*tree._unpack_index(target).T,)
            leaf_node = (*self.unique_leaf_node.T,)
        else:
            self._packed_ids = None
            if isinstance(leaf_key, int):
                leaf_key = torch.tensor([leaf_key], device=tree.data.device)
                self.single_key = True
            leaf_node = self.tree._all_leaves()
            leaf_node = leaf_node.__getitem__(leaf_key).T
        if isinstance(key, tuple):
            self.key = (*leaf_node, *key[1:])
        else:
            self.key = (*leaf_node,)
        self._value = None
        self._tree_ver = tree._ver

    def __repr__(self):
        self._check_ver()
        return "N3TreeView(" + repr(self.values) + ")"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self._check_ver()
        if kwargs is None:
            kwargs = {}
        new_args = []
        for arg in args:
            if isinstance(arg, N3TreeView):
                new_args.append(arg.values)
            else:
                new_args.append(arg)
        return func(*new_args, **kwargs)

    def set(self, value):
        self._check_ver()
        if isinstance(value, N3TreeView):
            value = value.values_nograd
        self.tree.data.data[self.key] = value

    #@profile
    def refine(self, repeats=1):
        """
        Refine selected leaves using tree.refine
        """
        self._check_ver()
        # sel = self._unique_node_key()
        sel = (*self.unique_leaf_node.T,)
        ret = self.tree.refine(repeats, sel=sel, leaf_node=self.unique_leaf_node)
        return ret

    @property
    def values(self):
        """
        Values of the selected leaves (autograd enabled)

        :return: (n_leaves, data_dim) float32 note this is 2D even if key is int
        """
        self._check_ver()
        ret = self.tree.data[self.key]
        return ret[0] if self.single_key else ret

    @property
    def values_nograd(self):
        """
        Values of the selected leaves (no autograd)

        :return: (n_leaves, data_dim) float32 note this is 2D even if key is int
        """
        self._check_ver()
        ret = self.tree.data.data[self.key]
        return ret[0] if self.single_key else ret

    @property
    def shape(self):
        self._check_ver()
        return self.values_nograd.shape

    @property
    def ndim(self):
        return 2

    @property
    def depths(self):
        """
        Get a list of selected leaf depths in tree,
        in same order as values, corners.
        Root is at depth -1. Any children of
        root will have depth 0, etc.

        :return: (n_leaves) int32
        """
        self._check_ver()
        return self.tree.parent_depth[self.key[0], 1]

    @property
    def lengths(self):
        """
        Get a list of selected leaf side lengths in tree (world dimensions),
        in same order as values, corners, depths

        :return: (n_leaves, 3) float
        """
        self._check_ver()
        return (2.0 ** (-self.depths.float() - 1.0))[:, None] / self.tree.invradius

    @property
    def lengths_local(self):
        """
        Get a list of selected leaf side lengths in tree
        (local index :math:`[0, 1]^3`),
        in same order as values, corners, depths

        :return: (n_leaves) float
        """
        self._check_ver()
        return 2.0 ** (-self.depths.float() - 1.0)

    @property
    def corners(self):
        """
        Get a list of selected leaf lower corners in tree
        (world coordinates),
        in same order as values, lengths, depths

        :return: (n_leaves, 3) float
        """
        self._check_ver()
        return (self.tree._calc_corners(self._indexer())
                - self.tree.offset) / self.tree.invradius

    @property
    def corners_local(self):
        """
        Get a list of selected leaf lower corners in tree
        (local index :math:`[0, 1]^3`),
        in same order as values, lengths, depths

        :return: (n_leaves, 3) float
        """
        self._check_ver()
        return self.tree._calc_corners(self._indexer())

    def sample(self, n_samples, device=None):
        """
        Sample n_samples uniform points in each selected leaf (world coordinates)

        :param n_samples: samples for each leaf
        :param device: device to output random samples in

        :return: (n_leaves, n_samples, 3) float
        """
        self._check_ver()
        if device is None:
            device = self.tree.data.device
        corn = self.corners.to(device=device)
        length = self.lengths.to(device=device)
        if length.ndim == 1:
            length = length[:, None]
        u = torch.rand((corn.shape[0], n_samples, 3),
                device=device,
                dtype=length.dtype) * length[:, None]
        return corn[:, None] + u

    def sample_local(self, n_samples):
        """
        Sample n_samples uniform points in each selected leaf
        (local index :math:`[0, 1]^3`)

        :return: (n_leaves, n_samples, 3) float
        """
        self._check_ver()
        corn = self.corners_local
        length = self.lengths_local
        u = torch.rand((corn.shape[0], n_samples, 3),
                device=length.device,
                dtype=length.dtype) * length[:, None, None]
        return corn[:, None] + u

    def aux(self, arr):
        """
        Index an auxiliary tree data array of size (capacity, N, N, N, Any)
        using this view
        """
        return arr[self.key]

    # In-place modification helpers
    def normal_(self, mean=0.0, std=1.0):
        """
        Set all values to random normal

        :param mean: normal mean
        :param std: normal std

        """
        self._check_ver()
        self.tree.data.data[self.key] = torch.randn_like(
                self.tree.data.data[self.key]) * std + mean

    def uniform_(self, min=0.0, max=1.0):
        """
        Set all values to random uniform

        :param min: interval min
        :param max: interval max

        """
        self._check_ver()
        self.tree.data.data[self.key] = torch.rand_like(
                self.tree.data.data[self.key]) * (max - min) + min

    def clamp_(self, min=None, max=None):
        """
        Clamp.

        :param min: clamp min value, None=disable
        :param max: clamp max value, None=disable

        """
        self._check_ver()
        self.tree.data.data[self.key] = self.tree.data.data[self.key].clamp(min, max)

    def relu_(self):
        """
        Apply relu to all elements.
        """
        self._check_ver()
        self.tree.data.data[self.key] = torch.relu(self.tree.data.data[self.key])

    def sigmoid_(self):
        """
        Apply sigmoid to all elements.
        """
        self._check_ver()
        self.tree.data.data[self.key] = torch.sigmoid(self.tree.data.data[self.key])

    def nan_to_num_(self, inf_val=2e4):
        """
        Convert nans to 0.0 and infs to inf_val
        """
        data = self.tree.data.data[self.key]
        data[torch.isnan(data)] = 0.0
        inf_mask = torch.isinf(data)
        data[inf_mask & (data > 0)] = inf_val
        data[inf_mask & (data < 0)] = -inf_val
        self.tree.data.data[self.key] = data

    def __setitem__(self, key, value):
        """
        Warning: inefficient impl
        """
        val = self.values_nograd
        val.__setitem__(key, value)
        self.set(val)

    def _indexer(self):
        return torch.stack(self.key[:4], dim=-1)
    
    
    #@profile
    def _unique_node_key(self):
        if self._packed_ids is None:
            return self.key[:4]

        uniq_ids = torch.unique(self._packed_ids)
        unpacked_ids = self.tree._unpack_index(uniq_ids)
        # # s = time.time()
        # hash_table = torch.zeros(self._packed_ids.max() + 1, device=self._packed_ids.device)
        # hash_table[self._packed_ids] = 1
        # uniq_ids = torch.nonzero(hash_table).squeeze()
        # # print(time.time() - s)
        return (*unpacked_ids.T,)

    def _check_ver(self):
        if self.tree._ver > self._tree_ver:
            self.key = self._packed_ids = None
            raise RuntimeError("N3TreeView has been invalidated because tree " +
                    "data layout has changed")

# Redirect functions to Tensor
def _redirect_funcs():
    redir_funcs = ['__floordiv__', '__mod__', '__div__',
                   '__eq__', '__ne__', '__ge__', '__gt__', '__le__',
                   '__lt__', '__floor__', '__ceil__', '__round__', '__len__',
                   'item', 'size', 'dim', 'numel']
    redir_grad_funcs = ['__add__', '__mul__', '__sub__',
                   '__mod__', '__div__', '__truediv__',
                   '__radd__', '__rsub__', '__rmul__',
                   '__rdiv__', '__abs__', '__pos__', '__neg__',
                   '__len__', 'clamp', 'clamp_max', 'clamp_min', 'relu', 'sigmoid',
                   'max', 'min', 'mean', 'sum', '__getitem__']
    def redirect_func(redir_func, grad=False):
        def redir_impl(self, *args, **kwargs):
            return getattr(self.values if grad else self.values_nograd, redir_func)(
                    *args, **kwargs)
        setattr(N3TreeView, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
    for redir_func in redir_grad_funcs:
        redirect_func(redir_func, grad=True)
_redirect_funcs()


def _get_c_extension():
    from warnings import warn
    try:
        import svox_t.csrc as _C
        if not hasattr(_C, "query_vertical"):
            _C = None
    except:
        _C = None

    if _C is None:
        warn("CUDA extension svox.csrc could not be loaded! " +
             "Operations will be slow.\n" +
             "Please do not import svox in the SVOX source directory.")
    return _C

class LocalIndex:
    """
    To query N3Tree using 'local' index :math:`[0,1]^3`,
    tree[LocalIndex(points)] where points (N, 3)
    """
    def __init__(self, val):
        self.val = val

class DataFormat:
    RGBA = 0
    SH = 1
    SG = 2
    ASG = 3
    def __init__(self, txt):
        nonalph_idx = [c.isalpha() for c in txt]
        if False in nonalph_idx:
            nonalph_idx = nonalph_idx.index(False)
            self.basis_dim = int(txt[nonalph_idx:])
            format_type = txt[:nonalph_idx]
            if format_type == "SH":
                self.format = DataFormat.SH
            elif format_type == "SG":
                self.format = DataFormat.SG
            elif format_type == "ASG":
                self.format = DataFormat.ASG
            else:
                self.format = DataFormat.RGBA
        else:
            self.format = DataFormat.RGBA
            self.basis_dim = -1

    def __repr__(self):
        if self.format == DataFormat.SH:
            r = "SH"
        elif self.format == DataFormat.SG:
            r = "SG"
        elif self.format == DataFormat.ASG:
            r = "ASG"
        else:
            r = "RGBA"
        if self.basis_dim >= 0:
            r += str(self.basis_dim)
        return r
