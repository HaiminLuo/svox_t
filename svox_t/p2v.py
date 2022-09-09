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
from torch import nn, autograd
from svox_t.helpers import _get_c_extension
from warnings import warn

_C = _get_c_extension()

class _VoxelizationFunction(autograd.Function):
    @staticmethod
    def forward(ctx, points, point_features, volume_corner, volume_size, n_voxels, kernel_radius, conv_radius):
        voxels = _C.p2v(points, point_features, volume_corner, volume_size, n_voxels, kernel_radius, conv_radius)
        
        ctx.save_for_backward(points, point_features, volume_corner, volume_size)
        ctx.n_voxels = n_voxels
        ctx.kernel_radius = kernel_radius
        ctx.conv_radius = conv_radius
        
        return voxels

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            points_grad, point_features_grad = _C.p2v_backward(grad_output, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.saved_tensors[2], ctx.saved_tensors[3], ctx.n_voxels, ctx.kernel_radius, ctx.conv_radius)

            return points_grad, point_features_grad, None, None, None, None, None
        return None, None, None, None, None, None, None

def voxelize(points, point_features, volume_corner, volume_size, n_voxels, kernel_radius, conv_radius):
    return _VoxelizationFunction.apply(points, point_features, volume_corner, volume_size, n_voxels, kernel_radius, conv_radius)