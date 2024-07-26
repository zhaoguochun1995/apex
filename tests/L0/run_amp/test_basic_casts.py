# Copyright (c) 2020, Huawei Technologies.
# Copyright (c) 2019, NVIDIA CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import functools as ft
import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils import common_init, generate_data
import utils

import sys
sys.path.append('../')
import device

npu_input_grad = None

def npu_input_grad_hook(grad):
   global npu_input_grad
   npu_input_grad = grad.to('cpu')

def run_layer_test(test_case, fns, expected, input_shape, test_backward=True):
    for fn, typ in it.product(fns, expected.keys()):
        x = generate_data(0, 10, input_shape, typ).requires_grad_()
        x = x.to(test_case.device)
        x.register_hook(npu_input_grad_hook)
        y = fn(x)
        test_case.assertEqual(y.type(), expected[typ])
        if test_backward:
            y.float().sum().backward(retain_graph=True)
            test_case.assertEqual(npu_input_grad.type().split(".")[-1], utils.MATCH_INPUT[typ].split(".")[-1])

class TestBasicCasts(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        self.device = device.CALCULATE_DEVICE
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def test_linear_is_half(self):
        m = nn.Linear(self.h, self.h).to(self.device)
        f = ft.partial(F.linear, weight=m.weight, bias=m.bias)
        run_layer_test(self, [m, f], utils.ALWAYS_HALF, (self.b, self.h))

    def test_conv2d_is_half(self):
        m = nn.Conv2d(self.c, self.c, self.k).to(self.device)
        f = ft.partial(F.conv2d, weight=m.weight, bias=m.bias)
        run_layer_test(self, [m, f], utils.ALWAYS_HALF, (self.b, self.c, self.h, self.h))

    def test_softmax_is_float(self):
        m = nn.Softmax(dim=1).to(self.device)
        f = ft.partial(F.softmax, dim=1)
        run_layer_test(self, [m, f], utils.ALWAYS_FLOAT, (self.b, self.h))

    @unittest.skipIf(device.is_npu(),"NPU does not support group_norm in half")
    def test_group_norm_is_float(self):
        m = nn.GroupNorm(num_groups=4, num_channels=self.c).to(self.device)
        run_layer_test(self, [m], utils.ALWAYS_FLOAT, (self.b, self.c, self.h, self.h))

    def test_mse_loss_is_float(self):
        shape = (self.b, self.h)
        target = torch.randn(shape).to(self.device)
        mod = nn.MSELoss().to(self.device)
        m = lambda x: mod(x, target)
        f = ft.partial(F.mse_loss, target=target)
        run_layer_test(self, [m], utils.ALWAYS_FLOAT, shape)

    def test_relu_is_match(self):
        run_layer_test(self, [nn.ReLU(), F.relu], utils.MATCH_INPUT, (self.b, self.h))

    def test_batch_norm_is_match(self):
        m = nn.BatchNorm2d(num_features=self.c).to(self.device)
        f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m.running_var,
                       weight=m.weight, bias=m.bias, training=True)
        run_layer_test(self, [m], utils.MATCH_INPUT, (self.b, self.c, self.h, self.h))

        # Test forward-only for BN inference
        m.eval()
        f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m.running_var,
                       weight=m.weight, bias=m.bias, training=False)
        run_layer_test(self, [m, f], utils.MATCH_INPUT, (self.b, self.c, self.h, self.h),
                            test_backward=False)

class TestBannedMethods(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        self.device = device.CALCULATE_DEVICE
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def bce_common(self, assertion):
        shape = (self.b, self.h)
        target = torch.rand(shape).to(self.device)
        mod = nn.BCELoss().to(self.device)
        m = lambda x: mod(x, target)
        f = ft.partial(F.binary_cross_entropy, target=target)
        for fn in [m, f]:
            x = generate_data(0, 10, shape, np.float16).to(self.device)
            assertion(fn, x)

    def test_bce_raises_by_default(self):
        assertion = lambda fn, x: self.assertRaises(NotImplementedError, fn, x)
        self.bce_common(assertion)

    def test_bce_is_float_with_allow_banned(self):
        self.handle._deactivate()
        self.handle = amp.init(enabled=True, allow_banned=True)
        assertion = lambda fn, x: self.assertEqual(fn(x).type(), utils.FLOAT)
        self.bce_common(assertion)

class TestTensorCasts(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        self.device = device.CALCULATE_DEVICE
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def test_matmul_method_is_half(self):
        other = torch.randn(self.h, self.h).to(self.device)
        lhs = lambda x: x.matmul(other)
        rhs = lambda x: other.matmul(x)
        run_layer_test(self, [lhs, rhs], utils.ALWAYS_HALF, (self.h, self.h))

    def test_matmul_op_is_half(self):
        other = torch.randn(self.h, self.h).to(self.device)
        lhs = lambda x: x @ other
        rhs = lambda x: other @ x
        run_layer_test(self, [lhs, rhs], utils.ALWAYS_HALF, (self.h, self.h))

    def test_pow_method_is_float(self):
        fn = lambda x: x.pow(2.)
        run_layer_test(self, [fn], utils.ALWAYS_FLOAT, (self.b, self.h))

    def test_pow_op_is_float(self):
        fn = lambda x: x ** 2.
        run_layer_test(self, [fn], utils.ALWAYS_FLOAT, (self.b, self.h))

    def test_cpu_is_float(self):
        fn = lambda x: x.cpu()
        always_cpu_float = {torch.float: 'torch.FloatTensor',
                            torch.half: 'torch.FloatTensor'}
        run_layer_test(self, [fn], always_cpu_float, (self.b, self.h))

    def test_sum_is_float(self):
        fn = lambda x: x.sum()
        run_layer_test(self, [fn], utils.ALWAYS_FLOAT, (self.b, self.h))

    # TODO: maybe more tests on disabled casting?

if __name__ == '__main__':
    unittest.main()
