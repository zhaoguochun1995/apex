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

import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F

from utils import common_init, HALF, FLOAT, DTYPES,\
    generate_data
import utils
import sys
sys.path.append('../')
import device

class TestPromotion(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        self.device = device.CALCULATE_DEVICE
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def run_binary_promote_test(self, fns, input_shape, x_inplace=False):
        type_pairs = it.product(DTYPES, DTYPES)
        for fn, (xtype, ytype) in it.product(fns, type_pairs):
            x = generate_data(0, 10, input_shape, xtype).requires_grad_()
            x_leaf = x
            if x_inplace:
                # We need a non-leaf to call in place on
                x = x.clone()
            y = generate_data(0, 10, input_shape, dtype=ytype).to(self.device)
            x = x.to(self.device)
            out = fn(x, y)
            if x_inplace:
                # In place: always match xtype
                self.assertEqual(out.type(), x.type())
            else:
                # Out of place: match widest type
                if xtype == torch.float or ytype == torch.float:
                    self.assertEqual(out.type(), utils.FLOAT)
                else:
                    self.assertEqual(out.type(), utils.HALF)
            out.float().sum().backward()
            self.assertEqual(x_leaf.grad.dtype, xtype)

    def test_atan2_matches_widest(self):
        fns = [lambda x, y : torch.atan2(x, y),
               lambda x, y : x.atan2(y)]
        self.run_binary_promote_test(fns, (self.b,))

    def test_mul_matches_widest(self):
        fns = [lambda x, y : torch.mul(x, y),
               lambda x, y: x.mul(y)]
        self.run_binary_promote_test(fns, (self.b,))

    def test_cat_matches_widest(self):
        shape = self.b
        ys = [generate_data(0, 10, shape, dtype=torch.half).to(self.device) for _ in range(5)]
        x_float = generate_data(0, 10, shape, dtype=torch.float).to(self.device)
        out = torch.cat(ys + [x_float])
        self.assertEqual(out.type(), utils.FLOAT)
        x_half = generate_data(0, 10, shape, dtype=torch.half).to(self.device)
        out = torch.cat(ys + [x_half])
        self.assertEqual(out.type(), utils.HALF)

    def test_inplace_exp_is_error_for_half(self):
        xs = generate_data(0, 10, self.b, dtype=torch.float).to(self.device)
        xs.exp_()
        self.assertEqual(xs.type(), utils.FLOAT)
        xs = generate_data(0, 10, self.b, dtype=torch.half).to(self.device)
        with self.assertRaises(NotImplementedError):
            xs.exp_()

    def test_inplace_add_matches_self(self):
        fn = lambda x, y: x.add_(y)
        self.run_binary_promote_test([fn], (self.b,), x_inplace=True)

if __name__ == '__main__':
    unittest.main()
