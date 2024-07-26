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
from apex.amp import _amp_state
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')
import device
import utils

from utils import common_init, HALF, FLOAT,\
    ALWAYS_HALF, ALWAYS_FLOAT, MATCH_INPUT,\
    generate_data
    

def get_reference_grad(i, w, ops):
    # Creating new tensors ensures, among other things, that the new tensors are not in the cache.
    # In fact, they are guaranteed not to use the cache because they are not torch.nn.Parameters.
    fp32_i = i.detach().clone().float()
    fp32_w = w.detach().clone().float().requires_grad_()
    loss = ops(fp32_i, fp32_w)
    loss.backward()
    return fp32_w.grad

class WhitelistModule(torch.nn.Module):
    def __init__(self, dtype):
        super(WhitelistModule, self).__init__()
        weight_parameter = torch.from_numpy(np.arange(8*8, dtype=dtype)).view(8,8).to(device.CALCULATE_DEVICE)
        self.weight = torch.nn.Parameter(weight_parameter)

    @staticmethod
    def ops(input, weight):
        return (input.mm(weight)).mm(weight).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class BlacklistModule(torch.nn.Module):
    def __init__(self, dtype):
        super(BlacklistModule, self).__init__()
        weight_parameter = torch.from_numpy(np.arange(2*8, dtype=dtype)).view(2,8).to(device.CALCULATE_DEVICE)
        self.weight = torch.nn.Parameter(weight_parameter)

    @staticmethod
    def ops(input, weight):
        return (input + torch.pow(weight, 2) + torch.pow(weight, 2)).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class PromoteModule(torch.nn.Module):
    def __init__(self, dtype):
        super(PromoteModule, self).__init__()
        weight_parameter = torch.from_numpy(np.arange(2*8, dtype=dtype)).view(2,8).to(device.CALCULATE_DEVICE)
        self.weight = torch.nn.Parameter(weight_parameter)

    @staticmethod
    def ops(input, weight):
        return ((input*weight)*weight).sum()

    def forward(self, input):
        return self.ops(input, self.weight)

class TestCache(unittest.TestCase):
    def setUp(self):
        self.x = torch.ones((2, 8), dtype=torch.float32).to(device.CALCULATE_DEVICE)
        common_init(self)

    def tearDown(self):
        pass

    def train_eval_train_test(self, module, t):
        model = module(t).to(device.CALCULATE_DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        _amp_state.allow_incoming_model_not_fp32 = True
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        _amp_state.allow_incoming_model_not_fp32 = False
        
        def training_step():
            for param in model.parameters():
                param.grad = None
        
            loss = model(self.x).sum()
            _amp_state.loss_scalers[0]._loss_scale = 4.0
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        
            self.assertEqual(len([p.grad for p in model.parameters() if p.grad is not None]), 1)
            self.assertEqual(model.weight.grad.type(), model.weight.type())
        
            reference_grad = get_reference_grad(self.x, model.weight, model.ops)
        
            # Currently there's no difference in the allclose calls, so no need for branching,
            # but I'm keeping this in case we want different tolerances for fp16 and fp32 checks. 
            if model.weight.grad.type() == utils.HALF:
                self.assertTrue(torch.allclose(model.weight.grad.float().to('cpu'), reference_grad.to('cpu')))
            elif model.weight.grad.type() == utils.FLOAT:
                self.assertTrue(torch.allclose(model.weight.grad.float().to('cpu'), reference_grad.to('cpu')))
            else:
                raise RuntimeError("model.weight.grad.type = {}".format(model.weight.grad.type()))

            model.weight.data -= 1.
        
        # Simulates first epoch
        training_step()
        
        # Simulates eval
        with torch.no_grad():
            loss = model(self.x).sum()
        
        # Simulates resuming training after eval
        training_step()

        _amp_state.handle._deactivate()
   
    # I could easily have these as a set of for loops in a single test,
    # instead of going for granularity.
    def test_whitelist_module_fp16_weight(self):
        self.train_eval_train_test(WhitelistModule, np.float16)


    def test_whitelist_module_fp32_weight(self):
        self.train_eval_train_test(WhitelistModule, np.float32)


    def test_blacklist_module_fp16_weight(self):
        self.train_eval_train_test(BlacklistModule, np.float16)


    def test_blacklist_module_fp32_weight(self):
        self.train_eval_train_test(BlacklistModule, np.float32)

    def test_promote_module_fp16_weight(self):
        self.train_eval_train_test(PromoteModule, np.float16)

    def test_promote_module_fp32_weight(self):
        self.train_eval_train_test(PromoteModule, np.float32)


if __name__ == '__main__':
    unittest.main()
