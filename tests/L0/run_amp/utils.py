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

import torch
import numpy as np

import sys
sys.path.append('../')
import device

HALF = 'torch.npu.HalfTensor'
FLOAT = 'torch.npu.FloatTensor'

DTYPES = [torch.half, torch.float]

ALWAYS_HALF = {torch.float: HALF,
               torch.half: HALF}
ALWAYS_FLOAT = {torch.float: FLOAT,
                torch.half: FLOAT}
MATCH_INPUT = {torch.float: FLOAT,
               torch.half: HALF}

def common_init(test_case):
    test_case.h = 64
    test_case.b = 16
    test_case.c = 16
    test_case.k = 3
    test_case.t = 10
    global HALF, FLOAT, DTYPES, ALWAYS_HALF, ALWAYS_FLOAT, MATCH_INPUT
    if device.is_npu():
        HALF = 'torch.npu.HalfTensor'
        FLOAT = 'torch.npu.FloatTensor'
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        HALF = 'torch.cuda.HalfTensor'
        FLOAT = 'torch.cuda.FloatTensor'
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    ALWAYS_HALF = {torch.float: HALF,
                   torch.half: HALF}
    ALWAYS_FLOAT = {torch.float: FLOAT,
                    torch.half: FLOAT}
    MATCH_INPUT = {torch.float: FLOAT,
                   torch.half: HALF}

def generate_data(min, max, shape, dtype):
    if dtype == torch.float32:
        dtype = np.float32
    if dtype == torch.float16:
        dtype = np.float16
    input1 = np.random.uniform(min, max, shape).astype(dtype)
    npu_input1 = torch.from_numpy(input1)
    return npu_input1