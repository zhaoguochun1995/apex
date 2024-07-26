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
import sys
import device
import torch
import torch_npu
import argparse

runner = unittest.TextTestRunner(verbosity=2)
errcode = 0

parser = argparse.ArgumentParser()
parser.add_argument('--npu',
                default=0,
                type=int,
                help='NPU id to use.')
args = parser.parse_args()

device.CALCULATE_DEVICE = "npu:{}".format(args.npu)
torch.npu.set_device(device.CALCULATE_DEVICE)

if device.is_npu():
    sys.path.append('./run_amp')
    sys.path.append('../../apex/contrib/test/')
    from test_basic_casts import TestBannedMethods, TestTensorCasts, TestBasicCasts
    from test_cache import TestCache
    from test_promotion import TestPromotion
    from test_larc import TestLARC
    from test_combine_tensors import TestCombineTensors
    test_dirs = ["run_amp"]
    suite=unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBannedMethods))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestTensorCasts))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBasicCasts))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCache))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPromotion))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestLARC))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCombineTensors))

    result = runner.run(suite)
    if not result.wasSuccessful():
        errcode = 1
    sys.exit(errcode)
else:
    test_dirs = ["run_amp", "run_fp16util", "run_optimizers", "run_fused_layer_norm", "run_pyprof_nvtx", "run_pyprof_data", "run_mlp"]

    for test_dir in test_dirs:
        suite = unittest.TestLoader().discover(test_dir)

        print("\nExecuting tests from " + test_dir)

        result = runner.run(suite)

        if not result.wasSuccessful():
            errcode = 1

    sys.exit(errcode)
