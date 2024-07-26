/*
 * Copyright (c) 2020, Huawei Technologies.All rights reserved.
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <torch/csrc/utils/tensor_flatten.h>
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_flatten.h

at::Tensor flatten(std::vector<at::Tensor> tensors)
{
  return torch::utils::flatten_dense_tensors(tensors);
}

std::vector<at::Tensor> unflatten(at::Tensor flat, std::vector<at::Tensor> tensors)
{
  return torch::utils::unflatten_dense_tensors(flat, tensors);
}

PYBIND11_MODULE(apex_C, m) {
  m.def("flatten", &flatten, "Flatten dense tensors");
  m.def("unflatten", &unflatten, "Unflatten dense tensors");
}
