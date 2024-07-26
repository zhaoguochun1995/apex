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

import types
import torch
import torch_npu
from change_data_ptr import change_data_ptr
import torch.distributed as dist
from ._amp_state import maybe_print
from ..fp16_utils import master_params_to_model_params
from ..multi_tensor_apply import multi_tensor_applier
from ..optimizers import FusedSGD
from ..contrib.combine_tensors import (
    combine_npu,
    get_part_combined_tensor,
    is_combined_tensor_valid,
    get_aligned_storage_size
)

TORCH_MAJOR = int(torch.__version__.split('.')[0])

if TORCH_MAJOR == 1:
    from torch._six import inf
else:
    from torch import inf


def get_grad_combined_tensor_from_param(list_of_params):
    if len(list_of_params) > 0 and list_of_params[0].grad is not None:
        list_of_grad = []
        for param in list_of_params:
            if param.requires_grad:
                list_of_grad.append(param.grad)
        original_combined_tensor = combine_npu(list_of_grad)
        return original_combined_tensor, list_of_grad
    else:
        return None, []


def get_grad_combined_tensor_mask_from_param(list_of_params):
    if len(list_of_params) > 0 and list_of_params[0].grad is not None:
        list_of_grad_mask = []
        for param in list_of_params:
            if param.requires_grad:
                grad_size = param.grad.size()
                grad_format = torch_npu.get_npu_format(param)
                list_of_grad_mask.append(torch_npu.npu_format_cast(torch.ones(grad_size).npu(), grad_format))
        grad_combined_tensor_mask = combine_npu(list_of_grad_mask)
        return grad_combined_tensor_mask
    else:
        return None


def clip_grad_norm_fused(combined_grads, combined_grad_masks, max_norm, norm_type):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    tmp_lst = []
    if norm_type == inf:
        for combined_grad, combined_grad_mask in zip(combined_grads, combined_grad_masks):
            if combined_grad is not None:
                tmp_lst.append(combined_grad.float().abs().mul_(combined_grad_mask).max())
        total_norm = max(tmp_lst)
    else:
        for combined_grad, combined_grad_mask in zip(combined_grads, combined_grad_masks):
            if combined_grad is not None:
                tmp_lst.append(combined_grad.float().abs().pow(norm_type).mul_(combined_grad_mask).sum())
        total_norm = torch.stack(tmp_lst).sum().pow(1/norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for combined_grad in combined_grads:
            if combined_grad is not None:
                combined_grad.mul_(clip_coef)
    return total_norm


class AmpOptimizerState(object):
    def __init__(self):
        pass


def _master_params_to_model_params(self):
    stash = self._amp_stash
    if multi_tensor_applier.available:
        if len(stash.all_fp16_params) > 0:
            multi_tensor_applier(
                stash.multi_tensor_scale,
                stash.dummy_overflow_buf,
                [stash.all_fp32_from_fp16_params, stash.all_fp16_params],
                1.0)
    else:
        for fp16_group, fp32_from_fp16_group in zip(stash.fp16_groups, stash.fp32_from_fp16_groups):
            master_params_to_model_params(fp16_group, fp32_from_fp16_group)


def lazy_init_with_master_weights(self):
    stash = self._amp_stash
    stash.fp16_groups = []
    stash.fp32_from_fp16_groups = []
    stash.fp32_from_fp32_groups = []
    for i, param_group in enumerate(self.param_groups):
        # maybe_print("FP16_Optimizer processing param group {}:".format(i))
        fp16_params_this_group = []
        fp32_params_this_group = []
        fp32_from_fp16_params_this_group = []
        for i, param in enumerate(param_group['params']):
            if param.requires_grad:
                if param.type() == 'torch.npu.HalfTensor':
                    # maybe_print("FP16_Optimizer received torch.cuda.HalfTensor with {}"
                    #             .format(param.size()))
                    fp16_params_this_group.append(param)
                    master_param = param.detach().clone().float()
                    master_param.requires_grad = True
                    param_group['params'][i] = master_param
                    fp32_from_fp16_params_this_group.append(master_param)
                    # Reset existing state dict key to the new master param.
                    # We still need to recast per-param state tensors, if any, to FP32.
                    if param in self.state:
                        self.state[master_param] = self.state.pop(param)
                elif param.type() == 'torch.npu.FloatTensor':
                    # maybe_print("FP16_Optimizer received torch.cuda.FloatTensor with {}"
                    #             .format(param.size()))
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param
                else:
                    raise TypeError("Optimizer's parameters must be either "
                                    "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                    "Received {}".format(param.type()))

        stash.fp16_groups.append(fp16_params_this_group)
        stash.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
        stash.fp32_from_fp32_groups.append(fp32_params_this_group)

    stash.all_fp16_params = []
    for group in stash.fp16_groups:
        stash.all_fp16_params += group

    stash.all_fp32_from_fp16_params = []
    for group in stash.fp32_from_fp16_groups:
        stash.all_fp32_from_fp16_params += group

    stash.all_fp32_from_fp32_params = []
    for group in stash.fp32_from_fp32_groups:
        stash.all_fp32_from_fp32_params += group

    # all_fp16_grad_stash is only needed for fused optimizers.
    stash.all_fp16_grad_stash = [None for _ in stash.all_fp16_params]
    # stash.all_fp32_from_fp16_grad_stash = [None for _ in stash.all_fp32_from_fp16_params]
    stash.all_fp32_from_fp32_grad_stash = [None for _ in stash.all_fp32_from_fp32_params]

    for param in stash.all_fp32_from_fp16_params:
        param.grad = None

    for param in stash.all_fp32_from_fp32_params:
        param.grad = None
    
    stash.main_fp16_grad_combine = None
    stash.main_fp32_from_fp16_grad_combine = None
    stash.main_fp32_from_fp32_grad_combine = None
    stash.main_fp16_grad_combine_mask = None
    stash.main_fp32_from_fp16_grad_combine_mask = None
    stash.main_fp32_from_fp32_grad_combine_mask = None

    stash.all_fp32_from_fp32_grad_stash_combine = None

    stash.main_fp16_param_combine = None
    stash.main_fp32_from_fp16_param_combine = None
    stash.main_fp32_from_fp32_param_combine = None
    # Leverage state_dict() and load_state_dict() to recast preexisting per-param state tensors
    self.load_state_dict(self.state_dict())


def post_backward_models_are_masters(scaler, params, stashed_grads, scale_override=None, 
                                     main_grads_combined=None, stashed_grads_combined=None, 
                                     use_npu_fused_optimizer=False, stashed_grads_are_zero=False, main_grads_list=None):
    grads_have_scale, stashed_have_scale, out_scale = scaler.loss_scale(), 1.0, 1.0

    # not much to do if scale == 1.0 and static scaling
    if scaler.loss_scale() == 1.0 and not scaler.dynamic:
        # Clear the stash.
        for i in range(len(stashed_grads)):
            stashed_grads[i] = None
        return

    if scale_override is not None:
        grads_have_scale, stashed_have_scale, out_scale = scale_override

    # This is a lot of python overhead...
    if main_grads_combined is not None:
        scaler.unscale_with_stashed_combined(
            main_grads_combined, 
            stashed_grads_combined if not stashed_grads_are_zero else None,
            scale_override=(grads_have_scale, stashed_have_scale, out_scale),
            grads_list=main_grads_list)
    else:
        grads_needing_unscale = []
        grads_needing_unscale_with_stash = []
        stashed = []
        for param, stashed_grad in zip(params, stashed_grads):
            if param.grad is None and stashed_grad is not None:
                param.grad = stashed_grad
            elif param.grad is not None and (stashed_grad is None or stashed_grads_are_zero):
                grads_needing_unscale.append(param.grad)
            elif param.grad is not None and stashed_grad is not None:
                grads_needing_unscale_with_stash.append(param.grad)
                stashed.append(stashed_grad)
            else:  # param.grad is None and stashed_grad is None
                continue

        # unscale() implements grads*(1/scale), so "scale" should be grads_have_scale/out_scale.
        if len(grads_needing_unscale) > 0:
            scaler.unscale(
                grads_needing_unscale,
                grads_needing_unscale,
                None,  # unused_scale, currently present to avoid API breakage elsewhere
                models_are_masters=True,
                scale_override=grads_have_scale / out_scale)

        if len(grads_needing_unscale_with_stash) > 0:
            scaler.unscale_with_stashed(
                grads_needing_unscale_with_stash,
                stashed,
                grads_needing_unscale_with_stash,
                scale_override=(grads_have_scale, stashed_have_scale, out_scale),
                use_npu_fused_optimizer=use_npu_fused_optimizer)

        if not use_npu_fused_optimizer:
            # Clear the stash.
            for i in range(len(stashed_grads)):
                stashed_grads[i] = None


def prepare_backward_with_master_weights(self):
    stash = self._amp_stash

    self._amp_lazy_init()
    self._check_already_combined_params_and_grads()

    if (self.accelerate or self.is_npu_fused_optimizer) and stash.already_combined:
        if stash.process_zero_grad:
            return

        if stash.main_fp16_grad_combine is not None:
            stash.main_fp16_grad_combine.zero_()

        if stash.main_fp32_from_fp32_grad_combine is not None:
            stash.all_fp32_from_fp32_grad_stash_combine.copy_(stash.main_fp32_from_fp32_grad_combine)
            stash.main_fp32_from_fp32_grad_combine.zero_()
    else:
        for i, param in enumerate(stash.all_fp16_params):
            # Set up to leverage grad copy elision.
            # This may behave differently from an unpatched optimizer if zero_grad is used and the param is unused.
            param.grad = None

        for i, param in enumerate(stash.all_fp32_from_fp32_params):
            stash.all_fp32_from_fp32_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None


def combine_ddp_hook_func(name, param, target_grads_size_list, current_param_size_list,
              name_dict, reduce_stream, partial_combined_grad_list, ready_reduce_index, world_size):
    def hook_function(grad):
        if ready_reduce_index:
            index = ready_reduce_index.pop()
            current_param_size_list[index] = 0
            partial_combined_grad_list[index].div_(world_size)
            reduce_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(reduce_stream):
                dist.all_reduce(partial_combined_grad_list[index])

        current_param_size_list[name_dict[name]] += get_aligned_storage_size(param)
        for i, _ in enumerate(current_param_size_list):
            if current_param_size_list[i] == target_grads_size_list[i] and current_param_size_list[i] != 0:
                ready_reduce_index.append(i)
                break
    return hook_function


def init_combine_ddp_no_master_weights(self):
    stash = self._amp_stash
    combined_grads_list = [stash.main_fp32_grad_combine]
    params_list = [stash.all_fp32_params]

    return self._init_combine_ddp_common(combined_grads_list, params_list)


def init_combine_ddp_with_master_weights(self):
    stash = self._amp_stash
    combined_grads_list = [stash.main_fp16_grad_combine, stash.main_fp32_from_fp32_grad_combine]
    params_list = [stash.all_fp16_params, stash.all_fp32_from_fp32_params]

    return self._init_combine_ddp_common(combined_grads_list, params_list)


def init_combine_ddp_common(self, combined_grads_list, params_list):
    exchange_threshold_max = 24 * 1024 * 1024
    exchange_threshold_min = 1 * 1024 * 1024
    ddp_replica_count = self.ddp_replica_count
    world_size = dist.get_world_size()
    all_reduce_stream = torch.npu.Stream()
    exchange_threshold_list = [0 for _ in combined_grads_list]
    target_grads_size_lists = [[] for _ in combined_grads_list]
    name_dict_list = [{} for _ in combined_grads_list]
    partial_combined_grad_lists = [[] for _ in combined_grads_list]

    for idx, combined_grads in enumerate(combined_grads_list):
        if combined_grads is None:
            continue

        if combined_grads.dim() == 1:
            combined_grads_len = combined_grads.shape[0]
            tmp_combined_grads = torch.tensor(combined_grads_len, dtype=torch.float32, device=combined_grads.device)
            gather_list = [torch.zeros(1, dtype=torch.float32).npu() for _ in range(world_size)]
            dist.all_gather(gather_list, tmp_combined_grads)

            for i in range(1, world_size):
                if gather_list[0] != gather_list[i]:
                    raise RuntimeError("When using combine_ddp, "
                                       "combine_grad does not support inconsistent parameters in each rank. "
                                       "Please consider using the consistent parameters of each rank instead.")

        tmp_combined_grads_len = combined_grads.shape[0] // ddp_replica_count
        exchange_threshold_list[idx] = min(tmp_combined_grads_len, exchange_threshold_max \
            if combined_grads.type() == 'torch.npu.FloatTensor' else exchange_threshold_max * 2)
        exchange_threshold_list[idx] = max(exchange_threshold_list[idx], exchange_threshold_min)
        dist.all_reduce(combined_grads.div_(world_size))

    for idx, params in enumerate(params_list):
        target_grads_size_list = target_grads_size_lists[idx]
        name_dict = name_dict_list[idx]
        tmp_size = 0
        name_order = 0
        for param_idx, param in enumerate(params):
            name = '%d_%d'%(idx, param_idx)
            cur_size = get_aligned_storage_size(param)
            if cur_size > exchange_threshold_list[idx] and tmp_size != 0:
                target_grads_size_list.append(tmp_size)
                tmp_size = 0
                name_order += 1
            tmp_size += cur_size
            name_dict[name] = name_order
            if tmp_size > exchange_threshold_list[idx]:
                target_grads_size_list.append(tmp_size)
                tmp_size = 0
                name_order += 1
        if tmp_size != 0:
            target_grads_size_list.append(tmp_size)
    maybe_print('Optimized combine_ddp replicas: {}'.format(target_grads_size_lists), rank0=True)

    for idx, target_grads_size_list in enumerate(target_grads_size_lists):
        combined_grads = combined_grads_list[idx]
        if combined_grads is None:
            continue

        ptr_index = 0
        partial_combined_grad_list = partial_combined_grad_lists[idx]
        for target_grads_size in target_grads_size_list:
            partial_combined_grad_list.append(get_part_combined_tensor(combined_grads, ptr_index, target_grads_size))
            ptr_index += target_grads_size

    current_param_size_lists = [[0] * len(target_grads_size_list) for target_grads_size_list in
                               target_grads_size_lists]
    ready_reduce_index_list = [[] for _ in combined_grads_list]

    for idx, params in enumerate(params_list):
        for param_idx, param in enumerate(params):
            name = '%d_%d'%(idx, param_idx)
            param.register_hook(
                combine_ddp_hook_func(name, param, target_grads_size_lists[idx], current_param_size_lists[idx],
                          name_dict_list[idx], all_reduce_stream, partial_combined_grad_lists[idx],
                          ready_reduce_index_list[idx], world_size))

    self.ready_reduce_index_list = ready_reduce_index_list
    self.partial_combined_grad_lists = partial_combined_grad_lists
    self.current_param_size_lists = current_param_size_lists
    self.all_reduce_stream = all_reduce_stream
    self.world_size = world_size


def combine_ddp_all_reduce(self):
    last_reduce_grad_list = []
    for idx, partial_combined_grad_list in enumerate(self.partial_combined_grad_lists):
        if partial_combined_grad_list:
            last_reduce_grad = partial_combined_grad_list[self.ready_reduce_index_list[idx][0]]
            last_reduce_grad.div_(self.world_size)
            last_reduce_grad_list.append(last_reduce_grad)

    torch.npu.current_stream().wait_stream(self.all_reduce_stream)
    for idx, last_reduce_grad in enumerate(last_reduce_grad_list):
        dist.all_reduce(last_reduce_grad)
        self.current_param_size_lists[idx][self.ready_reduce_index_list[idx][0]] = 0
        self.ready_reduce_index_list[idx].pop()

def combine_ddp_proc(self):
    if self.combine_ddp:
        if not self.init_combine_ddp:
            self._init_combine_ddp()
            self.init_combine_ddp = True
        else:
            self._combine_ddp_all_reduce()

def post_backward_with_master_weights(self, scaler):
    stash = self._amp_stash

    self._amp_lazy_init()
    self._check_already_combined_params_and_grads()
    self._amp_combined_init()
    self._combine_ddp_proc()

    if self.accelerate:
        scaler.unscale_grad_O2(
            model_grads_combined=stash.main_fp16_grad_combine,
            stashed_master_grads_combined=stash.main_fp32_from_fp16_grad_combine if not stash.process_zero_grad else None,
            master_grads_combined=stash.main_fp32_from_fp16_grad_combine,
            master_grads=stash.fp32_from_fp16_grad_list,
            model_grads=stash.fp16_grad_list)
        if stash.main_fp32_from_fp32_grad_combine is not None:
            scaler.unscale_grad_O2(
                model_grads_combined=stash.main_fp32_from_fp32_grad_combine,
                stashed_master_grads_combined=stash.all_fp32_from_fp32_grad_stash_combine if not stash.process_zero_grad else None,
                master_grads_combined=stash.main_fp32_from_fp32_grad_combine,
                model_grads=stash.fp32_from_fp32_grad_list)
    else:
        # This is a lot of python overhead...
        fp16_grads_needing_unscale = []
        new_fp32_grads = []
        fp16_grads_needing_unscale_with_stash = []
        preexisting_fp32_grads = []
        for fp16_param, fp32_param in zip(stash.all_fp16_params,
                                          stash.all_fp32_from_fp16_params):
            if fp16_param.grad is None and fp32_param.grad is not None:
                continue
            elif fp16_param.grad is not None and fp32_param.grad is None:
                fp32_param.grad = torch.empty_like(fp32_param)
                fp16_grads_needing_unscale.append(fp16_param.grad)
                new_fp32_grads.append(fp32_param.grad)
            elif fp16_param.grad is not None and fp32_param.grad is not None:
                if stash.process_zero_grad:
                    fp16_grads_needing_unscale.append(fp16_param.grad)
                    new_fp32_grads.append(fp32_param.grad)
                else:
                    fp16_grads_needing_unscale_with_stash.append(fp16_param.grad)
                    preexisting_fp32_grads.append(fp32_param.grad)
            else: # fp16_param.grad is None and fp32_param.grad is None:
                continue

        if len(fp16_grads_needing_unscale) > 0:
            scaler.unscale(
                fp16_grads_needing_unscale,
                new_fp32_grads,
                scaler.loss_scale(),
                models_are_masters=False)

        if len(fp16_grads_needing_unscale_with_stash) > 0:
            scaler.unscale_with_stashed(
                fp16_grads_needing_unscale_with_stash,
                preexisting_fp32_grads,
                preexisting_fp32_grads,
                use_npu_fused_optimizer=self.is_npu_fused_optimizer)

        # fp32 params can be treated as they would be in the "no_master_weights" case.
        post_backward_models_are_masters(
            scaler,
            stash.all_fp32_from_fp32_params,
            stash.all_fp32_from_fp32_grad_stash,
            use_npu_fused_optimizer=self.is_npu_fused_optimizer,
            stashed_grads_are_zero=stash.process_zero_grad)
    
    stash.process_zero_grad = False


def lazy_init_no_master_weights(self):
    stash = self._amp_stash
    stash.all_fp16_params = []
    stash.all_fp32_params = []

    check_param_require_grad = self.accelerate or self.is_npu_fused_optimizer

    for i, param_group in enumerate(self.param_groups):
        for i, param in enumerate(param_group['params']):
            if check_param_require_grad and not param.requires_grad:
                continue

            if param.type() == 'torch.npu.HalfTensor':
                stash.all_fp16_params.append(param)
            elif param.type() == 'torch.npu.FloatTensor':
                stash.all_fp32_params.append(param)
            else:
                raise TypeError("Optimizer's parameters must be either "
                                "torch.npu.FloatTensor or torch.npu.HalfTensor."
                                "Received {}".format(param.type()))

    stash.all_fp16_grad_stash = [None for _ in stash.all_fp16_params]
    stash.all_fp32_grad_stash = [None for _ in stash.all_fp32_params]

    stash.all_fp16_grad_stash_combine = None
    stash.all_fp32_grad_stash_combine = None

    stash.fp16_grad_list = []
    stash.main_fp16_grad_combine = None
    stash.main_fp16_grad_combine_mask = None

    stash.fp32_grad_list = []
    stash.main_fp32_grad_combine = None
    stash.main_fp32_grad_combine_mask = None

    stash.main_fp16_param_combine = None
    stash.main_fp32_param_combine = None


def prepare_backward_no_master_weights(self):
    stash = self._amp_stash

    self._amp_lazy_init()
    self._check_already_combined_params_and_grads()

    if (self.accelerate or self.is_npu_fused_optimizer) and stash.already_combined:
        if stash.process_zero_grad:
            return

        if stash.main_fp16_grad_combine is not None:
            stash.all_fp16_grad_stash_combine.copy_(stash.main_fp16_grad_combine)
            stash.main_fp16_grad_combine.zero_()
        if stash.main_fp32_grad_combine is not None:
            stash.all_fp32_grad_stash_combine.copy_(stash.main_fp32_grad_combine)
            stash.main_fp32_grad_combine.zero_()
    else:
        for i, param in enumerate(stash.all_fp16_params):
            stash.all_fp16_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None

        for i, param in enumerate(stash.all_fp32_params):
            stash.all_fp32_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None


def post_backward_no_master_weights(self, scaler):
    stash = self._amp_stash

    self._amp_lazy_init()
    self._check_already_combined_params_and_grads()
    self._amp_combined_init()
    self._combine_ddp_proc()

    if self.accelerate:
        split_types = ((stash.main_fp16_grad_combine, stash.all_fp16_grad_stash_combine, stash.fp16_grad_list),
                (stash.main_fp32_grad_combine, stash.all_fp32_grad_stash_combine, stash.fp32_grad_list))
        for main_grads_combined, stash_grads_combined, main_grads_list in split_types:
            if main_grads_combined is not None:
                post_backward_models_are_masters(scaler, None, None, None, 
                                                 main_grads_combined, stash_grads_combined,
                                                 use_npu_fused_optimizer=self.is_npu_fused_optimizer,
                                                 stashed_grads_are_zero=stash.process_zero_grad,
                                                 main_grads_list=main_grads_list)
    else:
        split_types = ((stash.all_fp16_params, stash.all_fp16_grad_stash),
                 (stash.all_fp32_params, stash.all_fp32_grad_stash))

        for params, stashed_grads in split_types:
            post_backward_models_are_masters(scaler, params, stashed_grads, 
                                             use_npu_fused_optimizer=self.is_npu_fused_optimizer,
                                             stashed_grads_are_zero=stash.process_zero_grad)
    stash.process_zero_grad = False


#####################################################################################
# FusedSGD versions
#####################################################################################

# FusedSGD never explicitly materializes the fp32 gradients for "fp32 from fp16" master params
# outside the kernel, so we must accumulate directly into the model grads.
def prepare_backward_with_master_weights_FusedSGD(self):
    if self.materialize_master_grads:
        prepare_backward_with_master_weights(self)
    else:
        stash = self._amp_stash

        self._amp_lazy_init()

        for i, param in enumerate(stash.all_fp16_params):
            stash.all_fp16_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None

        for i, param in enumerate(stash.all_fp32_from_fp32_params):
            stash.all_fp32_from_fp32_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None


def post_backward_with_master_weights_FusedSGD(self, scaler):
    if self.materialize_master_grads:
        post_backward_with_master_weights(self, scaler)
    else:
        stash = self._amp_stash

        self._amp_lazy_init()

        grads_have_scale = scaler.loss_scale()
        stashed_have_scale = self.most_recent_scale
        out_scale = grads_have_scale
        if self.scale_set_by_backward:
            out_scale = min(grads_have_scale, self.most_recent_scale)

        split_types = ((stash.all_fp16_params, stash.all_fp16_grad_stash),
                 (stash.all_fp32_from_fp32_params, stash.all_fp32_from_fp32_grad_stash))


        # unscale_with_stashed() implements grads*1/scale + stashed_grads*1.
        # stashed_grads are scaled by self.most_recent_scale.
        for params, stashed_grads in split_types:
            post_backward_models_are_masters(scaler, params, stashed_grads,
                                             (grads_have_scale, stashed_have_scale, out_scale))

        self.most_recent_scale = out_scale
        self.scale_set_by_backward = True


def prepare_backward_no_master_weights_FusedSGD(self):
    prepare_backward_no_master_weights(self)


def post_backward_no_master_weights_FusedSGD(self, scaler):
    post_backward_no_master_weights(self, scaler)


def _amp_lazy_init(self):
    stash = self._amp_stash

    if not stash.lazy_init_called:
        self._lazy_init_maybe_master_weights()
        stash.lazy_init_called = True


@torch.no_grad()
def combined_init_with_master_weights(self):
    stash = self._amp_stash
    if stash.already_combined:
        return

    if (not self.accelerate) and (not self.is_npu_fused_optimizer):
        return

    # fp32 from fp32
    all_fp32_from_fp32_params, all_fp32_from_fp32_grad_stash = [], []
    for param in stash.all_fp32_from_fp32_params:
        if param.grad is not None:
            if torch_npu.get_npu_format(param) != torch_npu.get_npu_format(param.grad):
                param.grad = torch_npu.npu_format_cast(param.grad, torch_npu.get_npu_format(param)).contiguous()
            all_fp32_from_fp32_params.append(param)
            all_fp32_from_fp32_grad_stash.append(torch.zeros_like(param.grad))
    stash.all_fp32_from_fp32_params = all_fp32_from_fp32_params
    stash.all_fp32_from_fp32_grad_stash = all_fp32_from_fp32_grad_stash

    if len(stash.all_fp32_from_fp32_grad_stash) > 0:
        stash.all_fp32_from_fp32_grad_stash_combine = combine_npu(stash.all_fp32_from_fp32_grad_stash)

    # fp32 from fp16
    all_fp16_params, all_fp32_from_fp16_params = [], []
    for fp16_param, fp32_from_fp16_param in zip(stash.all_fp16_params, stash.all_fp32_from_fp16_params):
        if fp16_param.grad is not None:
            if torch_npu.get_npu_format(fp16_param.grad) != torch_npu.get_npu_format(fp32_from_fp16_param):
                fp16_param.grad = torch_npu.npu_format_cast(fp16_param.grad,
                                                        torch_npu.get_npu_format(fp32_from_fp16_param)).contiguous()
            fp32_from_fp16_param.grad = torch.zeros_like(fp32_from_fp16_param)
            all_fp16_params.append(fp16_param)
            all_fp32_from_fp16_params.append(fp32_from_fp16_param)
    stash.all_fp16_params = all_fp16_params
    stash.all_fp32_from_fp16_params = all_fp32_from_fp16_params

    stash.main_fp16_grad_combine, stash.fp16_grad_list = get_grad_combined_tensor_from_param(stash.all_fp16_params)

    stash.main_fp32_from_fp16_grad_combine, stash.fp32_from_fp16_grad_list = \
        get_grad_combined_tensor_from_param(stash.all_fp32_from_fp16_params)
    stash.main_fp32_from_fp32_grad_combine, stash.fp32_from_fp32_grad_list = \
        get_grad_combined_tensor_from_param(stash.all_fp32_from_fp32_params)
    # please do not change the order of tensor in this list.
    stash.grads_list = [stash.main_fp16_grad_combine, 
                        stash.main_fp32_from_fp16_grad_combine, 
                        stash.main_fp32_from_fp32_grad_combine]

    if self.is_npu_fused_optimizer:
        # stash.main_fp16_param_combine = combine_npu(stash.all_fp16_params)
        stash.main_fp32_from_fp16_param_combine = combine_npu(stash.all_fp32_from_fp16_params)
        stash.main_fp32_from_fp32_param_combine = combine_npu(stash.all_fp32_from_fp32_params)
    
    stash.already_combined = True


@torch.no_grad()
def combined_init_no_master_weights(self):
    stash = self._amp_stash
    if stash.already_combined:
        return

    if (not self.accelerate) and (not self.is_npu_fused_optimizer):
        return

    all_fp16_params, all_fp16_grad_stash = [], []
    for param in stash.all_fp16_params:
        if param.grad is not None:
            if torch_npu.get_npu_format(param) != torch_npu.get_npu_format(param.grad):
                param.grad = torch_npu.npu_format_cast(param.grad, torch_npu.get_npu_format(param)).contiguous()
            all_fp16_params.append(param)
            all_fp16_grad_stash.append(torch.zeros_like(param.grad))

    stash.all_fp16_params = all_fp16_params
    stash.all_fp16_grad_stash = all_fp16_grad_stash

    all_fp32_params, all_fp32_grad_stash = [], []
    for param in stash.all_fp32_params:
        if param.grad is not None:
            if torch_npu.get_npu_format(param) != torch_npu.get_npu_format(param.grad):
                param.grad = torch_npu.npu_format_cast(param.grad, torch_npu.get_npu_format(param)).contiguous()
            all_fp32_params.append(param)
            all_fp32_grad_stash.append(torch.zeros_like(param.grad))

    stash.all_fp32_params = all_fp32_params
    stash.all_fp32_grad_stash = all_fp32_grad_stash

    if len(stash.all_fp16_grad_stash) > 0:
        # if len == 0, avoid to create a useless combined tensor
        stash.all_fp16_grad_stash_combine = combine_npu(stash.all_fp16_grad_stash, require_copy_value=False)
    if len(stash.all_fp32_grad_stash) > 0:
        stash.all_fp32_grad_stash_combine = combine_npu(stash.all_fp32_grad_stash, require_copy_value=False)

    stash.main_fp16_grad_combine, stash.fp16_grad_list = get_grad_combined_tensor_from_param(stash.all_fp16_params)
    stash.main_fp32_grad_combine, stash.fp32_grad_list = get_grad_combined_tensor_from_param(stash.all_fp32_params)
    # please do not change the order of tensor in this list.
    stash.grads_list = [stash.main_fp16_grad_combine, stash.main_fp32_grad_combine]

    if self.is_npu_fused_optimizer:
        # stash.main_fp16_param_combine = combine_npu(stash.all_fp16_params)
        stash.main_fp32_param_combine = combine_npu(stash.all_fp32_params)

    stash.already_combined = True


def reset_all_combine_flags(self):
    stash = self._amp_stash
    stash.already_combined = False
    stash.params_grads_are_combined_by_group = False
    stash.param_states_are_combined_by_group = False


def check_already_combined_params_and_grads_with_master_weights(self):
    stash = self._amp_stash
    if not self.check_combined_tensors or not stash.already_combined:
        return

    fp16_grad_list = []
    for param in stash.all_fp16_params:
        if param.requires_grad:
            fp16_grad_list.append(param.grad)

    fp32_from_fp16_grad_list = []
    for param in stash.all_fp32_from_fp16_params:
        if param.requires_grad:
            fp32_from_fp16_grad_list.append(param.grad)

    fp32_from_fp32_grad_list = []
    for param in stash.all_fp32_from_fp32_params:
        if param.requires_grad:
            fp32_from_fp32_grad_list.append(param.grad)

    if not is_combined_tensor_valid(stash.main_fp16_grad_combine, fp16_grad_list) or \
        not is_combined_tensor_valid(stash.main_fp32_from_fp16_grad_combine, fp32_from_fp16_grad_list) or \
        not is_combined_tensor_valid(stash.main_fp32_from_fp32_grad_combine, fp32_from_fp32_grad_list):
        maybe_print("Combined grad has been destroyed and will be recombined afterwards, please check if "
                    "there is any operation that may change the data_ptr/size/format of the grads.")
        self._reset_all_combine_flags()
        return

    if self.is_npu_fused_optimizer:
        if not is_combined_tensor_valid(stash.main_fp32_from_fp16_param_combine, stash.all_fp32_from_fp16_params) or \
            not is_combined_tensor_valid(stash.main_fp32_from_fp32_param_combine, stash.all_fp32_from_fp32_params):
            maybe_print("Combined param has been destroyed and will be recombined afterwards, please check if "
                        "there is any operation that may change the data_ptr/size/format of the params.")
            self._reset_all_combine_flags()
            return


def check_already_combined_params_and_grads_no_master_weights(self):
    stash = self._amp_stash
    if not self.check_combined_tensors or not stash.already_combined:
        return

    fp16_grad_list = []
    for param in stash.all_fp16_params:
        if param.requires_grad:
            fp16_grad_list.append(param.grad)

    fp32_grad_list = []
    for param in stash.all_fp32_params:
        if param.requires_grad:
            fp32_grad_list.append(param.grad)

    if not is_combined_tensor_valid(stash.main_fp16_grad_combine, fp16_grad_list) or \
        not is_combined_tensor_valid(stash.main_fp32_grad_combine, fp32_grad_list):
        maybe_print("Combined grad has been destroyed and will be recombined afterwards, please check if "
                    "there is any operation that may change the data_ptr/size/format of the grads.")
        self._reset_all_combine_flags()
        return

    if self.is_npu_fused_optimizer:
        if not is_combined_tensor_valid(stash.main_fp32_param_combine, stash.all_fp32_params):
            maybe_print("Combined param has been destroyed and will be recombined afterwards, please check if "
                        "there is any operation that may change the data_ptr/size/format of the params.")
            self._reset_all_combine_flags()
            return


def is_grad_in_combined_tensor(grad, combined_tensor):
    if combined_tensor is None:
        return False

    combined_tensor_data_start_addr = combined_tensor.data_ptr()
    combined_tensor_data_end_addr = combined_tensor.data_ptr() + \
                                    combined_tensor.numel() * combined_tensor.element_size()
    
    if combined_tensor_data_start_addr <= grad.data_ptr() < combined_tensor_data_end_addr:
        return True
    else:
        return False


def combine_params_and_grads_by_group_with_master_weights(self):
    stash = self._amp_stash
    if stash.params_grads_are_combined_by_group:
        return

    self._amp_combined_init()
    stash.combined_params_indexed_by_group = []
    stash.combined_grads_indexed_by_group = []
    stash.params_lists_indexed_by_group = []

    combined_fp32_from_fp32_param = stash.main_fp32_from_fp32_param_combine
    combined_fp32_from_fp16_param = stash.main_fp32_from_fp16_param_combine
    combined_fp32_from_fp32_grad = stash.main_fp32_from_fp32_grad_combine
    combined_fp32_from_fp16_grad = stash.main_fp32_from_fp16_grad_combine

    combined_group_fp32_from_fp32_param_index, combined_group_fp32_from_fp16_param_index = 0, 0
    combined_group_fp32_from_fp32_grad_index, combined_group_fp32_from_fp16_grad_index = 0, 0

    group_num = 0
    for group in self.param_groups:
        group_num += 1

        group_fp32_from_fp32_params = []
        group_fp32_from_fp16_params = []
        group_fp32_from_fp32_param_size, group_fp32_from_fp16_param_size = 0, 0
        group_fp32_from_fp32_grad_size, group_fp32_from_fp16_grad_size = 0, 0

        for p in group['params']:
            if p.grad is None:
                continue
            param_size = get_aligned_storage_size(p)
            grad_size = get_aligned_storage_size(p.grad)
            if is_grad_in_combined_tensor(p.grad, combined_fp32_from_fp32_grad):
                group_fp32_from_fp32_param_size += param_size
                group_fp32_from_fp32_params.append(p)
                group_fp32_from_fp32_grad_size += grad_size
            else:
                group_fp32_from_fp16_param_size += param_size
                group_fp32_from_fp16_params.append(p)
                group_fp32_from_fp16_grad_size += grad_size

        combined_group_fp32_from_fp32_param = None
        combined_group_fp32_from_fp16_param = None
        combined_group_fp32_from_fp32_grad = None
        combined_group_fp32_from_fp16_grad = None

        combined_group_fp32_from_fp32_param = get_part_combined_tensor(combined_fp32_from_fp32_param,
                                                                       combined_group_fp32_from_fp32_param_index,
                                                                       group_fp32_from_fp32_param_size)
        combined_group_fp32_from_fp16_param = get_part_combined_tensor(combined_fp32_from_fp16_param,
                                                                       combined_group_fp32_from_fp16_param_index,
                                                                       group_fp32_from_fp16_param_size)
        combined_group_fp32_from_fp32_grad = get_part_combined_tensor(combined_fp32_from_fp32_grad, 
                                                                      combined_group_fp32_from_fp32_grad_index,
                                                                      group_fp32_from_fp32_grad_size)
        combined_group_fp32_from_fp16_grad = get_part_combined_tensor(combined_fp32_from_fp16_grad, 
                                                                      combined_group_fp32_from_fp16_grad_index,
                                                                      group_fp32_from_fp16_grad_size)

        combined_group_fp32_from_fp32_param_index += group_fp32_from_fp32_param_size
        combined_group_fp32_from_fp16_param_index += group_fp32_from_fp16_param_size
        combined_group_fp32_from_fp32_grad_index += group_fp32_from_fp32_grad_size
        combined_group_fp32_from_fp16_grad_index += group_fp32_from_fp16_grad_size

        combined_params = []
        combined_grads = []
        params_list = []

        combined_params.append(combined_group_fp32_from_fp32_param)
        combined_params.append(combined_group_fp32_from_fp16_param)
        combined_grads.append(combined_group_fp32_from_fp32_grad)
        combined_grads.append(combined_group_fp32_from_fp16_grad)
        params_list.append(group_fp32_from_fp32_params)
        params_list.append(group_fp32_from_fp16_params)

        stash.combined_params_indexed_by_group.append(combined_params)
        stash.combined_grads_indexed_by_group.append(combined_grads)
        stash.params_lists_indexed_by_group.append(params_list)

    maybe_print("group num: {}".format(group_num))
    stash.params_grads_are_combined_by_group = True


def combine_params_and_grads_by_group_no_master_weights(self):
    stash = self._amp_stash
    if stash.params_grads_are_combined_by_group:
        return

    self._amp_combined_init()
    stash.combined_params_indexed_by_group = []
    stash.combined_grads_indexed_by_group = []
    stash.params_lists_indexed_by_group = []

    combined_fp32_param = stash.main_fp32_param_combine
    combined_fp32_grad = stash.main_fp32_grad_combine

    combined_group_fp32_param_index = 0
    combined_group_fp32_grad_index = 0

    group_num = 0
    for group in self.param_groups:
        group_num += 1

        group_fp32_params = []
        group_fp32_param_size = 0
        group_fp32_grad_size = 0

        for p in group['params']:
            if p.grad is None:
                continue

            param_size = get_aligned_storage_size(p)
            group_fp32_param_size += param_size
            group_fp32_params.append(p)

            grad_size = get_aligned_storage_size(p.grad)
            group_fp32_grad_size += grad_size

        combined_group_fp32_param = None
        combined_group_fp32_grad = None
        combined_group_fp32_param = get_part_combined_tensor(combined_fp32_param, 
                                                             combined_group_fp32_param_index,
                                                             group_fp32_param_size)
        combined_group_fp32_grad = get_part_combined_tensor(combined_fp32_grad, 
                                                            combined_group_fp32_grad_index, 
                                                            group_fp32_grad_size)
        combined_group_fp32_param_index += group_fp32_param_size
        combined_group_fp32_grad_index += group_fp32_grad_size

        combined_params = []
        combined_grads = []
        params_list = []

        combined_params.append(combined_group_fp32_param)
        combined_grads.append(combined_group_fp32_grad)
        params_list.append(group_fp32_params)

        stash.combined_params_indexed_by_group.append(combined_params)
        stash.combined_grads_indexed_by_group.append(combined_grads)
        stash.params_lists_indexed_by_group.append(params_list)

    maybe_print("group num: {}".format(group_num))
    stash.params_grads_are_combined_by_group = True


def new_zero_grad_with_master_weights(self):
    stash = self._amp_stash
    self._amp_lazy_init()
    # Zero the model grads.
    for param in stash.all_fp16_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
    for param in stash.all_fp32_from_fp32_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
    # Clear the master grads that are independent of model grads
    for param in stash.all_fp32_from_fp16_params:
        param.grad = None


def new_zero_grad_accelerate_with_master_weights(self):
    stash = self._amp_stash
    self._amp_lazy_init()
    self._check_already_combined_params_and_grads()
    # Zero the model grads.
    stash.process_zero_grad = True

    if not stash.already_combined:
        for param in stash.all_fp16_params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        for param in stash.all_fp32_from_fp32_params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        for param in stash.all_fp32_from_fp16_params:
            if param.grad is not None:
                param.grad.zero_()
        return

    if stash.main_fp16_grad_combine is not None:
        stash.main_fp16_grad_combine.zero_()
    if stash.main_fp32_from_fp32_grad_combine is not None:
        stash.main_fp32_from_fp32_grad_combine.zero_()
    # Clear the master grads that are independent of model grads
    if stash.main_fp32_from_fp16_grad_combine is not None:
        stash.main_fp32_from_fp16_grad_combine.zero_()


def can_get_combined_tensors(self, name):
    if name == 'params':
        if not self.is_npu_fused_optimizer:
            maybe_print("To get combined params, please use npu fused optimizer.")
            return False
    elif name == 'grads' or name == 'grad_masks':
        if (not self.accelerate) and (not self.is_npu_fused_optimizer):
            maybe_print("To get combined {}, please set combine_grad=True or use npu fused optimizer.".format(name))
            return False
    else:
        maybe_print("{} are not supported to be combined.".format(name))
        return False

    stash = self._amp_stash
    if not stash.already_combined:
        maybe_print("Please get the combined {} after backward phase.".format(name))
        return False
    return True


def get_model_combined_params(self):
    stash = self._amp_stash
    combined_params = []

    if not self._can_get_combined_tensors('params'):
        return combined_params

    self._check_already_combined_params_and_grads()
    self._amp_combined_init()

    if stash.master_weights:
        combined_params.append(stash.main_fp16_param_combine)
        combined_params.append(stash.main_fp32_from_fp32_param_combine)
    else:
        combined_params.append(stash.main_fp32_param_combine)
    return combined_params


def get_model_combined_grads(self):
    stash = self._amp_stash
    combined_grads = []

    if not self._can_get_combined_tensors('grads'):
        return combined_grads

    self._check_already_combined_params_and_grads()
    self._amp_combined_init()

    if stash.master_weights:
        combined_grads.append(stash.main_fp16_grad_combine)
        combined_grads.append(stash.main_fp32_from_fp32_grad_combine)
    else:
        combined_grads.append(stash.main_fp32_grad_combine)
    return combined_grads


def get_model_combined_grad_masks(self):
    stash = self._amp_stash
    combined_grad_masks = []

    if not self._can_get_combined_tensors('grad_masks'):
        return combined_grad_masks

    if stash.master_weights:
        if stash.main_fp16_grad_combine_mask is None:
            stash.main_fp16_grad_combine_mask = \
                get_grad_combined_tensor_mask_from_param(stash.all_fp16_params)
            stash.main_fp32_from_fp32_grad_combine_mask = \
                get_grad_combined_tensor_mask_from_param(stash.all_fp32_from_fp32_params)
        combined_grad_masks.append(stash.main_fp16_grad_combine_mask)
        combined_grad_masks.append(stash.main_fp32_from_fp32_grad_combine_mask)
    else:
        if stash.main_fp32_grad_combine_mask is None:
            stash.main_fp32_grad_combine_mask = \
                get_grad_combined_tensor_mask_from_param(stash.all_fp32_params)
        combined_grad_masks.append(stash.main_fp32_grad_combine_mask)
    return combined_grad_masks


def get_optimizer_combined_params(self):
    stash = self._amp_stash
    combined_params = []

    if not self._can_get_combined_tensors('params'):
        return combined_params

    self._check_already_combined_params_and_grads()
    self._amp_combined_init()

    if stash.master_weights:
        combined_params.append(stash.main_fp32_from_fp16_param_combine)
        combined_params.append(stash.main_fp32_from_fp32_param_combine)
    else:
        combined_params.append(stash.main_fp32_param_combine)
    return combined_params


def get_optimizer_combined_grads(self):
    stash = self._amp_stash
    combined_grads = []

    if not self._can_get_combined_tensors('grads'):
        return combined_grads

    self._check_already_combined_params_and_grads()
    self._amp_combined_init()

    if stash.master_weights:
        combined_grads.append(stash.main_fp32_from_fp16_grad_combine)
        combined_grads.append(stash.main_fp32_from_fp32_grad_combine)
    else:
        combined_grads.append(stash.main_fp32_grad_combine)
    return combined_grads


def get_optimizer_combined_grad_masks(self):
    stash = self._amp_stash
    combined_grad_masks = []

    if not self._can_get_combined_tensors('grad_masks'):
        return combined_grad_masks

    if stash.master_weights:
        if stash.main_fp32_from_fp16_grad_combine_mask is None:
            stash.main_fp32_from_fp16_grad_combine_mask = \
                get_grad_combined_tensor_mask_from_param(stash.all_fp32_from_fp16_params)
            stash.main_fp32_from_fp32_grad_combine_mask = \
                get_grad_combined_tensor_mask_from_param(stash.all_fp32_from_fp32_params)
        combined_grad_masks.append(stash.main_fp32_from_fp16_grad_combine_mask)
        combined_grad_masks.append(stash.main_fp32_from_fp32_grad_combine_mask)
    else:
        if stash.main_fp32_grad_combine_mask is None:
            stash.main_fp32_grad_combine_mask = \
                get_grad_combined_tensor_mask_from_param(stash.all_fp32_params)
        combined_grad_masks.append(stash.main_fp32_grad_combine_mask)
    return combined_grad_masks


def clip_model_grad_norm_fused(self, max_norm, norm_type=2):
    stash = self._amp_stash
    if stash.master_weights:
        raise RuntimeError("clip_model_grad_norm_fused can only be used when opt_level='O1'")

    combined_grads = self.get_model_combined_grads()
    combined_grad_masks = self.get_model_combined_grad_masks()
    total_norm = clip_grad_norm_fused(combined_grads, combined_grad_masks, max_norm, norm_type)
    return total_norm


def clip_optimizer_grad_norm_fused(self, max_norm, norm_type=2):
    combined_grads = self.get_optimizer_combined_grads()
    combined_grad_masks = self.get_optimizer_combined_grad_masks()
    total_norm = clip_grad_norm_fused(combined_grads, combined_grad_masks, max_norm, norm_type)
    return total_norm


def _process_optimizer(optimizer, properties):
    if hasattr(optimizer, "_amp_stash"):
        raise RuntimeError("A given optimizer should only be passed through amp.initialize once.")
    else:
        optimizer._amp_stash = AmpOptimizerState()

    optimizer._amp_stash.lazy_init_called = False
    optimizer._amp_stash.already_patched = False
    optimizer._amp_stash.params_have_scaled_gradients = False
    optimizer.accelerate = properties.combine_grad
    optimizer.combine_ddp = properties.combine_ddp
    optimizer.init_combine_ddp = False
    optimizer.ddp_replica_count = properties.ddp_replica_count
    optimizer.check_combined_tensors = properties.check_combined_tensors
    optimizer._amp_stash.master_weights = properties.master_weights
    optimizer._amp_stash.grads_list = []
    optimizer._amp_stash.already_combined = False

    optimizer._amp_stash.process_zero_grad = True

    optimizer._amp_stash.params_grads_are_combined_by_group = False
    optimizer._amp_stash.combined_params_indexed_by_group = []
    optimizer._amp_stash.combined_grads_indexed_by_group = []
    optimizer._amp_stash.params_lists_indexed_by_group = []
    optimizer._amp_stash.param_states_are_combined_by_group = False
    optimizer._amp_stash.combined_param_states_indexed_by_group = []

    for name in ("_lazy_init_maybe_master_weights",
                 "_master_params_to_model_params",
                 "_prepare_amp_backward",
                 "_post_amp_backward",
                 "_amp_lazy_init",
                 "_amp_combined_init",
                 "_reset_all_combine_flags",
                 "_check_already_combined_params_and_grads",
                 "_combine_params_and_grads_by_group",
                 "_can_get_combined_tensors",
                 "get_model_combined_params",
                 "get_model_combined_grads",
                 "get_optimizer_combined_params",
                 "get_optimizer_combined_grads"):
        if hasattr(optimizer, name):
            raise RuntimeError("Incoming optimizer already has {} defined.".format(name))

    if hasattr(optimizer, "is_npu_fused_optimizer") and optimizer.is_npu_fused_optimizer is True:
        maybe_print("Use npu fused optimizer")
        if properties.opt_level != "O1" and properties.opt_level != "O2":
            raise RuntimeError("Currently, npu fused optimizer can only be used when opt_level='O1' or opt_level='O2'")
    else:
        optimizer.is_npu_fused_optimizer = False

    if properties.combine_grad or optimizer.is_npu_fused_optimizer:
        if properties.opt_level == "O2" and properties.master_weights != True:
            raise RuntimeError("With opt_level O2, master_weights should be True when combine_grad is True or "
                               "npu fused optimizer is used")
    else:
        if properties.check_combined_tensors:
            maybe_print("Because combine_grad != True and no npu fused optimizer is used, "
                        "checking combined tensors function will not take effect!")

    if optimizer.is_npu_fused_optimizer:
        old_load_state_dict = optimizer.load_state_dict
        def new_load_state_dict(self, state_dict):
            old_load_state_dict(state_dict)
            self._amp_stash.param_states_are_combined_by_group = False
        optimizer.load_state_dict = types.MethodType(new_load_state_dict, optimizer)

    # TODO:  Centralize exposure and import error checking for the C backend.
    if multi_tensor_applier.available:
        import amp_C
        optimizer._amp_stash.multi_tensor_scale = amp_C.multi_tensor_scale
        optimizer._amp_stash.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
        optimizer._amp_stash.dummy_overflow_buf = torch.cuda.IntTensor([0]);

    if properties.master_weights:
        optimizer._lazy_init_maybe_master_weights = types.MethodType(
            lazy_init_with_master_weights, optimizer)

        optimizer._master_params_to_model_params = types.MethodType(
            _master_params_to_model_params, optimizer)

        old_step = optimizer.step
        def new_step(self, closure=None):
            stash = self._amp_stash
            if closure is not None:
                raise RuntimeError("Currently, Amp does not support closure use with optimizers.")
            retval = old_step()
            if not isinstance(self, FusedSGD):
                self._master_params_to_model_params()
            # Clear the master grads that wouldn't be zeroed by model.zero_grad()
            if optimizer.accelerate or optimizer.is_npu_fused_optimizer:
                if stash.main_fp32_from_fp16_grad_combine is not None:
                    stash.main_fp32_from_fp16_grad_combine.zero_()
            else:
                for param in stash.all_fp32_from_fp16_params:
                    param.grad = None
            return retval
        optimizer.step = types.MethodType(new_step, optimizer)

        old_zero_grad = optimizer.zero_grad
        if optimizer.accelerate or optimizer.is_npu_fused_optimizer:
            optimizer.zero_grad = types.MethodType(new_zero_grad_accelerate_with_master_weights, optimizer)
        else:
            optimizer.zero_grad = types.MethodType(new_zero_grad_with_master_weights, optimizer)

        if optimizer.is_npu_fused_optimizer:
            optimizer._combine_params_and_grads_by_group = types.MethodType(
                combine_params_and_grads_by_group_with_master_weights, optimizer)

        if isinstance(optimizer, FusedSGD):
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_with_master_weights_FusedSGD, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_with_master_weights_FusedSGD, optimizer)
        else:
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_with_master_weights, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_with_master_weights, optimizer)
            optimizer._init_combine_ddp = types.MethodType(
                init_combine_ddp_with_master_weights, optimizer)
        
        optimizer._amp_combined_init = types.MethodType(combined_init_with_master_weights, optimizer)
        optimizer._check_already_combined_params_and_grads = types.MethodType(
            check_already_combined_params_and_grads_with_master_weights, optimizer)
    else:
        optimizer._lazy_init_maybe_master_weights = types.MethodType(
            lazy_init_no_master_weights, optimizer)

        old_zero_grad = optimizer.zero_grad
        if optimizer.accelerate or optimizer.is_npu_fused_optimizer:
            def new_zero_grad_accelerate_no_master_weights(self):
                stash = self._amp_stash
                self._amp_lazy_init()
                self._check_already_combined_params_and_grads()
                # Zero the model grads.
                stash.process_zero_grad = True

                if not stash.already_combined:
                    old_zero_grad()
                    return

                if stash.main_fp16_grad_combine is not None:
                    stash.main_fp16_grad_combine.zero_()
                if stash.main_fp32_grad_combine is not None:
                    stash.main_fp32_grad_combine.zero_()
            optimizer.zero_grad = types.MethodType(new_zero_grad_accelerate_no_master_weights, optimizer)

        if optimizer.is_npu_fused_optimizer:
            optimizer._combine_params_and_grads_by_group = types.MethodType(
                combine_params_and_grads_by_group_no_master_weights, optimizer)

        if isinstance(optimizer, FusedSGD):
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_no_master_weights_FusedSGD, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_no_master_weights_FusedSGD, optimizer)
        else:
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_no_master_weights, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_no_master_weights, optimizer)
            optimizer._init_combine_ddp = types.MethodType(
                init_combine_ddp_no_master_weights, optimizer)

        optimizer._amp_combined_init = types.MethodType(combined_init_no_master_weights, optimizer)
        optimizer._check_already_combined_params_and_grads = types.MethodType(
            check_already_combined_params_and_grads_no_master_weights, optimizer)

    optimizer._amp_lazy_init = types.MethodType(_amp_lazy_init, optimizer)
    optimizer._reset_all_combine_flags = types.MethodType(reset_all_combine_flags, optimizer)
    optimizer._can_get_combined_tensors = types.MethodType(can_get_combined_tensors, optimizer)
    optimizer.get_model_combined_params = types.MethodType(get_model_combined_params, optimizer)
    optimizer.get_model_combined_grads = types.MethodType(get_model_combined_grads, optimizer)
    optimizer.get_model_combined_grad_masks = types.MethodType(get_model_combined_grad_masks, optimizer)
    optimizer.get_optimizer_combined_params = types.MethodType(get_optimizer_combined_params, optimizer)
    optimizer.get_optimizer_combined_grads = types.MethodType(get_optimizer_combined_grads, optimizer)
    optimizer.get_optimizer_combined_grad_masks = types.MethodType(get_optimizer_combined_grad_masks, optimizer)
    optimizer.clip_model_grad_norm_fused = types.MethodType(clip_model_grad_norm_fused, optimizer)
    optimizer.clip_optimizer_grad_norm_fused = types.MethodType(clip_optimizer_grad_norm_fused, optimizer)
    optimizer._combine_ddp_proc = types.MethodType(combine_ddp_proc, optimizer)
    optimizer._init_combine_ddp_common = types.MethodType(init_combine_ddp_common, optimizer)
    optimizer._combine_ddp_all_reduce = types.MethodType(combine_ddp_all_reduce, optimizer)

    old_add_param_group = optimizer.add_param_group

    def new_add_param_group(self, new_group):
        stash = self._amp_stash

        if not stash.lazy_init_called:
            self._lazy_init_maybe_master_weights()
            stash.lazy_init_called = True

        assert isinstance(new_group, dict), "param group must be a dict"

        new_params = new_group['params']
        if isinstance(new_params, torch.Tensor):
            new_group['params'] = [new_params]
        elif isinstance(new_params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            new_group['params'] = list(new_params)

        if properties.master_weights:
            # Mutate new_group in-place to use FP32 master params
            fp16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_fp16_params_this_group = []
            for i, param in enumerate(new_group['params']):
                if param.requires_grad:
                    if param.type() == 'torch.npu.HalfTensor':
                        fp16_params_this_group.append(param)
                        master_param = param.detach().clone().float()
                        master_param.requires_grad = True
                        new_group['params'][i] = master_param
                        fp32_from_fp16_params_this_group.append(master_param)
                    elif param.type() == 'torch.npu.FloatTensor':
                        fp32_params_this_group.append(param)
                        new_group['params'][i] = param
                    else:
                        raise TypeError("Optimizer's parameters must be either "
                                        "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                        "Received {}".format(param.type()))

            stash.fp16_groups.append(fp16_params_this_group)
            stash.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
            stash.fp32_from_fp32_groups.append(fp32_params_this_group)

            stash.all_fp16_params += fp16_params_this_group
            stash.all_fp32_from_fp16_params += fp32_from_fp16_params_this_group
            stash.all_fp32_from_fp32_params += fp32_params_this_group

            stash.all_fp32_from_fp32_grad_stash += [None for _ in fp32_params_this_group]
        else:
            for param in new_group['params']:
                if param.type() == 'torch.npu.HalfTensor':
                    stash.all_fp16_params.append(param)
                    stash.all_fp16_grad_stash.append(None)
                elif param.type() == 'torch.npu.FloatTensor':
                    stash.all_fp32_params.append(param)
                    stash.all_fp32_grad_stash.append(None)
                else:
                    raise TypeError("Optimizer's parameters must be either "
                                    "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                    "Received {}".format(param.type()))

        old_add_param_group(new_group)

    optimizer.add_param_group = types.MethodType(new_add_param_group, optimizer)

    return optimizer
