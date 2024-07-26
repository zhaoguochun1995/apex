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

import sys
from types import MethodType
import functools
import torch
import torch.distributed as dist
import numpy as np
import warnings
from ._amp_state import _amp_state, warn_or_err, container_abcs, maybe_print
from .handle import disable_casts
from .scaler import LossScaler
from ._process_optimizer import _process_optimizer
from apex.fp16_utils import convert_network
from ..fp16_utils import FP16_Optimizer as FP16_Optimizer_general
from ..contrib.optimizers import FP16_Optimizer as FP16_Optimizer_for_fused

if torch.distributed.is_available():
    from ..parallel import DistributedDataParallel as apex_DDP
    from ..parallel.LARC import LARC


def zero_grad(self, set_to_none: bool = False) -> None:
    r"""Patch for torch.nn.Module.zero_grad. For combined grad or NPU fused optimizers,
    set_to_none must be False.

    Args:
        set_to_none (bool): instead of setting to zero, set the grads to None.
            See :meth:`torch.optim.Optimizer.zero_grad` for details.
    """

    assert set_to_none is False, "For combined grad, `set_to_none` must be False."

    if getattr(self, '_is_replica', False):
        warnings.warn(
            "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
            "The parameters are copied (in a differentiable manner) from the original module. "
            "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
            "If you need gradients in your forward method, consider using autograd.grad instead.")

    for p in self.parameters():
        if p.grad is not None:
            if p.grad.grad_fn is not None:
                p.grad.detach_()
            else:
                p.grad.requires_grad_(False)
            p.grad.zero_()


def to_type(dtype, t):
    if isinstance(t, torch.Tensor):
        if not 'npu' in t.type():
            # This should not be a hard error, since it may be legitimate.
            warnings.warn("An input tensor was not npu.")
        # GANs require this.
        # if t.requires_grad:
        #     warn_or_err("input data requires grad.  Since input data is not a model parameter,\n"
        #         "its gradients will not be properly allreduced by DDP.")
        if t.is_floating_point():
            return t.to(dtype)
        return t
    else:
        # Trust the user's custom batch type, that's all I can do here.
        return t.to(dtype)


# Modified from torch.optim.optimizer.py.  This is a bit more general than casted_args in utils.py.
def applier(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    elif isinstance(value, str):
        return value
    elif isinstance(value, np.ndarray):
        return value
    elif hasattr(value, "to"): # Allow handling of custom batch classes
        return fn(value)
    elif isinstance(value, container_abcs.Mapping):
        return {applier(k, fn) : applier(v, fn) for k, v in value.items()}
    elif isinstance(value, container_abcs.Iterable):
        return type(value)(applier(v, fn) for v in value)
    else:
        # Do I want this to fire off even if someone chooses to pass something ordinary like
        # an int or float?  May be more annoying than it's worth.
        # print("Warning:  unrecognized type in applier.  If your input data is a custom class, "
        #     "provide it with a .to(dtype) method which converts its floating-point Tensors to dtype. "
        #     "Amp will check for your custom to() and invoke it to cast the batch's "
        #     "floating-point Tensors to the appropriate type. "
        #     "Also, if your data is a custom class, it is your responsibility to ensure that "
        #     "any Tensors you want to be cuda are already cuda."
        return value


def check_models(models):
    for model in models:
        parallel_type = None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            parallel_type = "torch.nn.parallel.DistributedDataParallel"
        if ('apex_DDP' in sys.modules) and isinstance(model, apex_DDP):
            parallel_type = "apex.parallel.DistributedDataParallel"
        if isinstance(model, torch.nn.parallel.DataParallel):
            parallel_type = "torch.nn.parallel.DataParallel"
        if parallel_type is not None:
            raise RuntimeError("Incoming model is an instance of {}. ".format(parallel_type) +
                "Parallel wrappers should only be applied to the model(s) AFTER \n"
                "the model(s) have been returned from amp.initialize.")


def check_params_fp32(models):
    for model in models:
        for name, param in model.named_parameters():
            if param.is_floating_point():
                if 'Half' in param.type():
                    warn_or_err("Found param {} with type {}, expected torch.npu.FloatTensor.\n"
                        "When using amp.initialize, you do not need to call .half() on your model\n"
                        "before passing it, no matter what optimization level you choose.".format(
                        name, param.type()))
                elif not 'npu' in param.type():
                    warn_or_err("Found param {} with type {}, expected torch.npu.FloatTensor.\n"
                        "When using amp.initialize, you need to provide a model with parameters\n"
                        "located on a Npu device before passing it no matter what optimization level\n"
                        "you chose. Use model.to('npu') to use the default device.".format(
                        name, param.type()))

        # Backward compatibility for PyTorch 0.4
        if hasattr(model, 'named_buffers'):
            buf_iter = model.named_buffers()
        else:
            buf_iter = model._buffers
        for obj in buf_iter:
            if type(obj)==tuple:
                name, buf = obj
            else:
                name, buf = obj, buf_iter[obj]
            if buf.is_floating_point():
                if 'Half' in buf.type():
                    warn_or_err("Found buffer {} with type {}, expected torch.npu.FloatTensor.\n"
                        "When using amp.initialize, you do not need to call .half() on your model\n"
                        "before passing it, no matter what optimization level you choose.".format(
                        name, buf.type()))
                elif not 'npu' in buf.type():
                    warn_or_err("Found buffer {} with type {}, expected torch.npu.FloatTensor.\n"
                        "When using amp.initialize, you need to provide a model with buffers\n"
                        "located on a Npu device before passing it no matter what optimization level\n"
                        "you chose. Use model.to('npu') to use the default device.".format(
                        name, buf.type()))


def check_optimizers(optimizers):
    for optim in optimizers:
        bad_optim_type = None
        if isinstance(optim, FP16_Optimizer_general):
            bad_optim_type = "apex.fp16_utils.FP16_Optimizer"
        if isinstance(optim, FP16_Optimizer_for_fused):
            bad_optim_type = "apex.optimizers.FP16_Optimizer"
        if bad_optim_type is not None:
            raise RuntimeError("An incoming optimizer is an instance of {}. ".format(bad_optim_type) +
                               "The optimizer(s) passed to amp.initialize() must be bare \n"
                               "instances of either ordinary Pytorch optimizers, or Apex fused \n"
                               "optimizers.\n")


class O2StateDictHook(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        for key in state_dict:
            param = state_dict[key]
            if 'Half' in param.type():
                param = param.to(torch.float32)
                state_dict[key] = param


def _initialize(models, optimizers, properties, num_losses=1, cast_model_outputs=None):
    from .amp import init as amp_init

    optimizers_was_list = False
    if isinstance(optimizers, torch.optim.Optimizer) or ('LARC' in globals() and isinstance(optimizers, LARC)):
        optimizers = [optimizers]
    elif optimizers is None:
        optimizers = []
    elif isinstance(optimizers, list):
        optimizers_was_list = True
        check_optimizers(optimizers)
    else:
        check_optimizers([optimizers])
        raise TypeError("optimizers must be either a single optimizer or a list of optimizers.")

    if isinstance(models, torch.nn.Module):
        models_was_list = False
        models = [models]
    elif isinstance(models, list):
        models_was_list = True
    else:
        raise TypeError("models must be either a single model or a list of models.")

    check_models(models)

    if not _amp_state.allow_incoming_model_not_fp32:
        check_params_fp32(models)

    # In the future, when FP16_Optimizer can be deprecated and master weights can
    # become an attribute, remember to stash master weights before casting the model.

    if properties.cast_model_type:
        if properties.keep_batchnorm_fp32:
            for model in models:
                convert_network(model, properties.cast_model_type)
        else:
            for model in models:
                model.to(properties.cast_model_type)

        input_caster = functools.partial(to_type, properties.cast_model_type)
        if cast_model_outputs is not None:
            output_caster = functools.partial(to_type, cast_model_outputs)
        else:
            output_caster = functools.partial(to_type, torch.float32)

        for model in models:
            # Patch the forward method to cast incoming data to the correct type, and
            # outgoing data to float32, so "the user never needs to call .half()."
            # I like writing things explicitly more than decorators.
            def patch_forward(old_fwd):
                def new_fwd(*args, **kwargs):
                    output = old_fwd(*applier(args, input_caster),
                                     **applier(kwargs, input_caster))
                    return applier(output, output_caster)
                return new_fwd

            model.forward = patch_forward(model.forward)

        # State dict trick to recast any preexisting per-param state tensors
        for optimizer in optimizers:
            optimizer.load_state_dict(optimizer.state_dict())

        # patch model.state_dict() to return float32 params
        for model in models:
            for module in model.modules():
                module._register_state_dict_hook(O2StateDictHook(functools.partial(to_type, torch.float32)))

    elif cast_model_outputs is not None:
        output_caster = functools.partial(to_type, cast_model_outputs)

        for model in models:
            def patch_forward(old_fwd):
                def new_fwd(*args, **kwargs):
                    output = old_fwd(*args, **kwargs)
                    return applier(output, output_caster)
                return new_fwd

            model.forward = patch_forward(model.forward)

    for i, optimizer in enumerate(optimizers):
        optimizers[i] = _process_optimizer(optimizer, properties)

    _amp_state.loss_scalers = []
    for _ in range(num_losses):
        _amp_state.loss_scalers.append(LossScaler(properties.loss_scale,
                                                  init_scale=_amp_state.dynamic_init_scale,
                                                  scale_growth_factor=_amp_state.scale_growth_factor,
                                                  scale_backoff_factor=_amp_state.scale_backoff_factor,
                                                  scale_window=_amp_state.scale_window,
                                                  min_loss_scale=_amp_state.min_loss_scale,
                                                  max_loss_scale=_amp_state.max_loss_scale))

    if properties.patch_torch_functions:
        # handle is unused here. It's accessible later through a global value anyway.
        handle = amp_init(loss_scale=properties.loss_scale,
                          verbose=(_amp_state.verbosity == 2),
                          user_cast_preferred=properties.user_cast_preferred)
        for optimizer in optimizers:
            # Disable Amp casting for the optimizer step, because it should only be
            # applied to FP32 master params anyway.
            def patch_step(old_step):
                def new_step(self, *args, **kwargs):
                    with disable_casts():
                        output = old_step(*args, **kwargs)
                    return output
                return new_step

            optimizer.step = MethodType(patch_step(optimizer.step), optimizer)


    is_npu_fused_optimizer = False
    for optimizer in optimizers:
        if hasattr(optimizer, 'is_npu_fused_optimizer') and optimizer.is_npu_fused_optimizer:
            is_npu_fused_optimizer = True
            break
    if properties.combine_grad or is_npu_fused_optimizer:
        torch.nn.Module.zero_grad = zero_grad
        maybe_print(
            "Warning: "
            "Default value of `set_to_none` in torch.nn.Module.zero_grad() is set as False for combine grad, "
            "which is True since torch 2.0.")

    if properties.combine_ddp:
        for model in models:
            for name, param in model.named_parameters():
                dist.broadcast(param, 0)

    if optimizers_was_list:
        if models_was_list:
            return models, optimizers
        else:
            return models[0], optimizers
    else:
        if models_was_list:
            if len(optimizers) == 0:
                return models
            else:
                return models, optimizers[0]
        else:
            if len(optimizers) == 0:
                return models[0]
            else:
                return models[0], optimizers[0]
