# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
from unittest.mock import patch

import pytest
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device

from trl.import_utils import is_deepspeed_available
from trl.models.utils import (
    disable_gradient_checkpointing,
    prepare_deepspeed,
    prepare_fsdp,
    unwrap_model_for_generation,
)


@pytest.mark.skipif(not is_deepspeed_available(), reason="deepspeed is not installed")
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_prepare_deepspeed_strips_optimizer_for_cpu_offload(stage):
    pytest.importorskip("deepspeed")
    # prepare_deepspeed initializes eval-only models without an optimizer. If the deep-copied config still carries the
    # training-only blocks, DeepSpeed builds a CPUAdam on GPU params (`optimizer` + `offload_optimizer: cpu`) or an LR
    # scheduler with no optimizer (`scheduler`), and the run dies before training. They must be dropped before
    # `deepspeed.initialize`. Intercept the call to check the config it would receive.
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": stage, "offload_optimizer": {"device": "cpu"}},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-5}},
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 1e-5, "warmup_num_steps": 10},
        },
    }
    accelerator = types.SimpleNamespace(
        state=types.SimpleNamespace(deepspeed_plugin=types.SimpleNamespace(deepspeed_config=ds_config))
    )

    captured = {}

    def fake_initialize(model, config):
        captured["config"] = config
        return types.SimpleNamespace(eval=lambda: None), None, None, None

    with patch("deepspeed.initialize", fake_initialize):
        prepare_deepspeed(model=None, accelerator=accelerator)

    assert "optimizer" not in captured["config"]
    assert "scheduler" not in captured["config"]
    assert "offload_optimizer" not in captured["config"]["zero_optimization"]


class TestDisableGradientCheckpointing:
    def test_when_disabled(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        assert model.is_gradient_checkpointing is False
        with disable_gradient_checkpointing(model):
            assert model.is_gradient_checkpointing is False
        assert model.is_gradient_checkpointing is False

    def test_when_enabled(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        model.gradient_checkpointing_enable()
        assert model.is_gradient_checkpointing is True
        with disable_gradient_checkpointing(model):
            assert model.is_gradient_checkpointing is False
        assert model.is_gradient_checkpointing is True


class TestUnwrapModelForGeneration:
    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_restores_gradient_checkpointing_kwargs(self, use_reentrant):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5").to(torch_device)
        model.train()  # checkpointing only runs in training mode
        model.gradient_checkpointing_enable({"use_reentrant": use_reentrant})
        with unwrap_model_for_generation(
            model, Accelerator(), gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        ):
            pass

        input_ids = torch.randint(0, model.config.vocab_size, (1, 8), device=torch_device)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        # Only non-reentrant checkpointing supports `torch.autograd.grad`, so this tells which one is enabled
        if use_reentrant:
            with pytest.raises(RuntimeError, match=r"\.grad\(\)"):
                torch.autograd.grad(loss, list(model.parameters()))
        else:
            torch.autograd.grad(loss, list(model.parameters()))


def _make_fsdp2_accelerator(cpu_ram_efficient=False):
    """Build a minimal mock accelerator for the FSDP2 branch of prepare_fsdp."""
    fsdp_plugin = types.SimpleNamespace(
        fsdp_version=2,
        reshard_after_forward=True,
        cpu_offload=None,
        mixed_precision_policy=None,
        ignored_modules=None,
        cpu_ram_efficient_loading=cpu_ram_efficient,
    )
    parallelism_config = types.SimpleNamespace(fsdp_dim_names=("dp",))
    return types.SimpleNamespace(
        state=types.SimpleNamespace(fsdp_plugin=fsdp_plugin),
        device=torch.device("cpu"),
        torch_device_mesh=None,
        parallelism_config=parallelism_config,
    )


class TestPrepareFsdp2Broadcast:
    @patch("trl.models.utils.get_parameters_from_modules", return_value=set())
    @patch("torch.distributed.fsdp.fully_shard")
    @patch("accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict")
    def test_broadcast_called_when_cpu_ram_efficient(self, mock_load, mock_shard, mock_get_params):
        """When cpu_ram_efficient_loading=True, fsdp2_load_full_state_dict must be called to broadcast weights."""
        model = torch.nn.Linear(4, 4)
        accelerator = _make_fsdp2_accelerator(cpu_ram_efficient=True)

        prepare_fsdp(model, accelerator)

        mock_shard.assert_called_once()
        mock_load.assert_called_once()
        args = mock_load.call_args
        assert args[0][0] is accelerator
        assert args[0][1] is model

    @patch("trl.models.utils.get_parameters_from_modules", return_value=set())
    @patch("torch.distributed.fsdp.fully_shard")
    @patch("accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict")
    def test_no_broadcast_without_cpu_ram_efficient(self, mock_load, mock_shard, mock_get_params):
        """When cpu_ram_efficient_loading=False, fsdp2_load_full_state_dict must NOT be called."""
        model = torch.nn.Linear(4, 4)
        accelerator = _make_fsdp2_accelerator(cpu_ram_efficient=False)

        prepare_fsdp(model, accelerator)

        mock_shard.assert_called_once()
        mock_load.assert_not_called()

    @patch("trl.models.utils.get_parameters_from_modules", return_value=set())
    @patch("torch.distributed.fsdp.fully_shard")
    @patch("accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict")
    def test_state_dict_captured_before_meta(self, mock_load, mock_shard, mock_get_params):
        """The original state dict must be captured before moving to meta device."""
        model = torch.nn.Linear(4, 4)
        original_weight = model.weight.data.clone()
        accelerator = _make_fsdp2_accelerator(cpu_ram_efficient=True)

        prepare_fsdp(model, accelerator)

        loaded_sd = mock_load.call_args[0][2]
        assert torch.equal(loaded_sd["weight"], original_weight)
