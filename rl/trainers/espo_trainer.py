import math
import random
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate.utils import gather_object, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerCallback
from transformers.utils import is_peft_available
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import print_prompt_completions_sample

from collections import defaultdict

from rl.trainers.train_utils import get_mask_id
from rl.trainers.train_utils import is_dream_model, apply_dream_logits_shift
from rl.llada2_compat import is_llada2_moe, generate_llada2

if is_peft_available():
    from peft import PeftConfig


# RewardFunc: callable(prompts, completions, **kwargs) -> list[float]
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

def _nanmin(x: torch.Tensor) -> torch.Tensor:
    """Compatibility shim for TRL versions without nanmin/nanmax helpers."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    finite = torch.isfinite(x)
    if finite.any():
        return x[finite].min()
    # If all NaN/Inf, return NaN to match typical semantics
    return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)


def _nanmax(x: torch.Tensor) -> torch.Tensor:
    """Compatibility shim for TRL versions without nanmin/nanmax helpers."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    finite = torch.isfinite(x)
    if finite.any():
        return x[finite].max()
    return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)


def _split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """Split tensor dict on dim0; keep non-tensors as-is."""
    chunks = []
    for i in range(num_chunks):
        chunk: dict[str, Optional[torch.Tensor]] = {}
        for key, val in tensor_dict.items():
            if isinstance(val, torch.Tensor):
                chunk_size = val.shape[0] // num_chunks
                chunk[key] = val[i * chunk_size : (i + 1) * chunk_size]
            else:
                chunk[key] = val
        chunks.append(chunk)
    return chunks


def _forward_process(batch, prompt_index, mask_id, seed=None):
    """Same forward masking process as dLLM-ESPO."""
    set_seed(seed.item() if isinstance(seed, torch.Tensor) else int(seed))
    b, l = batch.shape
    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(0, target_len + 1, (), device=batch.device)
    x = torch.round(
        torch.linspace(
            float(k),
            k + (b - 1) * ((target_len + 1) / b),
            steps=b,
            device=batch.device,
        )
    ).long()
    x = x % (target_len + 1)

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat(
        (
            torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device),
            is_mask,
        ),
        dim=1,
    )
    noisy_batch = torch.where(is_mask, mask_id, batch)
    return noisy_batch, (x / (target_len + 1)).unsqueeze(1).repeat(1, l)


def _compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """Schulman KL approximation estimators (as in dLLM-ESPO)."""
    if kl_estimator == "k1":
        log_ratio = log_probs - log_probs_base
        log_ratio += (log_probs - log_probs.detach()) * (
            log_probs.detach() - log_probs_base
        )
    elif kl_estimator == "k2":
        log_ratio = log_probs - log_probs_base
        log_ratio = log_ratio**2 / 2.0
    elif kl_estimator == "k3":
        log_ratio = log_probs - log_probs_base
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - log_ratio - 1
    else:
        raise ValueError(f"Unknown kl_estimator: {kl_estimator}")
    return log_ratio


class ESPOTrainer(GRPOTrainer):
    """
    Port of dLLM-ESPO ESPOTrainer into dLLM-R1.

    Notes:
    - Keeps GRPOTrainer integration so the rest of R1 pipeline (datasets/rewards/callbacks) stays compatible.
    - Uses DiffuGRPOConfig fields via `args`:
      - num_iterations, gradient_accumulation_steps, max_completion_length, diffusion_steps, block_length
      - temperature (R1) instead of generation_temperature (ESPO)
      - espo_num_mc, espo_reduce_var
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        # Align mask token id with R1 model family
        if self.processing_class is not None:
            # train_utils.get_mask_id(tokenizer=..., model=...) auto-detects per model family
            self.processing_class.mask_token_id = get_mask_id(
                tokenizer=self.processing_class, model=self.model
            )
        self.num_mc = int(getattr(self.args, "espo_num_mc", 2))
        self.espo_reduce_var = bool(getattr(self.args, "espo_reduce_var", True))
        # TRL version compatibility: some releases do not expose epsilon_low/high on GRPOTrainer.
        # In this repo we use symmetric clipping with epsilon from DiffuGRPOConfig.
        eps = float(getattr(self.args, "epsilon", 0.2))
        self.epsilon_low = eps
        self.epsilon_high = eps
        # Ensure attribute exists regardless of TRL internals.
        self.num_iterations = int(getattr(self.args, "num_iterations", 1))
        # GRPO group size (k generations per prompt). Some TRL versions keep it on the base trainer;
        # we materialize it here so reward/advantage normalization can match dLLM-ESPO semantics.
        self.num_generations = int(getattr(self.args, "num_generations", 1) or 1)
        self._buffered_inputs = None
        self._step = 0

    def add_gumbel_noise(self, logits, temperature, dtype):
        if temperature == 0.0:
            return logits
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def _get_num_transfer_tokens(self, mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = base.expand(-1, steps).clone()
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1
        return num_transfer_tokens.to(torch.int64)

    @torch.no_grad()
    def _llada_generate(
        self,
        model,
        attention_mask,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=None,
    ):
        # This code implement the same generation as dLLM-ESPO (block-wise token transfer), with R1 compatibility:
        # - Auto-detect mask_id (LLaDA/LLaDA2/Dream/SDAR/TraDo)
        # - Use official LLaDA2-MoE generation when needed.
        if mask_id is None:
            mask_id = get_mask_id(tokenizer=self.processing_class, model=model)

        if is_llada2_moe(model):
            # LLaDA2-MoE needs block-aligned incremental generation.
            return generate_llada2(
                model,
                prompt,
                tokenizer=self.processing_class,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
            )

        bs = prompt.shape[0]
        dtype = model.dtype
        x = torch.full(
            (bs, prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, gen_length), value=1)

        prompt_index = x != mask_id
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length
            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = self._get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_, attention_mask=attention_mask).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, attention_mask=attention_mask).logits

                logits_with_noise = self.add_gumbel_noise(
                    logits, temperature=temperature, dtype=dtype
                )
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(dtype), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, end_idx:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_index = torch.topk(confidence[j], k=num_tokens)
                        transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
        return x

    def _get_logits(self, model, batch):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch).logits
        # Dream models require a logits alignment shift (see wd1_grpo_trainer.py).
        if is_dream_model(model):
            logits = apply_dream_logits_shift(logits, model)
        return logits

    def _get_elbo(
        self, model, input_ids, logits_to_keep, mask_seeds, reduce_var: bool = True
    ):
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device

        assert len(mask_seeds) == num_iterations
        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True

        all_perturbed_seqs = []
        all_expanded_inputs = []
        all_p_masks = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]
            perturbed_seq, p_mask = _forward_process(
                expanded_input, prompt_index, self.processing_class.mask_token_id, seed=mask_seed
            )
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)
            all_p_masks.append(p_mask)

        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)
        expanded_input = torch.cat(all_expanded_inputs, dim=0)
        p_mask = torch.cat(all_p_masks, dim=0)

        targets_kept = expanded_input[:, -logits_to_keep:]
        p_mask_kept = p_mask[:, -logits_to_keep:]
        loss = torch.zeros(
            num_iterations * batch_size,
            logits_to_keep,
            device=device,
            dtype=p_mask.dtype,
        )
        mask_index_kept = (perturbed_seq == self.processing_class.mask_token_id)[
            :, -logits_to_keep:
        ]

        logits = self._get_logits(model, perturbed_seq)
        logits_kept = logits[:, -logits_to_keep:]
        loss[mask_index_kept] = (
            F.cross_entropy(
                logits_kept[mask_index_kept],
                targets_kept[mask_index_kept],
                reduction="none",
            )
            / p_mask_kept[mask_index_kept]
        )

        if reduce_var:
            coupled_perturbed_seq = expanded_input.clone()
            coupled_perturbed_seq[:, -logits_to_keep:] = torch.where(
                mask_index_kept,
                coupled_perturbed_seq[:, -logits_to_keep:],
                self.processing_class.mask_token_id,
            )
            coupled_logits = self._get_logits(model, coupled_perturbed_seq)
            coupled_logits_kept = coupled_logits[:, -logits_to_keep:]
            loss[~mask_index_kept] = (
                F.cross_entropy(
                    coupled_logits_kept[~mask_index_kept],
                    targets_kept[~mask_index_kept],
                    reduction="none",
                )
                / (logits_to_keep / (logits_to_keep + 1) - p_mask_kept[~mask_index_kept])
            )
            loss /= 2

        loss = -loss.view(num_iterations, batch_size, logits_to_keep).permute(1, 0, 2)
        return loss.sum(dim=-1)

    def _get_elbo_mc(
        self, model, input_ids, logits_to_keep, mask_seeds, reduce_var=True, num_mc=1
    ):
        assert mask_seeds.shape[-1] == num_mc
        mc_losses = []
        for mc_idx in range(num_mc):
            loss_single = self._get_elbo(
                model,
                input_ids,
                logits_to_keep,
                mask_seeds[:, mc_idx],
                reduce_var=reduce_var,
            )
            mc_losses.append(loss_single)
        return torch.stack(mc_losses, dim=0).mean(dim=0)

    def _get_elbo_by_chunk(
        self, model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, device
    ):
        elbos = torch.zeros(
            (prompt_completion_ids_expanded.shape[1], self.num_iterations), device=device
        )
        local_batch_size = (
            prompt_completion_ids_expanded.shape[1] // self.args.gradient_accumulation_steps
        )
        for i in range(self.args.gradient_accumulation_steps):
            for j in range(self.num_iterations):
                elbos[
                    i * local_batch_size : (i + 1) * local_batch_size, j
                ] = self._get_elbo_mc(
                    model,
                    prompt_completion_ids_expanded[
                        j : j + 1,
                        i * local_batch_size : (i + 1) * local_batch_size,
                        :,
                    ],
                    logits_to_keep,
                    mask_seeds[i][j : j + 1],
                    num_mc=self.num_mc,
                    reduce_var=self.espo_reduce_var,
                )[:, 0]
        return elbos

    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                accumulated_local_batch = self._generate_and_score_completions(
                    accumulated_local_batch
                )
                self._buffered_inputs = _split_tensor_dict(
                    accumulated_local_batch, self.args.gradient_accumulation_steps
                )
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(accumulated_local_batch)
        return inputs

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 与 DiffuGRPOTrainer 一致：跳过 HF 默认 evaluation_loop（其 batch 为 dict，
        # 与 GRPO 期望的「样本 dict 列表」不兼容）。周期性评测由 AccuracyEvalCallback 等完成。
        self._memory_tracker.start()
        metrics = {}
        for callback in self.callback_handler.callbacks:
            callback.on_evaluate(
                self.args,
                self.state,
                self.control,
                accelerator=self.accelerator,
                model=self.model,
            )
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def _generate_and_score_completions(self, inputs):
        if self.control.should_evaluate:
            return {
                "prompt_ids": None,
                "prompt_mask": None,
                "completion_ids": None,
                "completion_mask": None,
                "advantages": None,
                "old_per_token_logps": None,
                "ref_per_token_logps": None,
                "mask_seeds": None,
            }
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        gen_length = self.args.max_completion_length
        steps = self.args.diffusion_steps
        temperature = float(getattr(self.args, "temperature", 0.0))
        block_length = int(getattr(self.args, "block_length", 32))
        cfg_scale = float(getattr(self.args, "cfg_scale", 0.0))
        remasking = getattr(self.args, "remasking", "low_confidence")

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = int(getattr(self.args, "generation_batch_size", 4))
            prompt_completion_ids_all = []
            use_br1 = bool(getattr(self.args, "use_block_r1_dataset", False))
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]
                chunk_inputs = inputs[i:end_idx]
                cfg = getattr(unwrapped_model, "config", None)
                model_type = str(getattr(cfg, "model_type", "") or "").lower()
                name = str(getattr(cfg, "name_or_path", "") or "")
                is_llada = ("llada" in model_type) or ("LLaDA" in name)

                if is_llada:
                    if use_br1 and chunk_inputs:
                        from rl.trainers.block_r1_trainer import snap_block_size_for_gen_length

                        fb = int(block_length)
                        gl = int(gen_length)
                        b_list = [
                            snap_block_size_for_gen_length(
                                int(x.get("br1_best_block_size", fb)), gl, fallback=fb
                            )
                            for x in chunk_inputs
                        ]
                        bs_chunk = batch_prompt_ids.size(0)
                        if len(b_list) != bs_chunk:
                            b_list = [fb] * bs_chunk
                        groups: dict[int, list[int]] = defaultdict(list)
                        for j, b in enumerate(b_list):
                            groups[int(b)].append(j)
                        slices: list = [None] * bs_chunk
                        mid = get_mask_id(
                            tokenizer=self.processing_class, model=unwrapped_model
                        )
                        for bl, idxs in groups.items():
                            idx_t = torch.tensor(
                                idxs, dtype=torch.long, device=batch_prompt_ids.device,
                            )
                            sub_p = batch_prompt_ids.index_select(0, idx_t)
                            sub_m = batch_prompt_mask.index_select(0, idx_t)
                            sub_out = self._llada_generate(
                                model=unwrapped_model,
                                attention_mask=sub_m,
                                prompt=sub_p,
                                steps=steps,
                                gen_length=gen_length,
                                block_length=int(bl),
                                temperature=temperature,
                                cfg_scale=cfg_scale,
                                remasking=remasking,
                                mask_id=mid,
                            )
                            for li, gi in enumerate(idxs):
                                slices[gi] = sub_out[li : li + 1]
                        batch_prompt_completion_ids = torch.cat(slices, dim=0)
                    else:
                        batch_prompt_completion_ids = self._llada_generate(
                            model=unwrapped_model,
                            attention_mask=batch_prompt_mask,
                            prompt=batch_prompt_ids,
                            steps=steps,
                            gen_length=gen_length,
                            block_length=block_length,
                            temperature=temperature,
                            cfg_scale=cfg_scale,
                            remasking=remasking,
                            mask_id=get_mask_id(tokenizer=self.processing_class, model=unwrapped_model),
                        )
                    prompt_completion_ids_all.append(batch_prompt_completion_ids)
                else:
                    out = unwrapped_model.diffusion_generate(
                        batch_prompt_ids,
                        attention_mask=batch_prompt_mask,
                        max_new_tokens=gen_length,
                        output_history=False,
                        return_dict_in_generate=True,
                        steps=steps,
                        temperature=temperature,
                        top_p=0.95 if temperature > 0 else 1.0,
                        alg="entropy",
                        alg_temp=0.0,
                        mask_token_id=get_mask_id(tokenizer=self.processing_class, model=unwrapped_model),
                    )
                    prompt_completion_ids_all.append(out.sequences)

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(1)

        mask_seeds = torch.randint(
            0,
            2**12,
            (
                self.args.gradient_accumulation_steps,
                self.num_iterations,
                self.num_mc,
            ),
            device=device,
        )
        prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
            self.num_iterations, -1, -1
        )

        with torch.no_grad():
            old_per_token_logps = (
                self._get_elbo_by_chunk(
                    self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, device
                )
                if self.num_iterations > 1
                else None
            )

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_elbo_by_chunk(
                        self.ref_model,
                        prompt_completion_ids_expanded,
                        logits_to_keep,
                        mask_seeds,
                        device,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_elbo_by_chunk(
                            self.model,
                            prompt_completion_ids_expanded,
                            logits_to_keep,
                            mask_seeds,
                            device,
                        )

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"]
                    if prompt[-1]["role"] == "assistant"
                    else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                reward_func_name = (
                    f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                )
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                # reward funcs in R1 expect prompts, completions, run_name, step, rank, **kwargs
                rewards_list = reward_func(
                    prompts=prompts,
                    completions=completions,
                    run_name=getattr(self.args, "run_name", ""),
                    step=getattr(self.state, "global_step", None),
                    rank=getattr(self.accelerator, "process_index", None),
                    **{k: [x.get(k) for x in inputs] for k in inputs[0].keys() if k != "prompt"},
                )
                # Match dLLM-ESPO semantics: None -> NaN
                rewards_list = [
                    (r if r is not None else float("nan")) for r in rewards_list
                ]
                rewards_per_func[:, i] = torch.tensor(
                    rewards_list, device=device, dtype=torch.float32
                )

        # === This is to Match dLLM-ESPO reward aggregation & advantage computation ===
        # Gather rewards across processes so grouped advantages are computed on the global (batch*k) layout.
        rewards_per_func = self.accelerator.gather_for_metrics(rewards_per_func)

        if self.reward_weights is not None:
            if isinstance(self.reward_weights, torch.Tensor):
                weights = self.reward_weights.to(device=rewards_per_func.device, dtype=torch.float32)
            else:
                weights = torch.tensor(
                    self.reward_weights, device=rewards_per_func.device, dtype=torch.float32
                )
            rewards = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1)
        else:
            rewards = rewards_per_func.nansum(dim=1)

        k = int(self.num_generations)
        scale_rewards = bool(getattr(self, "scale_rewards", False))
        leave_one_out = True

        if k > 1:
            rewards_grouped = rewards.view(-1, k)  # (batch, k)
            if leave_one_out:
                sum_group = rewards_grouped.sum(dim=1, keepdim=True)  # (batch, 1)
                baseline = (sum_group - rewards_grouped) / (k - 1)
                advantages = (rewards_grouped - baseline).view(-1)  # (batch*k,)
            else:
                mean_grouped = rewards_grouped.mean(dim=1, keepdim=True)
                advantages = (rewards_grouped - mean_grouped).view(-1)

            std_grouped = rewards_grouped.std(dim=1, keepdim=True)  # (batch, 1)
            std_grouped = std_grouped.repeat_interleave(k, dim=1).view(-1)  # (batch*k,)
            if scale_rewards:
                advantages = advantages / (std_grouped + 1e-4)
        else:
            advantages = rewards - rewards.mean()

        # Slice advantages back to this process' local rows (same as dLLM-ESPO).
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps if old_per_token_logps is not None else torch.zeros((len(prompts), self.num_iterations), device=device),
            "ref_per_token_logps": ref_per_token_logps if ref_per_token_logps is not None else torch.zeros((len(prompts), self.num_iterations), device=device),
            "mask_seeds": mask_seeds,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        mask_seeds = inputs["mask_seeds"][0]  # [num_iterations, num_mc]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        batch_size, logits_to_keep = completion_ids.shape

        this_itr_idx = (
            ((self._step - 1) % (self.args.num_iterations * self.args.gradient_accumulation_steps))
            // self.args.gradient_accumulation_steps
        )
        this_itr_mask_seed = mask_seeds[this_itr_idx : this_itr_idx + 1]
        input_ids = input_ids.unsqueeze(0)

        per_token_logps = self._get_elbo_mc(
            model,
            input_ids,
            logits_to_keep,
            this_itr_mask_seed,
            num_mc=self.num_mc,
            reduce_var=self.espo_reduce_var,
        )

        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][:, this_itr_idx].unsqueeze(1)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )

        coef_1 = torch.exp((per_token_logps - old_per_token_logps) / logits_to_keep)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.view(-1, 1)
        per_token_loss2 = coef_2 * advantages.view(-1, 1)
        loss = -torch.min(per_token_loss1, per_token_loss2).sum() / batch_size

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][:, this_itr_idx].unsqueeze(1)
            kl = _compute_approx_kl(per_token_logps, ref_per_token_logps, "k2")
            mean_kl = kl.sum() / (batch_size * logits_to_keep)
            loss += self.beta * mean_kl

        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather_for_metrics(
                -per_token_logps.sum() / (batch_size * logits_to_keep)
            )
            .mean()
            .item()
        )
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        low_clip = is_low_clipped.float().mean()
        high_clip = is_high_clipped.float().mean()
        clip_ratio = is_region_clipped.float().mean()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(_nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(_nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss

