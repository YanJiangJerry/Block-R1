"""
StableDRL trainer: SPG bound + ELBO/EUBO mix + optional SNIS (dLLM-DRL / StableDRL)
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Union

import torch
import torch.nn as nn
import wandb
from accelerate.utils import gather, gather_object
from transformers import Trainer
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_rich_available
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import print_prompt_completions_sample

from rl.trainers.diffu_grpo_trainer import DiffuGRPOTrainer
from rl.llada2_compat import is_llada2_moe, set_llada2_eval_block_length
from rl.trainers.stable_drl_svpo import (
    SPGConfig,
    anti_short_boxed_reward,
    compute_loss_spg,
    generate_spg,
    _get_per_seq_logps_spg,
)
from rl.trainers.train_utils import (
    append_per_domain_reward_metrics,
    grpo_group_normalized_advantages,
)


class StableDRLTrainer(DiffuGRPOTrainer):
    """
    Diffusion LM RL with StableDRL objective (sequence-level SPG + SNIS), sharing
    generation/reward plumbing with :class:`DiffuGRPOTrainer`.
    """

    def _make_spg_config(self) -> SPGConfig:
        a = self.args
        mw = float(getattr(a, "stable_drl_spg_omega", 0.5))
        le = getattr(a, "stable_drl_logp_estimation", None)
        if le is None:
            if mw >= 1.0 - 1e-6:
                le = "eubo"
            elif mw <= 1e-6:
                le = "elbo"
            else:
                le = "mix"
        return SPGConfig(
            forward_type=str(getattr(a, "stable_drl_forward_type", "block_random")),
            num_t=int(getattr(a, "stable_drl_num_mc_samples", 1)),
            min_t=0.0,
            max_t=1.0,
            block_length=int(a.block_length or 32),
            use_mask_prompt=True,
            p_mask_prompt=float(getattr(a, "stable_drl_p_mask_perturb", 0.15)),
            mask_id=int(a.mask_id),
            cfg_scale=float(a.cfg_scale or 0.0),
            logp_estimation=le,
            eubo_beta=float(getattr(a, "stable_drl_spg_beta", 1.5)),
            mix_weight=mw,
            use_snis=bool(getattr(a, "stable_drl_use_snis", False)),
            ais_clip_iw=float(getattr(a, "stable_drl_ais_clip_iw", 5.0)),
        )

    @profiling_decorator
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        if return_outputs:
            raise ValueError("StableDRLTrainer does not support return_outputs")
        spg_cfg = self._make_spg_config()
        this_itr = self._step % max(self.args.num_iterations, 1)
        out = compute_loss_spg(
            model,
            inputs,
            this_itr,
            spg_cfg,
            self.accelerator,
            random_masking=self.args.random_masking,
        )
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["stable_drl/iw_mean"].append(float(out.get("iw_mean", 0.0)))
        self._metrics[mode]["stable_drl/iw_max"].append(float(out.get("iw_max", 0.0)))
        self._metrics[mode]["stable_drl/base_reward"].append(float(out.get("base_reward", 0.0)))
        self._metrics[mode]["stable_drl/anti_reward"].append(float(out.get("anti_reward", 0.0)))
        return out["loss_tensor"]

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        if self.control.should_evaluate:
            return {
                "prompt_ids": None,
                "prompt_mask": None,
                "completion_ids": None,
                "completion_mask": None,
                "ref_per_token_logps": None,
                "mask_ids": None,
            }

        device = self.accelerator.device
        spg_cfg = self._make_spg_config()

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
        default_block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale
        use_fp16 = bool(getattr(self.args, "fp16", False) or getattr(self.args, "bf16", False))

        def _generate_spg_chunk(
            unwrapped_model, p_ids, p_mask, bl_int: int
        ) -> torch.Tensor:
            try:
                if is_llada2_moe(unwrapped_model):
                    set_llada2_eval_block_length(unwrapped_model, bl_int)
            except Exception:
                pass
            chunks = []
            for i in range(0, p_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, p_ids.size(0))
                bp = p_ids[i:end_idx]
                bm = p_mask[i:end_idx]
                chunks.append(
                    generate_spg(
                        model=unwrapped_model,
                        prompt=bp,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=bl_int,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        remasking=self.args.remasking,
                        mask_id=self.args.mask_id,
                        prompt_mask=bm,
                        use_fp16=use_fp16,
                        tokenizer=self.processing_class,
                    )
                )
                del bp, bm
                torch.cuda.empty_cache()
            return torch.cat(chunks, dim=0)

        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator
        ) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            ctrl = getattr(self, "block_size_controller", None)

            use_br1 = bool(getattr(self.args, "use_block_r1_dataset", False))
            ds_block_sizes = None
            if use_br1 and inputs:
                from rl.trainers.block_r1_trainer import snap_block_size_for_gen_length

                fb = int(self.args.block_length or 32)
                gl = int(gen_length)
                ds_block_sizes = [
                    snap_block_size_for_gen_length(
                        int(x.get("br1_best_block_size", fb)), gl, fallback=fb
                    )
                    for x in inputs
                ]
                if len(ds_block_sizes) != prompt_ids.size(0):
                    ds_block_sizes = None

            if ctrl is not None or ds_block_sizes is not None:
                bs = prompt_ids.size(0)
                if ds_block_sizes is not None:
                    block_sizes = ds_block_sizes
                else:
                    block_sizes = ctrl.assign_group_block_sizes(bs)
                if hasattr(self, "_r1_sample_block_sizes"):
                    self._r1_sample_block_sizes.extend(block_sizes)
                groups: dict[int, list[int]] = defaultdict(list)
                for j, b in enumerate(block_sizes):
                    groups[int(b)].append(j)
                slices: list = [None] * bs
                for bl, indices in groups.items():
                    idx_tensor = torch.tensor(
                        indices, dtype=torch.long, device=device,
                    )
                    sub_p = prompt_ids.index_select(0, idx_tensor)
                    sub_pm = prompt_mask.index_select(0, idx_tensor)
                    sub_out = _generate_spg_chunk(unwrapped_model, sub_p, sub_pm, bl)
                    for li, gi in enumerate(indices):
                        slices[gi] = sub_out[li : li + 1]
                prompt_completion_ids = torch.cat(slices, dim=0)
                entropy_rewards_tensor = torch.zeros(bs, device=device)
            else:
                try:
                    if is_llada2_moe(unwrapped_model):
                        set_llada2_eval_block_length(
                            unwrapped_model, default_block_length
                        )
                except Exception:
                    pass
                prompt_completion_ids_all = []
                entropy_rewards_all = []
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx]
                    batch_prompt_mask = prompt_mask[i:end_idx]

                    batch_prompt_completion_ids = generate_spg(
                        model=unwrapped_model,
                        prompt=batch_prompt_ids,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=default_block_length,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        remasking=self.args.remasking,
                        mask_id=self.args.mask_id,
                        prompt_mask=batch_prompt_mask,
                        use_fp16=use_fp16,
                        tokenizer=self.processing_class,
                    )
                    prompt_completion_ids_all.append(batch_prompt_completion_ids)
                    entropy_rewards_all.append(
                        torch.zeros(batch_prompt_ids.size(0), device=device)
                    )
                    del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                    torch.cuda.empty_cache()

                prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
                entropy_rewards_tensor = torch.cat(entropy_rewards_all, dim=0)

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        ref_elbo = None
        if spg_cfg.use_snis:
            self.model.eval()
            input_ids_cat = torch.cat([prompt_ids, completion_ids], dim=1)
            logits_to_keep = completion_ids.size(1)
            ref_mask_seed = (
                torch.randint(0, 2**31 - 1, (1,), device=device).item()
                if self.args.random_masking
                else 42
            )
            dummy_reward_mask = torch.ones(
                input_ids_cat.size(0), dtype=torch.bool, device=device
            )
            with torch.no_grad():
                _, ref_elbo = _get_per_seq_logps_spg(
                    self.model,
                    input_ids_cat.unsqueeze(0),
                    logits_to_keep,
                    [ref_mask_seed],
                    prompt_mask,
                    completion_mask,
                    reward_mask=dummy_reward_mask,
                    args=spg_cfg,
                )
            ref_elbo = ref_elbo.squeeze(0).detach()
        self.model.train()

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        reward_kwargs: dict = {}
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
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {
                    key: [example[key] for example in inputs] for key in keys
                }
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.run_name,
                    **reward_kwargs,
                )
                output_reward_func = [
                    reward if reward is not None else torch.nan
                    for reward in output_reward_func
                ]
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        base_rewards_local = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        anti_local = torch.zeros(len(prompts), device=device, dtype=torch.float32)
        if getattr(self.args, "stable_drl_anti_short_boxed", True):
            try:
                anti_local = anti_short_boxed_reward(
                    {}, completions_text, self.num_generations, device
                ).to(device=device, dtype=torch.float32)
            except Exception:
                anti_local = torch.zeros(len(prompts), device=device, dtype=torch.float32)

        rewards_per_func = gather(rewards_per_func)
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)
        rewards = rewards + gather(anti_local.to(device))
        entropy_rewards_tensor = gather(entropy_rewards_tensor)
        rewards = rewards + entropy_rewards_tensor.to(rewards.device)

        # dLLM-DRL uses group mean only (generate_and_score_completions_spg); d1/wd1 here use GRPO std norm.
        adv_mode = str(getattr(self.args, "stable_drl_advantage_mode", "center")).lower()
        if adv_mode == "grpo_std":
            advantages, std_grouped_rewards, zero_std_ratio = (
                grpo_group_normalized_advantages(rewards, self.num_generations)
            )
        elif adv_mode == "center":
            ng = self.num_generations
            rv = rewards.view(-1, ng)
            mean_rep = rv.mean(dim=1).repeat_interleave(ng, dim=0)
            advantages = rewards - mean_rep
            std_g = rv.std(dim=1)
            std_grouped_rewards = std_g.repeat_interleave(ng, dim=0)
            zero_std_ratio = (
                float((std_g < 1e-6).sum().item()) / float(std_g.numel())
                if std_g.numel() > 0
                else 0.0
            )
        else:
            raise ValueError(
                f"stable_drl_advantage_mode must be 'center' or 'grpo_std', got {adv_mode!r}"
            )

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        reward_mask = (advantages > 0).bool()

        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            if reward_func_name != "_route":
                self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if inputs and "domain" in inputs[0]:
            domain_per_sample_local = [example.get("domain") for example in inputs]
            append_per_domain_reward_metrics(
                self._metrics[mode],
                rewards,
                domain_per_sample_local,
                gather_object,
            )

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()
            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                else:
                    print(
                        f"[StableDRL log_completions step={self.state.global_step}] "
                        f"(pip install rich for full table)",
                        flush=True,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    import pandas as pd

                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        mask_seeds = torch.zeros(self.num_iterations, dtype=torch.long, device=device)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": [],
            "ref_per_token_logps": None,
            "advantages": advantages,
            "mask_seeds": mask_seeds,
            "reward_mask": reward_mask,
            "base_rewards": base_rewards_local,
            "anti_rewards": anti_local,
            "rewards": (base_rewards_local + anti_local).detach(),
            "ref_elbo": ref_elbo,
            "completion_length": completion_length,
        }


__all__ = ["StableDRLTrainer"]
