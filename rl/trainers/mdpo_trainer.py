"""
This code follows MDPO's implementation
"""

from itertools import groupby
from typing import Any, Callable, Optional, Sized, Union

import torch
import torch.nn.functional as F
from accelerate.utils import gather
from datasets import Dataset, IterableDataset
from torch.utils.data import Sampler
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.utils import is_peft_available
from trl.extras.profiling import profiling_decorator
from trl.trainer.grpo_config import GRPOConfig
from rl.trainers.gdpo_trainer import GDPOTrainer

if is_peft_available():
    from peft import PeftConfig, PeftModel

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def _is_peft_model(model):
    if not is_peft_available():
        return False
    return isinstance(model, PeftModel)


def extract_target_from_pred(
    pred: str,
    target_res,
    timeout_seconds: int,
    fallback_mode="no_fallback",
    extraction_mode="any_match",
):
    """Best-effort target extraction kept for MDPO source compatibility."""
    try:
        from math_verify.parser import extract_match
    except Exception:
        return [], []

    extracted_predictions = []
    fallbacks = []
    all_patterns = [
        (pattern, target_type, priority)
        for target_patterns, target_type in target_res
        for pattern, priority in target_patterns
    ]

    match_found = False
    string_matches = []
    sorted_patterns = sorted(all_patterns, key=lambda x: x[2])
    grouped_patterns = list(
        (gr, list(val)) for gr, val in groupby(sorted_patterns, key=lambda x: x[2])
    )
    for _, patterns_group in grouped_patterns:
        matches_with_pos = (
            (match, match.start(), match.end(), target_type)
            for pattern, target_type, _ in patterns_group
            for match in pattern.finditer(pred)
        )
        matches_with_pos = sorted(
            matches_with_pos, key=lambda x: (x[2], -x[1]), reverse=True
        )

        for match, _, _, target_type in matches_with_pos:
            extracted_match, str_fallback = extract_match(
                match, target_type, timeout_seconds=timeout_seconds
            )

            match_found = True
            if str_fallback:
                fallbacks.append(str_fallback)

            if extracted_match is not None:
                string_matches.append(match)
                extracted_predictions.append(extracted_match)
                break

            if extraction_mode == "first_match":
                break

        if extracted_predictions or (match_found and extraction_mode == "first_match"):
            break

    if fallback_mode == "first_match" and fallbacks:
        extracted_predictions += [fallbacks[0]]

    return extracted_predictions, string_matches


def find_subtensor_mask(tensor, subtensor, method="sliding"):
    """Find positions of a subtensor within a larger tensor."""
    if method in {"sliding", "unique"}:
        mask = torch.zeros_like(tensor, dtype=torch.float)
        for i in range(len(tensor) - len(subtensor) + 1):
            window = tensor[i : i + len(subtensor)]
            if torch.equal(window, subtensor):
                mask[i : i + len(subtensor)] = 1
        return mask
    raise ValueError(f"Unsupported method: {method}")


class RepeatRandomSampler(Sampler):
    """Sampler kept for MDPO compatibility (repeat prompts across updates)."""

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class MDPOTrainer(GDPOTrainer):
    """
    Masked Diffusion Policy Optimization (MDPO) trainer.

    This implementation is intentionally API-compatible with existing trainers
    used in this repository (e.g. GDPOTrainer) so it can be plugged into the
    same training/evaluation pipeline.

    To integrate with MDPO, I need to:
    - Reuses GDPO generation/scoring path to keep data flow and callbacks stable.
    - Replaces the policy loss with an MDPO-style clipped objective.
    - Supports optional confidence weighting when `inputs["conf"]` is provided.
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
        ] = (
            None,
            None,
        ),
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

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset):
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def split_into_micro_batches(self, traj):
        """Source-compatible helper kept for MDPO trajectories."""
        prompt_ids, prompt_mask = traj["prompt_ids"], traj["prompt_mask"]
        advantages = traj["advantages"]
        logits_to_keep = traj["logits_to_keep"]
        all_steps_input_ids = traj["all_steps_input_ids"]
        all_steps_completion_ids = traj["all_steps_completion_ids"]
        all_confidence = traj["all_confidence"]

        top_steps = torch.topk(
            torch.abs(advantages).sum(dim=0),
            k=min(self.args.sample_train_steps, advantages.shape[-1]),
            dim=-1,
        ).indices

        for step in top_steps:
            step = int(step.item())
            input_answer_ids = all_steps_input_ids[step]
            completion_ids = all_steps_completion_ids[step]

            input_ids = torch.cat(
                [
                    prompt_ids,
                    input_answer_ids.to(prompt_ids.device).to(prompt_ids.dtype),
                ],
                dim=-1,
            )
            target_ids = torch.cat(
                [prompt_ids, completion_ids.to(prompt_ids.device).to(prompt_ids.dtype)],
                dim=-1,
            )
            input_mask = torch.cat(
                [
                    prompt_mask,
                    torch.ones_like(input_answer_ids)
                    .to(prompt_mask.device)
                    .to(prompt_mask.dtype),
                ],
                dim=-1,
            )
            conf = all_confidence[step][:, -logits_to_keep:].to(input_ids.device)

            with torch.no_grad():
                if self.beta != 0.0:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        input_ids,
                        target_ids,
                        input_mask,
                        logits_to_keep,
                    )
                else:
                    ref_per_token_logps = None

            yield {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "target_ids": target_ids,
                "advantages": (advantages[:, step : step + 1])
                .expand_as(input_ids)
                .to(input_ids.device),
                "conf": conf,
                "ref_per_token_logps": ref_per_token_logps,
                "logits_to_keep": logits_to_keep,
                "step": torch.ones_like(input_ids) * step,
            }

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The MDPOTrainer does not support returning outputs")

        # Support both the default GRPO/GDPO batch format and MDPO micro-batch format.
        if "input_ids" in inputs and "target_ids" in inputs:
            logits_to_keep = inputs["logits_to_keep"]
            per_token_logps = self._get_per_token_logps(
                model,
                inputs["input_ids"],
                inputs["target_ids"],
                inputs["input_mask"],
                logits_to_keep,
            )
            confidence = inputs.get("conf", None)

            if self.beta != 0.0:
                ref_per_token_logps = inputs["ref_per_token_logps"]
                per_token_kl = ((ref_per_token_logps - per_token_logps) ** 2) / 2

            completion_mask = inputs["input_ids"][:, -logits_to_keep:] == getattr(
                self.args, "mask_id", 126336
            )
            lambda_t = logits_to_keep / (
                completion_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            )

            old_per_token_logps = (
                inputs["old_per_token_logps"]
                if self.num_iterations > 1 and "old_per_token_logps" in inputs
                else per_token_logps.detach()
            )
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
            per_token_loss1 = coef_1 * inputs["advantages"][:, -logits_to_keep:]
            per_token_loss2 = coef_2 * inputs["advantages"][:, -logits_to_keep:]
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2) * lambda_t
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            if confidence is None:
                confidence = torch.ones_like(per_token_loss)
            loss = (per_token_loss * completion_mask * confidence).sum() / (
                completion_mask.sum().clamp_min(1.0)
            )

            mode = "eval" if self.control.should_evaluate else "train"
            if self.beta != 0.0:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                self._metrics[mode]["kl"].append(
                    self.accelerator.gather_for_metrics(mean_kl).mean().item()
                )
            return loss

        prompt_ids, completion_ids = inputs["prompt_ids"], inputs["completion_ids"]
        completion_mask = inputs["completion_mask"].float()
        mask_seeds = inputs["mask_seeds"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]

        logps, _ = self.logp_estimator.get_log_likelihood(
            self.model,
            input_ids,
            logits_to_keep=logits_to_keep,
            seed=this_itr_mask_seed,
        )

        advantages = inputs["advantages"]
        old_logps = (
            inputs["old_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else logps.detach()
        )

        ratio = torch.exp((logps - old_logps) / completion_ids.shape[-1])
        ratio_clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

        valid_tokens = completion_mask.sum(dim=-1).clamp_min(1.0)
        lambda_t = logits_to_keep / valid_tokens

        loss_unclipped = ratio * advantages
        loss_clipped = ratio_clipped * advantages
        policy_loss = -torch.min(loss_unclipped, loss_clipped) * lambda_t

        if self.beta != 0.0:
            ref_logps = inputs["ref_logps"][this_itr_idx].squeeze(0)
            kl = ((ref_logps - logps) ** 2) / 2
            loss = policy_loss + self.beta * kl
        else:
            kl = None
            loss = policy_loss

        mode = "eval" if self.control.should_evaluate else "train"
        if kl is not None:
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(kl).mean().item()
            )

        is_clipped = (loss_unclipped > loss_clipped).float()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(is_clipped).mean().item()
        )
        self._metrics[mode]["mdpo_lambda_t"].append(
            self.accelerator.gather_for_metrics(lambda_t).mean().item()
        )

        return loss.mean()

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None
