"""
This code follows GDPO's implementation
"""

import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    Trainer,
)
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)

from rl.trainers.likelihood_estimators import get_estimator
from rl.trainers.dynamic_generate import dynamic_generate
from rl.trainers.train_utils import (
    append_per_domain_reward_metrics,
    get_mask_id,
    grpo_group_normalized_advantages,
    is_dream_model,
    apply_dream_logits_shift,
)
from rl.llada2_compat import is_llada2_moe, generate_llada2

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GDPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
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
        # IMPORTANT: Prevent Trainer from removing columns needed for reward computation
        # Without this, columns like "target" and "numbers" are removed during evaluation
        if args is not None:
            args.remove_unused_columns = False

        # Initialize the parent class
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

        # Force remove_unused_columns=False again after parent init (parent may override)
        self.args.remove_unused_columns = False

        self.logp_estimator = get_estimator(args.logp_estimator)

    def _set_signature_columns_if_needed(self):
        """
        Override parent method to preserve all dataset columns needed for reward computation.

        The parent GRPOTrainer only keeps "prompt" column, but we need additional columns
        like "target" and "numbers" for the countdown reward function.
        """
        # Set to None to prevent column removal, or include all needed columns
        # Setting to None with remove_unused_columns=False should preserve all columns
        if self._signature_columns is None:
            # Include all columns that reward functions might need
            self._signature_columns = [
                "prompt",
                "target",
                "numbers",
                "answer",
                "completion",
            ]

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Skip the normal evaluation and just run the callbacks.
        This is needed because the eval_callback expects accelerator and model in kwargs.
        """
        self._memory_tracker.start()
        metrics = {}

        # Run callbacks with accelerator and model
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

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
 
        logps, _ = self.logp_estimator.get_log_likelihood(
            self.model,
            input_ids,
            logits_to_keep=logits_to_keep,
            seed=this_itr_mask_seed,
        )

        # Default KL divergence to zero in GDPO
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_logps = inputs["ref_logps"][this_itr_idx].squeeze(0)
            kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1

        # Compute the loss using the GRPO objective
        advantages = inputs["advantages"]
        old_logps = (
            inputs["old_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else logps.detach()
        )
        coef_1 = torch.exp(
            (logps - old_logps) / completion_ids.shape[-1]
        )  # Divide by sequence length
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        loss_1 = coef_1 * advantages
        loss_2 = coef_2 * advantages
        loss = -torch.min(loss_1, loss_2)
        if self.beta != 0.0:
            loss = loss + self.beta * kl

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(kl).mean().item()
            )

        is_clipped = (loss_1 < loss_2).float()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(is_clipped).mean().item()
        )

        return loss.mean()

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=None,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        # Auto-detect mask_id: supports LLaDA2-mini (156895), Dream (151666), LLaDA (126336)
        if mask_id is None:
            mask_id = get_mask_id(tokenizer=self.processing_class, model=model)

        if is_llada2_moe(model):
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

        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full(
                (bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
            ).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(
                    block_mask_index, steps_per_block
                )

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(
                        torch.cuda.amp, "autocast"
                    ):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_).logits
                                # Apply Dream logits shift if necessary
                                logits = apply_dream_logits_shift(logits, model)
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (
                                    logits - un_logits
                                )
                            else:
                                logits = model(x).logits
                                # Apply Dream logits shift if necessary
                                logits = apply_dream_logits_shift(logits, model)

                            # Apply Gumbel noise for sampling
                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            del logits_with_noise

                            # Handle remasking strategy
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(
                                        p, dim=-1, index=torch.unsqueeze(x0, -1)
                                    ),
                                    -1,
                                )
                            elif remasking == "random":
                                x0_p = torch.rand(
                                    (x0.shape[0], x0.shape[1]), device=x0.device
                                )
                            else:
                                raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(
                                x0, dtype=torch.bool, device=x0.device
                            )
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(
                                        confidence[j], k=num_tokens
                                    )
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]
                            del x0, confidence, transfer_index

            return x

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        input = batch
        logits = model(input).logits
        # Apply Dream logits shift if necessary
        logits = apply_dream_logits_shift(logits, model)

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        with torch.no_grad():
            mode = "eval" if self.control.should_evaluate else "train"
            if mode == "train":
                if self.state.global_step % self.num_iterations == 0:
                    inputs = self._generate_and_score_completions(inputs)
                    self._buffered_inputs[
                        self._step % self.args.gradient_accumulation_steps
                    ] = inputs
                else:
                    inputs = self._buffered_inputs[
                        self._step % self.args.gradient_accumulation_steps
                    ]
                self._step += 1
            else:
                # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
                inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Main orchestrator for generation and scoring process.
        Simplified to match rev_grpo_trainer.py pattern.
        """
        device = self.accelerator.device

        # inputs is list of dicts from dataloader
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

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator
        ) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            self._r1_generate_offset = 0
            prompt_completion_ids_all = []
            entropy_rewards_all = []
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]

                # Enable dynamic_generate for both "b1_gdpo" and "r1_b1_gdpo".
                if "b1_gdpo" in str(self.args.trainer_type):
                    batch_prompt_completion_ids, batch_entropy_rewards = (
                        dynamic_generate(
                            model=unwrapped_model,
                            prompt=batch_prompt_ids,
                            steps=steps,
                            gen_length=gen_length,
                            temperature=temperature,
                            cfg_scale=cfg_scale,
                            remasking=self.args.remasking,
                            mask_id=self.args.mask_id,
                            tokenizer=self.processing_class,
                        )
                    )
                    entropy_rewards_all.append(batch_entropy_rewards)
                else:
                    batch_prompt_completion_ids = self.generate(
                        model=unwrapped_model,
                        prompt=batch_prompt_ids,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        remasking=self.args.remasking,
                        mask_id=self.args.mask_id,
                    )
                    entropy_rewards_all.append(
                        torch.zeros(batch_prompt_ids.size(0), device=device)
                    )
                    self._r1_generate_offset += batch_prompt_ids.size(0)
                prompt_completion_ids_all.append(batch_prompt_completion_ids)
                entropy_rewards_tensor = torch.cat(entropy_rewards_all, dim=0)

                del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        if self.args.random_masking:
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            mask_seeds = [42] * self.num_iterations

        # Compute old and ref logps if needed
        logits_to_keep = completion_ids.size(1)
        bs, seq_len = prompt_completion_ids.shape
        with torch.no_grad():
            if self.num_iterations > 1:
                prompt_completion_ids_expanded = (
                    prompt_completion_ids.unsqueeze(0)
                    .expand(self.num_iterations, -1, -1)
                    .reshape(self.num_iterations * bs, seq_len)
                )
                this_itr_mask_seed = mask_seeds[self._step % self.args.num_iterations]
                old_logps, _ = self.logp_estimator.get_log_likelihood(
                    self.model,
                    prompt_completion_ids_expanded,
                    logits_to_keep=logits_to_keep,
                    seed=this_itr_mask_seed,
                )
                old_logps = old_logps.reshape(self.num_iterations, bs)
            else:
                old_logps = None

            if self.beta == 0.0:
                ref_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_logps, _ = self.logp_estimator.get_log_likelihood(
                        self.model,
                        prompt_completion_ids_expanded,
                        logits_to_keep=logits_to_keep,
                        seed=this_itr_mask_seed,
                    )
                    ref_logps = ref_logps.reshape(self.num_iterations, bs)

        # Decode completions
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

        # Compute rewards
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

        rewards_per_func = gather(rewards_per_func)
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        entropy_rewards_tensor = gather(entropy_rewards_tensor)
        if "b1_gdpo" in str(self.args.trainer_type):
            rewards += entropy_rewards_tensor.to(rewards.device)

        advantages, std_grouped_rewards, zero_std_ratio = (
            grpo_group_normalized_advantages(rewards, self.num_generations)
        )

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        if "b1" in self.args.trainer_type:
            mean_entropy_reward = entropy_rewards_tensor.mean().item()
            self._metrics[mode]["rewards/block_entropy_reward_func"].append(
                mean_entropy_reward
            )

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

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_logps": old_logps,
            "ref_logps": ref_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,
        }
