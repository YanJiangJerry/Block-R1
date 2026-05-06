"""
This code follows d1 official implementation.
"""

import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate.utils import (
    gather_object,
)
from rl.eval.parse_and_get_acc import (
    parse_gsm_answers,
    parse_math_answers,
    parse_countdown_answers,
    parse_sudoku_answers,
    parse_code_answers,
    parse_mc_answers,
    parse_knights_knaves_answers,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback

from rl.trainers.dynamic_generate import dynamic_generate
from rl.trainers.train_utils import is_dream_model, apply_dream_logits_shift
from rl.llada2_compat import is_llada2_moe, generate_llada2


def _dedupe_and_truncate_generations(all_generations: list[dict], target_n: int) -> list[dict]:
    """
    In distributed eval, Accelerate may "even out" batches by repeating samples (even_batches=True),
    so `gather_object(all_generations)` can be longer than the true eval set size.
    To keep accuracy denominator consistent with the real dataset size, dedupe by a stable key and
    then truncate to `target_n`.
    """
    if target_n <= 0 or not all_generations:
        return all_generations

    seen = set()
    uniq: list[dict] = []
    for item in all_generations:
        q = item.get("question", "")
        gt = item.get("ground_truth", "")
        key = (q, str(gt))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
        if len(uniq) >= target_n:
            break

    if len(uniq) < min(target_n, len(all_generations)):
        return all_generations[:target_n]
    return uniq


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
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


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=None,
    disable_bar=True,
):
    """
    Optimized version of the generate function.
    """
    # Auto-detect mask_id: supports LLaDA2-mini (156895), Dream (151666), LLaDA (126336)
    if mask_id is None:
        from rl.trainers.train_utils import get_mask_id

        mask_id = get_mask_id(tokenizer=tokenizer, model=model)

    if is_llada2_moe(model):
        return generate_llada2(
            model,
            prompt,
            tokenizer=tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
        )

    def _model_logits(input_ids):
        """Forward pass with bf16 autocast to ensure flash attention compatibility."""
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return model(input_ids).logits

    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=prompt.device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=disable_bar, leave=False):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = _model_logits(x_)
                    # Apply Dream logits shift if necessary
                    logits = apply_dream_logits_shift(logits, model)
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = _model_logits(x)
                    # Apply Dream logits shift if necessary
                    logits = apply_dream_logits_shift(logits, model)

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, torch.tensor(-np.inf, device=x0.device)
                )

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
        return x


class AccuracyEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset,
        tokenizer,
        gen_length=256,
        temperature=0.0,
        steps=128,
        block_length=32,
        batch_size=4,
    ):
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.temperature = temperature
        self.steps = steps
        self.block_length = block_length

        self.eval_dataset = eval_dataset
        self.dataloader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            collate_fn=eval_dataset.collate_fn,
            # drop_last=True,
        )

    def on_evaluate(self, args, state, control, **kwargs):
        accelerator = kwargs["accelerator"]
        model = kwargs["model"]
        # Split dataset across GPUs
        eval_dataloader = accelerator.prepare(self.dataloader)

        # Generate single completion for each prompt
        all_generations = []
        if accelerator.is_main_process:
            eval_dataloader = tqdm(eval_dataloader, desc="Evaluating", leave=True)

        for batch in eval_dataloader:
            input_ids = batch["input_ids"]
            gt_answers = batch["answers"]
            questions = batch["questions"]
            prompts = batch["prompts"]
            test_lists = batch.get("test_list", [None] * len(gt_answers))

            with torch.no_grad():
                # b1 variants can appear as "b1_*" (non-R1) or "r1_b1_*" (R1/Block-R1).
                tt = str(getattr(args, "trainer_type", "") or "")
                # If this is any b1 variant, eval should use b1's dynamic_generate
                # (block markers + dynamic block routing) instead of fixed block_length.
                if re.search(r"(^|_)b1_", tt):
                    out, _ = dynamic_generate(
                        model=model,
                        prompt=input_ids,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        tokenizer=self.tokenizer,
                    )
                else:
                    out = generate(
                        model,
                        input_ids,
                        self.tokenizer,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        disable_bar=accelerator.is_main_process,
                    )

            generated_texts = self.tokenizer.batch_decode(
                out[:, -self.gen_length :], skip_special_tokens=False
            )
            example_result = [
                {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    "ground_truth": gt_answers[j],
                    "test_list": test_lists[j] if test_lists[j] is not None else "",
                }
                for j in range(len(gt_answers))
            ]
            all_generations.extend(example_result)

        # Compute accuracy using task-specific parsers when possible
        all_generations = gather_object(all_generations)
        if accelerator.is_main_process:
            # Detect dataset type from eval_dataset class name and dispatch
            dataset_name = getattr(self.eval_dataset.__class__, "__name__", "").lower()
            true_n = len(self.eval_dataset) if self.eval_dataset is not None else 0
            all_generations = _dedupe_and_truncate_generations(all_generations, true_n)
            json_data = {"generations": all_generations}

            try:
                num_correct = None
                num_total = None
                if "gsm" in dataset_name:
                    correct, processed, _, _ = parse_gsm_answers(json_data=json_data)
                    accuracy = correct / processed if processed > 0 else 0.0
                    num_correct, num_total = correct, processed
                elif "math" in dataset_name:
                    correct, processed, _, _ = parse_math_answers(json_data=json_data)
                    accuracy = correct / processed if processed > 0 else 0.0
                    num_correct, num_total = correct, processed
                elif "countdown" in dataset_name or "ctd" in dataset_name:
                    correct, processed, _, _ = parse_countdown_answers(
                        json_data=json_data
                    )
                    accuracy = correct / processed if processed > 0 else 0.0
                    num_correct, num_total = correct, processed
                elif "sudoku" in dataset_name:
                    correct_cells, total_empty, _, _ = parse_sudoku_answers(
                        json_data=json_data
                    )
                    accuracy = correct_cells / total_empty if total_empty > 0 else 0.0
                    num_correct, num_total = correct_cells, total_empty
                elif (
                    "mbpp" in dataset_name
                    or "humaneval" in dataset_name
                    or "kodcode" in dataset_name
                ):
                    correct, processed, _, _ = parse_code_answers(json_data=json_data)
                    accuracy = correct / processed if processed > 0 else 0.0
                    num_correct, num_total = correct, processed
                elif (
                    "mmlu" in dataset_name
                    or "hellaswag" in dataset_name
                    or "arc_c" in dataset_name
                    or "arc_e" in dataset_name
                    or "gpqa" in dataset_name
                ):
                    correct, processed, _, _ = parse_mc_answers(json_data=json_data)
                    accuracy = correct / processed if processed > 0 else 0.0
                    num_correct, num_total = correct, processed
                elif "knights" in dataset_name or "knaves" in dataset_name:
                    correct, processed, _, _ = parse_knights_knaves_answers(
                        json_data=json_data
                    )
                    accuracy = correct / processed if processed > 0 else 0.0
                    num_correct, num_total = correct, processed
                else:
                    # Fallback: try the original boxed/<answer> numeric parsing for unknown tasks
                    total_correct = 0
                    for example_result in all_generations:
                        parsed_answer = None
                        raw_generation = example_result["generations"]
                        ground_truth = example_result["ground_truth"]

                        # Find \boxed{} content for math500-style answers
                        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
                        if boxed_matches:
                            for boxed_content in boxed_matches:
                                boxed_content = boxed_content.strip()
                                if (
                                    boxed_content
                                    and boxed_content != "..."
                                    and not re.match(r"^\.+$", boxed_content)
                                ):
                                    try:
                                        parsed_answer = float(boxed_content)
                                        break
                                    except ValueError:
                                        numbers = re.findall(
                                            r"-?\d+\.?\d*", boxed_content
                                        )
                                        if numbers:
                                            try:
                                                parsed_answer = float(numbers[0])
                                                break
                                            except ValueError:
                                                pass

                        # If no valid \\boxed{} content found, try <answer> tags
                        if parsed_answer is None:
                            answer_match = re.search(
                                r"<answer>(.*?)</answer>", raw_generation, re.DOTALL
                            )
                            if answer_match:
                                answer_text = answer_match.group(1).strip()
                                if answer_text:
                                    try:
                                        parsed_answer = float(answer_text)
                                    except ValueError:
                                        numbers = re.findall(
                                            r"-?\d+\.?\d*", answer_text
                                        )
                                        if numbers:
                                            try:
                                                parsed_answer = float(numbers[-1])
                                            except ValueError:
                                                pass

                        is_correct = (
                            parsed_answer is not None and parsed_answer == ground_truth
                        )
                        if is_correct:
                            total_correct += 1
                    accuracy = (
                        total_correct / len(all_generations)
                        if len(all_generations) > 0
                        else 0.0
                    )
                    num_correct, num_total = total_correct, len(all_generations)

                # Log to wandb if enabled
                if args.report_to and "wandb" in args.report_to:
                    # Do not need to specific global step here
                    wandb.log({"eval/accuracy": accuracy})
                    if num_correct is not None and num_total is not None:
                        print(f"Eval Accuracy: {accuracy:.4f} ({int(num_correct)}/{int(num_total)})")
                    else:
                        print(f"Eval Accuracy: {accuracy:.4f} ({len(all_generations)}/{len(all_generations)})")
                    # This is duplicate logging, commented out
                    # Log to WandB using accelerator to ensure correct step logging
                    # metrics = {"accuracy": accuracy}
                    # accelerator.log(metrics, step=state.global_step)

                    # Construct the top 20 testing samples table
                    table_data = []
                    for item in all_generations[:20]:
                        table_data.append(
                            {
                                "prompt": item["prompt_input"],
                                "completion": item["generations"],
                                "ground_truth": str(item["ground_truth"]),
                            }
                        )

                    df_table = pd.DataFrame(table_data)
                    # Log the testing table to WandB
                    wandb.log({"testing_samples": wandb.Table(dataframe=df_table)})
                    print(
                        f"Uploaded evaluation table with {len(df_table)} samples to WandB."
                    )

            except Exception as e:
                # If parsing fails, print a helpful message but don't crash training
                print(f"Evaluation parsing error: {e}")
                if args.report_to and "wandb" in args.report_to:
                    wandb.log({"eval/accuracy": 0.0})
                accelerator.log({"accuracy": 0.0}, step=state.global_step)

        # Synchronize all processes
        accelerator.wait_for_everyone()
