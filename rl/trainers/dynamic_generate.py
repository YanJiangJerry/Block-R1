"""
This code follows b1 official implementation.
"""

import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from accelerate.utils import broadcast

from rl.trainers.train_utils import (
    get_mask_id,
    is_dream_model,
    apply_dream_logits_shift,
)


def add_gumbel_noise(logits, temperature, dtype):
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


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    In each block, the number of token to be unmask at each step
    = total_masked_tokens / steps
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
def dynamic_generate(
    model,
    prompt,
    tokenizer,
    steps=128,
    gen_length=256,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=None,
    dynamic=1,
):
    """
    Final Optimized Dynamic Generation (While-Loop Version).
    Includes:
    1. Dynamic Block Truncation (Detection separated from Execution).
    2. Normalized Block Entropy Reward Calculation.
    """
    # Auto-detect mask_id using utility function
    if mask_id is None:
        mask_id = get_mask_id(tokenizer=tokenizer, model=model)

    block_token_id = None
    target_str = "block"
    try:
        ids = tokenizer.encode(target_str, add_special_tokens=False)
        # print(f"Tag: '{target_str}' -> IDs: {ids} (Count: {len(ids)})")
        if len(ids) == 1:
            block_token_id = ids[0]
    except Exception as e:
        print(f"Error encoding target string '{target_str}': {e}")
        pass

    with torch.cuda.amp.autocast(enabled=True):
        bs = prompt.shape[0]
        dtype = model.dtype
        device = prompt.device

        total_seq_len = prompt.shape[1] + gen_length

        # Initialize full sequence with mask tokens
        x = torch.full(
            (bs, total_seq_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        # For LLaDA2 mini models: pre-fill answer structure to guarantee
        # the model produces <answer> tags before hitting the length limit.
        from rl.trainers.train_utils import is_llada2_mini, prefill_answer_structure

        if is_llada2_mini(model) and tokenizer is not None:
            x = prefill_answer_structure(
                x, tokenizer, prompt.shape[1], gen_length, mask_id
            )

        initial_length = gen_length // 2
        num_standard_blocks = max(1, gen_length // initial_length)
        steps_per_block = max(1, steps // num_standard_blocks)

        sample_starts = [prompt.shape[1]] * bs
        current_round_block_sizes = [initial_length] * bs
        all_executed_block_sizes = [[] for _ in range(bs)]

        # [Entropy Storage] List to store average entropy for each block per sample
        block_entropies = [[] for _ in range(bs)]

        # [Entropy Tensor] Store the entropy of tokens at the current diffusion step
        # We use a full-sized tensor to hold the state
        last_step_entropy = torch.zeros(
            (bs, total_seq_len), dtype=torch.float32, device=device
        )

        # Loop Control
        loop_counter = 0
        MAX_LOOPS = gen_length  # Safety guard
        start_time = time.time()

        while True:
            # Safety break
            if loop_counter >= MAX_LOOPS:
                break
            loop_counter += 1

            # Construct active mask for the current block round
            active_mask = torch.zeros_like(x, dtype=torch.bool)
            active_indices = []
            samples_pending = False

            # For each sample in the batch, set active window
            for b in range(bs):
                start = sample_starts[b]
                # If this sample is full or EOS, early stop
                if start >= total_seq_len:
                    continue

                samples_pending = True
                active_indices.append(b)

                # Determine end for this round based on standard block length
                end = min(start + initial_length, total_seq_len)
                active_mask[b, start:end] = True
                current_round_block_sizes[b] = end - start

            # If aLL samples are finished or early stopped,
            # exit the global loop for this batch of samples
            if not samples_pending:
                break

            # Determine transfer tokens (standard logic)
            target_mask_index = (x == mask_id) & active_mask
            num_transfer_tokens = get_num_transfer_tokens(
                target_mask_index, steps_per_block
            )

            # Inner diffusion steps per block
            for i in range(steps_per_block):
                # If no tokens to transfer for all active samples, break early
                if num_transfer_tokens[active_indices, i].sum() == 0:
                    break

                mask_index = x == mask_id

                # Model Forward Pass
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    # Apply Dream logits shift if necessary
                    logits = apply_dream_logits_shift(logits, model)
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
                    # Apply Dream logits shift if necessary
                    logits = apply_dream_logits_shift(logits, model)

                # Avoid block tag generation in the early tokens of a block
                # if dynamic > 0 and block_token_id is not None:
                #     for b in active_indices:
                #         start = sample_starts[b]
                #         end_forbidden = min(start + dynamic, total_seq_len)
                #         logits[b, start:end_forbidden, block_token_id] = -float("inf")

                # Calculate entropy for the current step to capture model uncertainty
                # P(x) = Softmax(logits)
                probs = F.softmax(logits.float(), dim=-1)
                # Entropy = -sum(p * log(p))
                current_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

                # Update entropy record only for currently active tokens
                if len(active_indices) > 0:
                    last_step_entropy[active_mask] = current_entropy[active_mask]

                # Sampling by Gumbel-Max Trick based on temperature
                logits_with_noise = add_gumbel_noise(logits, temperature, dtype=dtype)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Remasking Strategy
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=device)
                else:
                    raise NotImplementedError(remasking)

                # Constrain to active window
                x0_p[~active_mask] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, torch.tensor(-np.inf, device=device)
                )

                # Select tokens to unmask
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        valid_mask = confidence[j, select_indices] > -float("inf")
                        final_indices = select_indices[valid_mask]
                        if final_indices.numel() > 0:
                            x[j, final_indices] = x0[j, final_indices]

                # ==========================================
                # Dynamic Batch Decode Refactored
                # ==========================================
                if dynamic > 0 and (i + 1) % dynamic == 0:
                    truncation_plan = {}  # Key: batch_idx, Value: new_len

                    if block_token_id is not None:
                        # --- Strategy A: Token ID ---
                        for b in active_indices:
                            start = sample_starts[b]
                            # if (total_seq_len - start) < 5:
                            #     continue  # Tiny block protection

                            curr_len = current_round_block_sizes[b]
                            window_tokens = x[b, start : start + curr_len]
                            matches = (window_tokens == block_token_id).nonzero(
                                as_tuple=True
                            )[0]

                            # Second occurrence after start of block
                            # if len(matches) >= 2:
                            #     second_occ_idx = matches[1].item()
                            #     truncation_plan[b] = (
                            #         second_occ_idx + 1
                            #     )  # Include the block token

                            # Detect the end of the block marker
                            valid_matches = matches[matches >= dynamic]
                            if len(valid_matches) > 0:
                                first_valid_idx = valid_matches[0].item()
                                truncation_plan[b] = first_valid_idx + 1

                    # else:
                    #     # --- Strategy B: String Match ---
                    #     seqs_to_decode = []
                    #     valid_batch_indices = []
                    #     for b in active_indices:
                    #         start = sample_starts[b]
                    #         # if (total_seq_len - start) < 5:
                    #         #     continue
                    #         curr_len = current_round_block_sizes[b]
                    #         seqs_to_decode.append(x[b, start : start + curr_len])
                    #         valid_batch_indices.append(b)

                    #     if seqs_to_decode:
                    #         decoded_texts = tokenizer.batch_decode(seqs_to_decode)
                    #         target_str_detect = "block"
                    #         for idx, text in enumerate(decoded_texts):
                    #             b = valid_batch_indices[idx]
                    #             if text.count(target_str_detect) >= 2:
                    #                 first_occ = text.find(target_str_detect)
                    #                 second_occ = text.find(
                    #                     target_str_detect, first_occ + 1
                    #                 )
                    #                 # Truncate string and re-encode to find length
                    #                 text_up_to_split = text[:second_occ + len(target_str_detect)]
                    #                 tokens_up_to_split = tokenizer.encode(
                    #                     text_up_to_split, add_special_tokens=False
                    #                 )
                    #                 truncation_plan[b] = len(tokens_up_to_split)

                    # Dynamic truncation for current block
                    for b, new_len in truncation_plan.items():
                        if new_len == 0:
                            new_len = 1
                        original_len = current_round_block_sizes[b]
                        start = sample_starts[b]

                        if new_len < original_len:
                            new_end = start + new_len
                            old_end = start + original_len
                            # Apply mask to truncated region
                            x[b, new_end:old_end] = mask_id
                            current_round_block_sizes[b] = new_len
                            active_mask[b, new_end:old_end] = False

                            # Recalculate num_transfer_tokens for remaining steps
                            remaining_steps = steps_per_block - (i + 1)
                            if remaining_steps > 0:
                                current_masked_count = (
                                    (x[b, start:new_end] == mask_id).sum().item()
                                )
                                base = current_masked_count // remaining_steps
                                rem = current_masked_count % remaining_steps
                                num_transfer_tokens[b, i + 1 : i + 1 + rem] = base + 1
                                num_transfer_tokens[b, i + 1 + rem :] = base

            # Detect block end and update sample_starts
            if dynamic > 0 and block_token_id is not None:
                truncation_plan = {}

                for b in active_indices:
                    start = sample_starts[b]
                    curr_len = current_round_block_sizes[b]
                    window_tokens = x[b, start : start + curr_len]
                    matches = (window_tokens == block_token_id).nonzero(as_tuple=True)[
                        0
                    ]

                    valid_matches = matches[matches >= dynamic]
                    if len(valid_matches) > 0:
                        first_valid_idx = valid_matches[0].item()
                        truncation_plan[b] = first_valid_idx + 1

                for b, new_len in truncation_plan.items():
                    if new_len == 0:
                        new_len = 1
                    original_len = current_round_block_sizes[b]
                    start = sample_starts[b]

                    if new_len < original_len:
                        new_end = start + new_len
                        old_end = start + original_len

                        x[b, new_end:old_end] = mask_id
                        current_round_block_sizes[b] = new_len

            # End of Inner Diffusion Loop (Block Finished)
            for b in range(bs):
                start = sample_starts[b]
                # Check if this sample was active and finished a block
                if start < total_seq_len:
                    actual_size = current_round_block_sizes[b]
                    all_executed_block_sizes[b].append(actual_size)

                    # [Entropy Stats] Calculate average entropy for this finished block
                    block_end = start + actual_size
                    # We take the mean entropy of the tokens that effectively constitute this block
                    avg_block_entropy = (
                        last_step_entropy[b, start:block_end].mean().item()
                    )
                    block_entropies[b].append(avg_block_entropy)

                    # Update sample_starts for next block
                    # sample_starts[b] += actual_size
                    # Check for EOS within the executed block
                    current_block = x[b, start : start + actual_size]
                    # if (
                    #     tokenizer.eos_token_id is not None
                    #     and (current_block == tokenizer.eos_token_id).all()
                    # ):
                    if (
                        tokenizer.eos_token_id is not None
                        and (current_block == tokenizer.eos_token_id).any()
                    ):
                        # Trigger the early stop
                        sample_starts[b] = total_seq_len
                    else:
                        sample_starts[b] += actual_size

        # print(f"Dynamic reasoning blocks per device: {all_executed_block_sizes}")

        # end_time = time.time()
        # elapsed = end_time - start_time
        # total_tokens = bs * gen_length
        # speed = elapsed / total_tokens if total_tokens > 0 else 0.0
        # print(
        #     f"[Speed] Total Time: {elapsed:.2f}s | Tokens: {total_tokens} | Speed: {speed:.4f} s/token"
        # )

        # ==========================================
        # Normalized Dynamic Block Entropy Reward
        # Reward = (Number of Decreasing Transitions) / (Total Transitions)
        # Range: [0.0, 1.0]
        # ==========================================
        entropy_rewards = torch.zeros(bs, device=device)
        target_num_blocks = 10
        for b in range(bs):
            entropies = block_entropies[b]
            num_blocks = len(entropies)

            # # ====================================================
            # # Sigmoid Indicator Reward
            # # Center = 5 (Target/2), so at 10 it's fully saturated
            # # ====================================================
            # quantity_score = 1 / (
            #     1 + math.exp(-(num_blocks - target_num_blocks / 2))
            # )
            # quantity_score = min(num_blocks / target_num_blocks, 1.0)
            if num_blocks >= target_num_blocks:
                quantity_score = 1.0
            else:
                # n=1 => 0.29, n=5 => 0.75 (Target=10)
                quantity_score = math.log(num_blocks + 1) / math.log(
                    target_num_blocks + 1
                )

            # Need at least 2 blocks to establish a block entropy trend
            if num_blocks >= 2:
                decrease_count = 0.0
                total_transitions = num_blocks - 1

                for k in range(1, num_blocks):
                    prev_ent = entropies[k - 1]
                    curr_ent = entropies[k]

                    # Reward if entropy decreases (confidence increases)
                    if curr_ent < prev_ent:
                        decrease_count += 1.0
                # Combine trend and quantity into final reward
                trend_score = decrease_count / total_transitions

                entropy_rewards[b] = 0.5 * trend_score + 0.5 * quantity_score
                # entropy_rewards[b] = decrease_count / total_transitions
            else:
                # 0 or 1 block implies no trend data
                entropy_rewards[b] = 0.0

            # # 2. MED Rewards (r_SCC)
            # trend_score = 0.0
            # if num_blocks >= 2:
            #     # Indices k = [1, 2, ..., K]
            #     indices = torch.arange(1, num_blocks + 1, dtype=torch.float32)
            #     entropy_vals = torch.tensor(entropies, dtype=torch.float32)

            #     # Compute Ranks
            #     # Rank of indices is just indices themselves (since they are sorted 1..K)
            #     rank_k = indices

            #     # Rank of entropies (using argsort twice)
            #     # argsort gives indices of sorted elements. argsort(argsort) gives rank (0-based)
            #     rank_h = torch.argsort(torch.argsort(entropy_vals)).float() + 1.0

            #     # Compute d_k = rank(k) - rank(H)
            #     d_k = rank_k - rank_h
            #     sum_d2 = torch.sum(d_k**2)

            #     # Calculate standard Spearman (rho)
            #     # rho = 1 - (6 * sum_d2) / (num_blocks * (num_blocks^2 - 1))
            #     # Note: This is the standard formula assuming no ties, which is safe for float entropy
            #     rho = 1.0 - (6.0 * sum_d2) / (num_blocks * (num_blocks**2 - 1))

            #     # Calculate r_SCC (Negative Spearman) as per paper definition
            #     r_scc = -rho

            #     # Normalize r_scc [-1, 1] to [0, 1] for final reward usage
            #     # If strictly monotonic descent -> r_scc = 1 -> score = 1
            #     trend_score = (r_scc + 1.0) / 2.0

            #     # Clamp for safety
            #     trend_score = torch.clamp(trend_score, 0.0, 1.0).item()

            # else:
            #     trend_score = 0.0
            # entropy_rewards[b] = 0.5 * trend_score + 0.5 * quantity_score

        # # ==========================================
        # # Print first generated text & dynamic blocks
        # # ==========================================
        # print(f"\n[DEBUG] First Sample (Batch Index 0):")
        # # Filter out remaining mask tokens (if any) to clean up output
        # valid_indices = x[0] != mask_id
        # decoded_text = tokenizer.decode(x[0][valid_indices], skip_special_tokens=False)
        # print(f"Decoded Text:\n{decoded_text}")
        # print(f"Block Sizes: {all_executed_block_sizes[0]}\n")
        # import sys

        # sys.exit(0)
        # # ==========================================

        return x, entropy_rewards
