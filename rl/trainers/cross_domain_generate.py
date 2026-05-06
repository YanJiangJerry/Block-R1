"""
R1: Cross-Domain Dynamic Block Size RL for Diffusion Language Models (dLLMs)

Core Insight:
  B1 relies on \\block token markers to align reasoning steps (single-domain only).
  R1 learns the optimal block size per domain through a multi-armed bandit within
  GRPO groups, and routes unseen test prompts via embedding similarity.

Key Design Choices:
  1. Per-GRPO-group block size diversity: within each prompt's N completions,
     each uses a DIFFERENT block size. GRPO's advantage normalization naturally
     discovers which block sizes work best for each task type.
  2. Efficiency reward: larger block = more parallel = more efficient.
       R_R1(x,y,d_k,b) = r_k(x,y) · (1 + γ·b/b_max)
  3. Embedding-based zero-shot routing at test time: record per-domain prompt
     embedding centroids during training; at inference, find nearest centroid
     via cosine similarity to select the best block size for unseen prompts.

Mathematical Framework:
  - K domains {d_1,...,d_K} with datasets {T_k} and rewards {r_k}
  - Block size candidates B = {b_1,...,b_M}
  - Per-domain Q-value bandit:
      Q_k(b) ← (1-α)·Q_k(b) + α · [advantage_i + γ·b/b_max]
  - Inference routing via prompt embeddings:
      d* = argmax_k cos(embed(x), centroid_k)
      b* = argmax_b Q_{d*}(b)
"""

import logging
import math
import random
import json
import os
from collections import defaultdict
from typing import Optional, Dict, List, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import gather
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation

from rl.trainers.train_utils import get_mask_id
from rl.trainers.r1_adaptive_proto import AdaptiveProtoBlockController

RewardFunc = Union[str, callable]

_log_route = logging.getLogger(__name__)
_route_reward_err_count: List[int] = [0]


def _q_values_from_json(state_table: Dict) -> Dict[str, Dict[int, float]]:
    """JSON makes nested dict keys strings; restore int block sizes for Q tables."""
    out: Dict[str, Dict[int, float]] = {}
    for dom, inner in state_table.items():
        if not isinstance(inner, dict):
            out[dom] = inner  # type: ignore[assignment]
            continue
        out[dom] = {}
        for bk, bv in inner.items():
            ik = int(bk) if not isinstance(bk, int) else bk
            out[dom][ik] = float(bv)
    return out


def _counts_from_json(state_table: Dict) -> Dict[str, Dict[int, int]]:
    out: Dict[str, Dict[int, int]] = {}
    for dom, inner in state_table.items():
        if not isinstance(inner, dict):
            out[dom] = inner  # type: ignore[assignment]
            continue
        out[dom] = {}
        for bk, bv in inner.items():
            ik = int(bk) if not isinstance(bk, int) else bk
            out[dom][ik] = int(bv)
    return out


# =========================================================================
# This is the Block Size Controller (Per-Domain Multi-Armed Bandit + Embedding Router)
# =========================================================================
class BlockSizeController:
    """
    Domain-adaptive block size selection via multi-armed bandit,
    with embedding-based zero-shot routing for unseen test domains.

    Training: update Q_k(b) with EMA from per-sample advantages.
    Inference: embed the prompt → find nearest domain centroid → Q-table lookup.
    """

    def __init__(
        self,
        domains: List[str],
        block_size_candidates: List[int] = None,
        gen_length: int = 256,
        lr: float = 0.1,
        exploration_rate: float = 0.3,
        temperature: float = 1.0,
        default_block_size: int = 32,
    ):
        if block_size_candidates is None:
            block_size_candidates = [16, 32, 64, 128]

        self.domains = list(domains)
        self.lr = lr
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        self.default_block_size = default_block_size

        self.candidates = sorted(
            [b for b in block_size_candidates if gen_length % b == 0]
        )
        if not self.candidates:
            self.candidates = [default_block_size]

        self.q_values: Dict[str, Dict[int, float]] = {
            d: {b: 0.0 for b in self.candidates} for d in self.domains
        }
        self.counts: Dict[str, Dict[int, int]] = {
            d: {b: 0 for b in self.candidates} for d in self.domains
        }

        # Embedding centroids for zero-shot routing at inference time
        # Per-domain running sum of prompt embeddings and total sample count (Scheme B:
        # global mean = sum / N_d, each prompt weighted equally — not mean-of-batch-means).
        # Multi-GPU: _domain_embed_* hold merged totals after sync; _domain_embed_local_*
        # hold only this rank's delta since last sync (all_reduce SUM on locals, then merge).
        self._domain_embed_sum: Dict[str, torch.Tensor] = {}
        self._domain_embed_sample_count: Dict[str, int] = {}
        self._domain_embed_local_sum: Dict[str, torch.Tensor] = {}
        self._domain_embed_local_count: Dict[str, int] = {}

        # Rotating offset for block size assignment across generate() calls.
        # Ensures all candidates are covered even when generation_batch_size < len(candidates).
        self._assign_offset: int = 0

    # ----- block size assignment for GRPO groups -----
    def reset_assignment(self):
        """Reset offset at the start of each _generate_and_score_completions round."""
        self._assign_offset = 0

    def assign_group_block_sizes(self, group_size: int) -> List[int]:
        """
        Cycle through candidates with a persistent offset.

        Within one training step (between reset_assignment calls), consecutive
        generate() calls continue cycling where the last one left off.
        This guarantees every candidate is explored regardless of batch size.

        Ideal: num_generations == len(candidates) so each GRPO group
        tests every candidate exactly once.  Works with any ratio though.
        """
        result = []
        for _ in range(group_size):
            result.append(self.candidates[self._assign_offset % len(self.candidates)])
            self._assign_offset += 1
        return result

    def get_best_block_size(self, domain: str) -> int:
        if domain not in self.q_values:
            return int(self.default_block_size)
        q = self.q_values[domain]
        # max() returns a dict key; JSON-loaded state may have left str keys—int() is safe after load_state_dict normalizes.
        return int(max(q, key=q.get))

    # ----- Q-value update -----
    def update(self, domain: str, block_size: int, signal: float):
        """EMA update: Q_k(b) ← (1-α)Q_k(b) + α·signal."""
        if domain not in self.q_values or block_size not in self.q_values[domain]:
            return
        old_q = self.q_values[domain][block_size]
        self.q_values[domain][block_size] = (1 - self.lr) * old_q + self.lr * signal
        self.counts[domain][block_size] += 1

    # ----- embedding centroid management -----
    def update_domain_embedding(
        self, domain: str, embedding_sum: torch.Tensor, num_samples: int
    ):
        """Accumulate embedding sum per domain for a global sample-weighted centroid.

        centroid_d = (sum of all prompt embeddings in d) / (total #prompts seen in d).

        Args:
            domain: Domain label.
            embedding_sum: Sum of per-prompt embeddings for this domain in the current
                batch, shape (D,). Use sum, not mean, so num_samples weights correctly.
            num_samples: Number of prompts in this batch for this domain (>0).
        """
        if num_samples <= 0:
            return
        s = embedding_sum.detach().float().cpu()
        if domain not in self._domain_embed_local_sum:
            self._domain_embed_local_sum[domain] = s.clone()
            self._domain_embed_local_count[domain] = num_samples
        else:
            self._domain_embed_local_sum[domain] += s
            self._domain_embed_local_count[domain] += num_samples

    def get_domain_centroid(self, domain: str) -> Optional[torch.Tensor]:
        g_sum = self._domain_embed_sum.get(domain)
        l_sum = self._domain_embed_local_sum.get(domain)
        g_n = int(self._domain_embed_sample_count.get(domain, 0))
        l_n = int(self._domain_embed_local_count.get(domain, 0))
        if g_sum is None and l_sum is None:
            return None
        total_n = g_n + l_n
        if total_n <= 0:
            return None
        if g_sum is None:
            total_sum = l_sum
        elif l_sum is None:
            total_sum = g_sum
        else:
            total_sum = g_sum + l_sum
        return total_sum / float(total_n)

    def select_block_size_by_embedding(self, prompt_embedding: torch.Tensor) -> int:
        """Zero-shot routing: find nearest domain centroid and return its best block size."""
        domains_seen = set(self._domain_embed_sum) | set(self._domain_embed_local_sum)
        if not domains_seen:
            return self.default_block_size
        emb = prompt_embedding.detach().float().cpu()
        best_domain, best_sim = None, -float("inf")
        for domain in domains_seen:
            centroid = self.get_domain_centroid(domain)
            if centroid is None:
                continue
            sim = F.cosine_similarity(emb.unsqueeze(0), centroid.unsqueeze(0)).item()
            if math.isnan(sim):
                continue
            if sim > best_sim:
                best_sim = sim
                best_domain = domain
        if best_domain is None:
            return self.default_block_size
        return self.get_best_block_size(best_domain)

    def sync_across_processes(self, device: torch.device) -> None:
        """All-reduce R1 state so every rank matches global training statistics.

        - Embedding sums & sample counts: SUM across ranks (exact global centroid).
        - Q-values: mean across ranks (each rank applies local EMA; mean fuses estimates).
        - Arm visit counts: SUM across ranks (total pulls cluster-wide).

        Requires the same embedding dim D on all ranks (same model). If a rank has a
        stale tensor with wrong numel(), that domain row is treated as zeros for that rank.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        ws = dist.get_world_size()
        if ws <= 1:
            return

        # Agree on embedding dimension (max across ranks; 0 if none yet).
        local_d = 0
        for d in self.domains:
            if d in self._domain_embed_sum:
                local_d = max(local_d, self._domain_embed_sum[d].numel())
            if d in self._domain_embed_local_sum:
                local_d = max(local_d, self._domain_embed_local_sum[d].numel())
        dim_t = torch.tensor([local_d], device=device, dtype=torch.long)
        dist.all_reduce(dim_t, op=dist.ReduceOp.MAX)
        D = int(dim_t.item())

        if D > 0:
            sums = []
            emb_counts: List[int] = []
            for d in self.domains:
                if d in self._domain_embed_local_sum and self._domain_embed_local_sum[d].numel() == D:
                    sums.append(
                        self._domain_embed_local_sum[d].to(
                            device=device, dtype=torch.float32
                        )
                    )
                    emb_counts.append(int(self._domain_embed_local_count.get(d, 0)))
                else:
                    sums.append(torch.zeros(D, device=device, dtype=torch.float32))
                    emb_counts.append(0)
            S = torch.stack(sums, dim=0)
            Nemb = torch.tensor(emb_counts, device=device, dtype=torch.int64)
            dist.all_reduce(S, op=dist.ReduceOp.SUM)
            dist.all_reduce(Nemb, op=dist.ReduceOp.SUM)
            for i, d in enumerate(self.domains):
                delta_n = int(Nemb[i].item())
                delta_s = S[i].detach().cpu()
                self._domain_embed_local_sum.pop(d, None)
                self._domain_embed_local_count.pop(d, None)
                if delta_n <= 0:
                    continue
                if d not in self._domain_embed_sum:
                    self._domain_embed_sum[d] = delta_s.clone()
                    self._domain_embed_sample_count[d] = delta_n
                else:
                    self._domain_embed_sum[d] = self._domain_embed_sum[d] + delta_s
                    self._domain_embed_sample_count[d] = (
                        int(self._domain_embed_sample_count.get(d, 0)) + delta_n
                    )

        n_dom, n_c = len(self.domains), len(self.candidates)
        Q = torch.zeros(n_dom, n_c, device=device, dtype=torch.float32)
        # int64 all_reduce: exact arm visit counts (float32 would lose precision when large).
        C = torch.zeros(n_dom, n_c, device=device, dtype=torch.int64)
        for i, d in enumerate(self.domains):
            for j, b in enumerate(self.candidates):
                Q[i, j] = float(self.q_values[d][b])
                C[i, j] = int(self.counts[d][b])
        dist.all_reduce(Q, op=dist.ReduceOp.SUM)
        dist.all_reduce(C, op=dist.ReduceOp.SUM)
        Q /= float(ws)
        for i, d in enumerate(self.domains):
            for j, b in enumerate(self.candidates):
                self.q_values[d][b] = float(Q[i, j].item())
                self.counts[d][b] = int(C[i, j].item())

    def get_stats(self) -> Dict[str, float]:
        stats = {}
        for d in self.domains:
            best_b = self.get_best_block_size(d)
            stats[f"r1/best_block_size/{d}"] = float(best_b)
            for b in self.candidates:
                stats[f"r1/q/{d}/b{b}"] = self.q_values[d][b]
        return stats

    def state_dict(self) -> Dict:
        centroids = {}
        embed_sum_out = {}
        sample_count_out = {}
        domains_seen = set(self._domain_embed_sum) | set(self._domain_embed_local_sum)
        for d in domains_seen:
            c = self.get_domain_centroid(d)
            if c is not None:
                centroids[d] = c.tolist()
            g = self._domain_embed_sum.get(d)
            l = self._domain_embed_local_sum.get(d)
            gn = int(self._domain_embed_sample_count.get(d, 0))
            ln = int(self._domain_embed_local_count.get(d, 0))
            if g is None and l is None:
                continue
            if g is None:
                merged_sum = l
            elif l is None:
                merged_sum = g
            else:
                merged_sum = g + l
            embed_sum_out[d] = merged_sum.tolist()
            sample_count_out[d] = gn + ln
        return {
            "q_values": self.q_values,
            "counts": self.counts,
            "domain_centroids": centroids,
            "domain_embed_sum": embed_sum_out,
            "domain_embed_sample_count": sample_count_out,
            "candidates": self.candidates,
        }

    def load_state_dict(self, state: Dict):
        self.q_values = _q_values_from_json(state["q_values"])
        self.counts = _counts_from_json(state["counts"])
        if "candidates" in state:
            self.candidates = sorted(int(x) for x in state["candidates"])
        # Prefer exact resume: total embedding sum + per-domain sample counts.
        if "domain_embed_sum" in state and "domain_embed_sample_count" in state:
            self._domain_embed_sum.clear()
            self._domain_embed_sample_count.clear()
            self._domain_embed_local_sum.clear()
            self._domain_embed_local_count.clear()
            sum_map = state["domain_embed_sum"]
            cnt_map = state["domain_embed_sample_count"]
            for d, vec in sum_map.items():
                self._domain_embed_sum[d] = torch.tensor(vec, dtype=torch.float32)
                self._domain_embed_sample_count[d] = int(cnt_map.get(d, 0))
        elif "domain_centroids" in state:
            # Backward compat: only mean vectors saved (old checkpoints) — treat as
            # one pseudo-sample so routing matches previous eval behavior.
            self._domain_embed_sum.clear()
            self._domain_embed_sample_count.clear()
            self._domain_embed_local_sum.clear()
            self._domain_embed_local_count.clear()
            for d, vec in state["domain_centroids"].items():
                t = torch.tensor(vec, dtype=torch.float32)
                self._domain_embed_sum[d] = t.clone()
                self._domain_embed_sample_count[d] = 1


# =========================================================================
# This is the Prompt Embedding Utility
# =========================================================================
@torch.no_grad()
def compute_prompt_embedding(model, token_ids: torch.Tensor, pad_token_id: int):
    """Mean-pool input embeddings (no forward pass needed, very cheap)."""
    embed_layer = None
    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    if embed_layer is None:
        inner = getattr(model, "base_model", model)
        inner = getattr(inner, "model", inner)
        if hasattr(inner, "get_input_embeddings"):
            embed_layer = inner.get_input_embeddings()
        elif hasattr(inner, "embed_tokens"):
            embed_layer = inner.embed_tokens
    if embed_layer is None:
        return None
    embeddings = embed_layer(token_ids)  # (B, L, D)
    mask = (token_ids != pad_token_id).unsqueeze(-1).float()
    return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


# =========================================================================
# This is the Multi-Domain Reward Routing
# =========================================================================
def create_multi_domain_reward_func(domain_reward_map: Dict[str, List[callable]]):
    """Unified reward that routes to domain-specific functions via the `domain` kwarg."""

    def _route(prompts, completions, domain=None, step=None, run_name=None, **kwargs):
        if domain is None:
            return [0.0] * len(prompts)
        rewards = [0.0] * len(prompts)
        for i in range(len(prompts)):
            d = domain[i] if i < len(domain) else "general"
            rk = "guru" if isinstance(d, str) and d.startswith("guru_") else d
            for func in domain_reward_map.get(rk, []):
                try:
                    kw = {}
                    for k, v in kwargs.items():
                        if isinstance(v, (list, tuple)) and len(v) > i and v[i] is not None:
                            kw[k] = [v[i]]
                        elif not isinstance(v, (list, tuple)) and v is not None:
                            kw[k] = v
                    r = func(
                        prompts=[prompts[i]], completions=[completions[i]],
                        step=step, run_name=run_name, **kw,
                    )
                    if r and r[0] is not None:
                        rewards[i] += r[0]
                except ImportError:
                    # Guru / Reasoning360 reward must not fail silently (would look like "training runs" with 0 reward).
                    raise
                except NotImplementedError:
                    # Unknown data_source vs trimmed rl.eval.guru — do not swallow.
                    raise
                except Exception as e:
                    if _route_reward_err_count[0] < 8:
                        _log_route.warning(
                            "R1 _route: domain reward raised (rk=%r, domain=%r, i=%s): %s",
                            rk,
                            d,
                            i,
                            e,
                            exc_info=_route_reward_err_count[0] < 3,
                        )
                        _route_reward_err_count[0] += 1
        return rewards

    return _route


# =========================================================================
# This is the R1 Trainer
# =========================================================================
class _R1Mixin:
    """
    Mixin for R1 cross-domain dynamic block-size RL.

    Core mechanics injected into any base trainer:
      • generate(): per-sample block size variation within each GRPO group.
      • _prepare_inputs(): detect domain, compute + store embedding centroids.
      • _generate_and_score_completions(): update Q-table after rewards are known.
    """

    block_size_controller: Optional[Union[BlockSizeController, AdaptiveProtoBlockController]]

    def _init_r1(self, block_size_controller=None):
        self.block_size_controller = block_size_controller
        self._r1_sample_block_sizes: List[int] = []
        self._current_r1_domain: Optional[str] = None
        self._current_r1_domains: Optional[List[str]] = None
        self._r1_proto_keys: Optional[List[str]] = None

    # ---- override generate: per-sample block sizes ----

    def generate(self, model, prompt, block_length=32, **kwargs):
        """Each sample in the batch gets a different block size (GRPO diversity)."""
        from rl.trainers.block_r1_trainer import snap_block_size_for_gen_length

        bs = prompt.shape[0]
        off = int(getattr(self, "_r1_generate_offset", 0))
        gen_length = int(
            kwargs.get("gen_length", getattr(self.args, "max_completion_length", 256))
        )
        fallback = int(block_length)
        use_br1 = getattr(self.args, "use_block_r1_dataset", False)
        ds_blocks = getattr(self, "_r1_dataset_block_sizes", None)
        if (
            use_br1
            and ds_blocks is not None
            and off + bs <= len(ds_blocks)
        ):
            block_sizes = [
                snap_block_size_for_gen_length(
                    int(ds_blocks[off + j]), gen_length, fallback=fallback
                )
                for j in range(bs)
            ]
            self._r1_sample_block_sizes.extend(block_sizes)
            groups = defaultdict(list)
            for j, b in enumerate(block_sizes):
                groups[int(b)].append(j)

            slices = [None] * bs
            for block_size, indices in groups.items():
                idx_tensor = torch.tensor(
                    indices, dtype=torch.long, device=prompt.device,
                )
                sub_prompt = prompt[idx_tensor]
                sub_out = super().generate(
                    model, sub_prompt, block_length=block_size, **kwargs,
                )
                for li, gi in enumerate(indices):
                    slices[gi] = sub_out[li: li + 1]

            return torch.cat(slices, dim=0)

        if self.block_size_controller is None:
            return super().generate(model, prompt, block_length=block_length, **kwargs)

        block_sizes = self.block_size_controller.assign_group_block_sizes(bs)
        self._r1_sample_block_sizes.extend(block_sizes)

        groups = defaultdict(list)
        for j, b in enumerate(block_sizes):
            groups[b].append(j)

        slices = [None] * bs
        for block_size, indices in groups.items():
            idx_tensor = torch.tensor(
                indices, dtype=torch.long, device=prompt.device,
            )
            sub_prompt = prompt[idx_tensor]
            sub_out = super().generate(
                model, sub_prompt, block_length=block_size, **kwargs,
            )
            for li, gi in enumerate(indices):
                slices[gi] = sub_out[li: li + 1]

        return torch.cat(slices, dim=0)

    # ---- _prepare_inputs: detect domain + compute embeddings ----
    def _prepare_inputs(self, inputs):
        # Only reset routing + update centroids on steps that actually run generation
        # and scoring. When num_iterations>1, intermediate steps reuse buffered
        # completions from the generation step — the dataloader batch here does NOT
        # match those completions; updating centroids would poison domain means.
        if (
            getattr(self.args, "use_block_r1_dataset", False)
            and isinstance(inputs, (list, tuple))
            and len(inputs) > 0
            and isinstance(inputs[0], dict)
        ):
            ni = int(getattr(self, "num_iterations", 1) or 1)
            in_eval_loop = getattr(self.control, "should_evaluate", False)
            will_generate = (not in_eval_loop) and (
                self.state.global_step % ni == 0
            )
            if will_generate:
                fb = int(getattr(self.args, "block_length", 32) or 32)
                self._r1_sample_block_sizes = []
                self._r1_generate_offset = 0
                self._r1_dataset_block_sizes = []
                for x in inputs:
                    try:
                        b = int(x.get("br1_best_block_size", fb))
                    except (TypeError, ValueError):
                        b = fb
                    self._r1_dataset_block_sizes.append(b)
                domains = [
                    str(x.get("domain") or x.get("br1_domain") or "general")
                    for x in inputs
                ]
                self._current_r1_domains = domains
                self._current_r1_domain = domains[0] if len(domains) else "general"
                self._r1_proto_keys = None

            return super()._prepare_inputs(inputs)

        if (
            self.block_size_controller is not None
            and isinstance(inputs, (list, tuple))
            and len(inputs) > 0
            and isinstance(inputs[0], dict)
        ):
            ni = int(getattr(self, "num_iterations", 1) or 1)
            in_eval_loop = getattr(self.control, "should_evaluate", False)
            will_generate = (not in_eval_loop) and (
                self.state.global_step % ni == 0
            )
            if will_generate:
                self._r1_sample_block_sizes = []
                self.block_size_controller.reset_assignment()
                # NOTE: batches are mixed-domain after concatenation+shuffle, so we must
                # track domain per-sample (not just inputs[0]) to update Q-values correctly.
                domains = [x.get("domain", "general") for x in inputs]
                self._current_r1_domains = domains
                # Keep a best-effort single-domain field for backward-compat/logging.
                self._current_r1_domain = domains[0] if len(domains) else "general"

                # Compute prompt embeddings: DSCB prototypes or per-domain centroids
                try:
                    prompts_text = [
                        maybe_apply_chat_template(x, self.processing_class)["prompt"]
                        for x in inputs
                    ]
                    tok = self.processing_class(
                        text=prompts_text, return_tensors="pt",
                        padding=True, padding_side="left", add_special_tokens=False,
                    )
                    token_ids = tok["input_ids"].to(self.accelerator.device)
                    emb = compute_prompt_embedding(
                        self.model, token_ids, self.processing_class.pad_token_id or 0,
                    )
                    ctrl = self.block_size_controller
                    if emb is not None and hasattr(ctrl, "observe_prompt_embeddings"):
                        self._r1_proto_keys = ctrl.observe_prompt_embeddings(emb)
                    elif emb is not None:
                        self._r1_proto_keys = None
                        by_domain: Dict[str, List[int]] = defaultdict(list)
                        for i, d in enumerate(domains):
                            by_domain[d].append(i)
                        known = set(ctrl.domains)
                        for d, idxs in by_domain.items():
                            if d not in known:
                                continue
                            idx_t = torch.tensor(
                                idxs, device=emb.device, dtype=torch.long,
                            )
                            n_t = len(idxs)
                            batch_sum = emb.index_select(0, idx_t).sum(dim=0)
                            ctrl.update_domain_embedding(d, batch_sum, n_t)
                    else:
                        self._r1_proto_keys = None
                except Exception:
                    self._r1_proto_keys = None

        return super()._prepare_inputs(inputs)

    # ---- _generate_and_score_completions: update controller ----
    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)

        ctrl = self.block_size_controller
        domains = self._current_r1_domains
        bs_list = self._r1_sample_block_sizes

        proto_keys = getattr(self, "_r1_proto_keys", None)
        is_adaptive = ctrl is not None and hasattr(ctrl, "observe_prompt_embeddings")

        if ctrl is not None and bs_list and domains:
            advantages = result.get("advantages")
            if advantages is not None:
                eff_w = getattr(self.args, "r1_efficiency_weight", 0.1)
                max_bs = max(ctrl.candidates)

                n = min(len(bs_list), len(advantages), len(domains))
                use_proto = (
                    is_adaptive
                    and proto_keys is not None
                    and len(proto_keys) >= n
                )
                if not (is_adaptive and not use_proto):
                    for i in range(n):
                        b = bs_list[i]
                        adv = advantages[i].item() if torch.is_tensor(advantages[i]) else float(advantages[i])
                        signal = adv + eff_w * (b / max_bs)
                        qkey = proto_keys[i] if use_proto else domains[i]
                        ctrl.update(qkey, b, signal)

                # Log metrics
                try:
                    mode = "eval" if getattr(self.control, "should_evaluate", False) else "train"
                    for k, v in ctrl.get_stats().items():
                        self._metrics[mode][k].append(v)
                except Exception:
                    pass

        # Merge embedding sums / Q / counts across ranks (centroids exact; Q fused by mean).
        if ctrl is not None:
            try:
                ctrl.sync_across_processes(self.accelerator.device)
            except Exception as e:
                if getattr(self.accelerator, "is_main_process", False):
                    print(f"[R1] sync_across_processes failed: {e!r}")

        return result


# =========================================================================
# R1 Trainer Variants
# =========================================================================
def _get_wd1_trainer():
    from rl.trainers.wd1_grpo_trainer import RevDiffuGRPOTrainer
    return RevDiffuGRPOTrainer


def _get_d1_trainer():
    from rl.trainers.diffu_grpo_trainer import DiffuGRPOTrainer
    return DiffuGRPOTrainer


def _get_gdpo_trainer():
    from rl.trainers.gdpo_trainer import GDPOTrainer
    return GDPOTrainer


def _get_mdpo_trainer():
    from rl.trainers.mdpo_trainer import MDPOTrainer
    return MDPOTrainer


def _get_stable_drl_trainer():
    from rl.trainers.stable_drl_trainer import StableDRLTrainer
    return StableDRLTrainer


def _get_espo_trainer():
    from rl.trainers.espo_trainer import ESPOTrainer
    return ESPOTrainer


class R1WD1Trainer(_R1Mixin, _get_wd1_trainer()):
    """R1 + WD1 (NSR+PSR)."""
    def __init__(self, *args, block_size_controller=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_r1(block_size_controller)


class R1D1Trainer(_R1Mixin, _get_d1_trainer()):
    """R1 + D1 (clipped GRPO)."""
    def __init__(self, *args, block_size_controller=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_r1(block_size_controller)


class R1GDPOTrainer(_R1Mixin, _get_gdpo_trainer()):
    """R1 + GDPO."""
    def __init__(self, *args, block_size_controller=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_r1(block_size_controller)


class R1MDPOTrainer(_R1Mixin, _get_mdpo_trainer()):
    """R1 + MDPO."""
    def __init__(self, *args, block_size_controller=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_r1(block_size_controller)


class R1StableDRLTrainer(_R1Mixin, _get_stable_drl_trainer()):
    """R1 + StableDRL (SPG + SNIS)."""

    def __init__(self, *args, block_size_controller=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_r1(block_size_controller)


class R1ESPOTrainer(_R1Mixin, _get_espo_trainer()):
    """R1 + ESPO (ELBO-based sequence-level policy optimization)."""

    def __init__(self, *args, block_size_controller=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_r1(block_size_controller)


R1_TRAINER_MAP = {
    "r1_wd1": R1WD1Trainer,
    # b1 variants: same R1 trainer class, but run_multi_train will set base_algo to b1_*,
    # enabling \\block prompts and b1-specific trainer behavior in the underlying algorithm.
    "r1_b1_wll": R1WD1Trainer,
    "r1_d1": R1D1Trainer,
    "r1_b1_d1": R1D1Trainer,
    "r1_gdpo": R1GDPOTrainer,
    "r1_b1_gdpo": R1GDPOTrainer,
    "r1_mdpo": R1MDPOTrainer,
    "r1_b1_mdpo": R1MDPOTrainer,
    "r1_stable_drl": R1StableDRLTrainer,
    "r1_b1_stable_drl": R1StableDRLTrainer,
    "r1_espo": R1ESPOTrainer,
    "r1_b1_espo": R1ESPOTrainer,
}


def get_r1_trainer_class(trainer_type: str):
    if trainer_type in R1_TRAINER_MAP:
        return R1_TRAINER_MAP[trainer_type]
    raise ValueError(f"Unknown R1 trainer type: {trainer_type}. Available: {list(R1_TRAINER_MAP.keys())}")
