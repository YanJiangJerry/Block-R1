"""
Block-R1 multi-domain training: same RL trainers as ``run_multi_train`` (``r1_*``),
but the training set is an offline ``train.jsonl`` from ``build_block_r1``, and each
sample uses its own ``br1_best_block_size`` for generation (no bandit).

Intentional differences vs ``run_multi_train`` (besides data source and per-row block size):

- No R1 bandit / DSCB prototypes: ``block_size_controller=None`` (eval callbacks use
  ``fallback_block_length`` only).
- No ``countdown``/``sudoku`` train trimming (R1 drops last 500) or ``mbpp`` slice; rows
  are exactly those in ``build_block_r1`` indexed by manifest ``orig_idx`` into the
  same loaders as R1.

Label columns: ``build_block_r1`` copies full loader rows (``_train_row_to_record``), so
``answer`` / ``target``+``numbers`` / ``puzzle``+``solution`` / ``test_list`` etc. match
``create_multi_domain_dataset``.

Usage:
  accelerate launch rl/run_block_r1.py --config rl/train.yaml \\
    --model_path ... --trainer_type br1_stable_drl --use_r1 true \\
    --block_r1_train_jsonl path/to/train.jsonl \\
    [--r1_domains gsm8k,math,...]   # optional; default = domains present in JSONL
"""

import os
import warnings
import logging

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def _noop(*_args, **_kwargs):
    return


warnings.showwarning = _noop
warnings.warn = _noop
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for _name in ["transformers", "torch", "pydantic", "torch.distributed.run", "accelerate"]:
    try:
        logging.getLogger(_name).setLevel(logging.ERROR)
    except Exception:
        pass

import json

import torch
import wandb
from peft import LoraConfig
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
)
from trl import ModelConfig, TrlParser

import rl.run_multi_train as rmt
try:
    from data_utils import set_random_seed, set_trainer_type  # type: ignore
except ModuleNotFoundError:
    from rl.data_utils import set_random_seed, set_trainer_type
from rl.trainers.block_r1_trainer import (
    infer_block_r1_domains,
    load_block_r1_train_jsonl,
    validate_block_r1_reward_schema,
)
from rl.trainers.diffu_grpo_config import DiffuGRPOConfig


def _balance_domains_min_count(train_set, *, seed: int):
    """
    Subsample each domain to the same size (min count), using the current dataset order.
    Assumes train_set has a 'domain' column and was already shuffled.
    """
    from collections import defaultdict, Counter

    domains = list(train_set["domain"])
    idx_by_dom = defaultdict(list)
    for i, d in enumerate(domains):
        idx_by_dom[str(d)].append(i)

    counts = Counter({d: len(idxs) for d, idxs in idx_by_dom.items()})
    if not counts:
        return train_set, {"balanced": False, "reason": "no_domains", "counts": {}}

    target = min(counts.values())
    if target <= 0:
        return train_set, {"balanced": False, "reason": "empty_domain", "counts": dict(counts)}

    keep = []
    for d, idxs in idx_by_dom.items():
        keep.extend(idxs[:target])

    keep.sort()
    out = train_set.select(keep).shuffle(seed=seed)
    return out, {"balanced": True, "target_per_domain": target, "counts_before": dict(counts)}


def main(grpo_config, model_config):
    set_random_seed(grpo_config.seed)

    tt_raw = (grpo_config.trainer_type or "").strip()
    if not tt_raw.startswith("br1_"):
        raise ValueError(
            "run_block_r1.py expects --trainer_type br1_* "
            "(e.g. br1_stable_drl, br1_wd1, br1_d1). "
            f"Got {tt_raw!r}."
        )
    r1_tt = "r1_" + tt_raw[4:]
    grpo_config.trainer_type = r1_tt

    if not getattr(grpo_config, "use_r1", False):
        raise ValueError("run_block_r1.py requires --use_r1 true (same flag as R1 training).")

    grpo_config.use_block_r1_dataset = True

    jsonl_path = getattr(grpo_config, "block_r1_train_jsonl", None) or os.environ.get(
        "BLOCK_R1_TRAIN_JSONL"
    )
    if not jsonl_path:
        raise ValueError(
            "Set --block_r1_train_jsonl to the train.jsonl from build_block_r1 "
            "(or BLOCK_R1_TRAIN_JSONL)."
        )
    jsonl_path = os.path.abspath(jsonl_path)
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"block_r1_train_jsonl not found: {jsonl_path}")

    train_set = load_block_r1_train_jsonl(jsonl_path, seed=grpo_config.seed)
    if len(train_set) == 0:
        raise ValueError("[Block-R1] Training dataset is empty.")

    # Same label columns as ``create_multi_domain_dataset`` (full loader rows copied in
    # ``build_block_r1`` via ``_train_row_to_record``). Optionally restrict with --r1_domains.
    domains_in_file = infer_block_r1_domains(train_set)
    if grpo_config.r1_domains:
        requested = [d.strip() for d in grpo_config.r1_domains.split(",") if d.strip()]
        requested_set = set(requested)
        dropped = sorted(set(domains_in_file) - requested_set)
        if dropped and rmt.is_main_process():
            warnings.warn(
                f"[Block-R1] Dropping domains not listed in --r1_domains: {dropped}. "
                "Only requested domains will be used for training.",
                stacklevel=1,
            )
        extra_req = sorted(requested_set - set(domains_in_file))
        if extra_req and rmt.is_main_process():
            warnings.warn(
                f"[Block-R1] --r1_domains lists domains with no rows in JSONL: {extra_req}. "
                "Training only uses domains that appear in the file.",
                stacklevel=1,
            )
        train_set = train_set.filter(lambda ex: ex["domain"] in requested_set)
        domain_names = requested
    else:
        domain_names = sorted(domains_in_file)

    if len(train_set) == 0:
        raise ValueError("[Block-R1] Training dataset is empty after domain filter.")

    if getattr(grpo_config, "block_r1_balance_domains", False):
        train_set, bal = _balance_domains_min_count(train_set, seed=grpo_config.seed)
        if rmt.is_main_process():
            print(f"[Block-R1] balance_domains=true → {bal}")
        if len(train_set) == 0:
            raise ValueError("[Block-R1] Training dataset is empty after domain balancing.")

    domains_for_reward = infer_block_r1_domains(train_set)
    unknown = [d for d in domains_for_reward if d not in rmt.DOMAIN_REWARD_MAP]
    if unknown:
        raise ValueError(
            f"[Block-R1] Unknown domain(s) for reward routing: {unknown}. "
            f"Known: {list(rmt.DOMAIN_REWARD_MAP.keys())}"
        )

    validate_block_r1_reward_schema(train_set, domains=sorted(domains_for_reward))

    active_reward_map = {d: rmt.DOMAIN_REWARD_MAP[d] for d in sorted(domains_for_reward)}
    unified_reward_func = rmt.create_multi_domain_reward_func(active_reward_map)
    reward_functions = [unified_reward_func]

    if rmt.is_main_process():
        print(f"[Block-R1] train.jsonl: {jsonl_path}")
        print(f"[Block-R1] Domains in file (before filter): {sorted(domains_in_file)}")
        print(f"[Block-R1] Domains used for training & rewards: {sorted(domains_for_reward)}")
        print(f"[Block-R1] domain / eval list (r1_domains order): {domain_names}")
        print(f"[Block-R1] Unified columns: {train_set.column_names}")
        print(f"[Block-R1] Trainer (maps to {r1_tt}): {tt_raw}")
        print(f"[Block-R1] Rows: {len(train_set)}")

    base_algo = r1_tt.replace("r1_", "")
    # Block-R1 training uses per-row br1_best_block_size for generation, so b1-style
    # "\\block" prompting is not required during training. However, keeping trainer_type
    # as r1_b1_* is still useful so eval callbacks can switch to b1 dynamic_generate.
    prompt_algo = base_algo[3:] if base_algo.startswith("b1_") else base_algo
    set_trainer_type(prompt_algo)

    ws = int(os.environ.get("WORLD_SIZE", "1"))
    pdb = int(grpo_config.per_device_train_batch_size)
    gas = int(getattr(grpo_config, "gradient_accumulation_steps", 1) or 1)
    global_bs = pdb * max(ws, 1) * max(gas, 1)
    ng = int(getattr(grpo_config, "num_generations", 1) or 1)
    if ng > 0 and global_bs % ng != 0:
        raise ValueError(
            "[Block-R1] GRPO needs (per_device_train_batch_size * WORLD_SIZE * "
            f"gradient_accumulation_steps) divisible by num_generations; "
            f"got {pdb}*{ws}*{gas}={global_bs}, num_generations={ng}."
        )

    if (
        getattr(grpo_config, "dataloader_drop_last", False)
        and len(train_set) < global_bs
        and rmt.is_main_process()
    ):
        warnings.warn(
            f"[Block-R1] len(train_set)={len(train_set)} < global train batch {global_bs} "
            f"and dataloader_drop_last=True — a rank may see no batches.",
            stacklevel=1,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        import transformers.modeling_utils as _tmu

        _orig_caching_allocator_warmup = _tmu.caching_allocator_warmup

        def _safe_caching_allocator_warmup(model_to_load, *args, **kwargs):
            try:
                if getattr(model_to_load, "_tp_plan", None) is None:
                    model_to_load._tp_plan = []
            except Exception:
                pass
            return _orig_caching_allocator_warmup(model_to_load, *args, **kwargs)

        _tmu.caching_allocator_warmup = _safe_caching_allocator_warmup
    except Exception:
        pass

    _PINNED_REVISIONS = {
        "inclusionAI/LLaDA2.1-mini": "bbb5715c881500b34234071e68dbf38c3d657c4e",
        "inclusionAI/LLaDA2.0-mini": "d23215abc5f5675daf171f6739d0386eab53f712",
    }
    _revision = _PINNED_REVISIONS.get(grpo_config.model_path)
    try:
        _cfg = AutoConfig.from_pretrained(
            grpo_config.model_path, trust_remote_code=True, revision=_revision
        )
    except ValueError as e:
        if rmt.is_main_process():
            print(
                f"[WARN] AutoConfig failed for {grpo_config.model_path}: {e}\n"
                f"       Falling back to PretrainedConfig to read config.json."
            )
        _cfg = PretrainedConfig.from_pretrained(
            grpo_config.model_path, trust_remote_code=True, revision=_revision
        )

    _auto_map = getattr(_cfg, "auto_map", {}) or {}
    _model_type = getattr(_cfg, "model_type", "")

    if _model_type == "llada2_moe":
        grpo_config.ddp_find_unused_parameters = True
        _load_kwargs = dict(
            pretrained_model_name_or_path=grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            revision=_revision,
        )
    elif _model_type == "sdar":
        _load_kwargs = dict(
            pretrained_model_name_or_path=grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            revision=_revision,
        )
    else:
        _load_kwargs = dict(
            pretrained_model_name_or_path=grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            revision=_revision,
        )

    if "AutoModelForCausalLM" in _auto_map:
        model = AutoModelForCausalLM.from_pretrained(**_load_kwargs).to(device)
    else:
        model = AutoModel.from_pretrained(**_load_kwargs).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        grpo_config.model_path, trust_remote_code=True, revision=_revision
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    from rl.trainers.train_utils import get_mask_id as _get_mask_id

    grpo_config.mask_id = _get_mask_id(
        tokenizer=tokenizer, model=model, default=grpo_config.mask_id
    )
    if rmt.is_main_process():
        print(f"Using mask_id: {grpo_config.mask_id}")

    if not hasattr(model, "prepare_inputs_for_generation"):
        import types

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}

        model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation, model
        )

    if getattr(model.config, "model_type", "") == "llada2_moe":
        import functools

        _llada2_block_length = getattr(grpo_config, "block_length", 32)
        _inner = getattr(model, "model", model)
        _inner_cls = _inner.__class__
        _orig_inner_fwd = _inner_cls.forward

        @functools.wraps(_orig_inner_fwd)
        def _llada2_inner_forward(
            self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
            if attention_mask is None:
                if input_ids is not None:
                    bs, sl = input_ids.shape[:2]
                    dev = input_ids.device
                elif inputs_embeds is not None:
                    bs, sl = inputs_embeds.shape[:2]
                    dev = inputs_embeds.device
                else:
                    return _orig_inner_fwd(
                        self, input_ids=input_ids, attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds, **kwargs,
                    )
                bl = _llada2_block_length
                n_blocks = (sl + bl - 1) // bl
                block_mask = torch.tril(torch.ones(n_blocks, n_blocks, device=dev))
                attention_mask = (
                    block_mask.repeat_interleave(bl, dim=0)[:sl, :]
                    .repeat_interleave(bl, dim=1)[:, :sl]
                    .unsqueeze(0).unsqueeze(0).log().to(torch.bfloat16).expand(bs, -1, -1, -1)
                )
            return _orig_inner_fwd(
                self, input_ids=input_ids, attention_mask=attention_mask,
                inputs_embeds=inputs_embeds, **kwargs,
            )

        _inner_cls.forward = _llada2_inner_forward
        if rmt.is_main_process():
            print(f"Patched LLaDA2-MoE inner model (block_length={_llada2_block_length})")

    controller = None
    eval_callbacks = []
    eval_domains_added = set()
    tt_eval = str(getattr(grpo_config, "trainer_type", "") or "")
    is_b1_eval = ("b1_" in tt_eval) or tt_eval.startswith("b1_")

    def _with_block_instruction(system_prompt: str) -> str:
        """
        For b1 eval, encourage explicit \\block markers so dynamic_generate can
        reliably detect block boundaries across tasks.
        """
        sp = str(system_prompt or "")
        if "\\block" in sp:
            return sp
        suffix = (
            "\n\nAppend the tag \\block directly to the end of the last sentence of each reasoning step without starting a new line."
        )
        return sp + suffix

    _B1_BLOCK_PROMPT_DOMAINS = {
        "sudoku",
        "kodcode",
        "mbpp",
        "humaneval",
        "knights_and_knaves",
    }

    for domain_name in domain_names:
        if domain_name in rmt.DATASET_MAP:
            ds_cls = rmt.DATASET_MAP[domain_name]
            ds_kwargs = dict(
                tokenizer=tokenizer,
                subsample=rmt.SUB_SAMPLE_MAP.get(domain_name, 100),
                num_examples=0,
                add_reasoning=True,
            )
            # For b1 variants: ensure eval prompts ask for \\block markers even for
            # datasets whose default system prompt doesn't mention it (e.g. sudoku/code/knights).
            if is_b1_eval and domain_name in _B1_BLOCK_PROMPT_DOMAINS:
                try:
                    val_dataset = ds_cls(**ds_kwargs)
                    sp0 = getattr(val_dataset, "system_prompt", "")
                    setattr(val_dataset, "system_prompt", _with_block_instruction(sp0))
                except Exception:
                    ds_kwargs["system_prompt"] = _with_block_instruction("")
                    val_dataset = ds_cls(**ds_kwargs)
            else:
                val_dataset = ds_cls(**ds_kwargs)
            eval_callbacks.append(
                rmt._r1_per_domain_eval_callback(
                    domain_name,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    block_size_controller=controller,
                    gen_length=grpo_config.max_completion_length,
                    steps=grpo_config.diffusion_steps,
                    fallback_block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            )
            eval_domains_added.add(domain_name)

    if "guru" in domain_names:
        for ed in rmt.GURU_TRAIN_EVAL_DOMAINS:
            if ed in eval_domains_added or ed not in rmt.DATASET_MAP:
                continue
            ds_cls = rmt.DATASET_MAP[ed]
            ds_kwargs = dict(
                tokenizer=tokenizer,
                subsample=rmt.SUB_SAMPLE_MAP.get(ed, 100),
                num_examples=0,
                add_reasoning=True,
            )
            if is_b1_eval and ed in _B1_BLOCK_PROMPT_DOMAINS:
                try:
                    val_dataset = ds_cls(**ds_kwargs)
                    sp0 = getattr(val_dataset, "system_prompt", "")
                    setattr(val_dataset, "system_prompt", _with_block_instruction(sp0))
                except Exception:
                    ds_kwargs["system_prompt"] = _with_block_instruction("")
                    val_dataset = ds_cls(**ds_kwargs)
            else:
                val_dataset = ds_cls(**ds_kwargs)
            eval_callbacks.append(
                rmt._r1_per_domain_eval_callback(
                    ed,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    block_size_controller=controller,
                    gen_length=grpo_config.max_completion_length,
                    steps=grpo_config.diffusion_steps,
                    fallback_block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            )
            eval_domains_added.add(ed)

    primary_eval_domain = domain_names[0]
    if primary_eval_domain not in rmt.DATASET_MAP:
        for cand in (*domain_names, *rmt.GURU_TRAIN_EVAL_DOMAINS):
            if cand in rmt.DATASET_MAP:
                primary_eval_domain = cand
                break

    primary_val_dataset = None
    if primary_eval_domain in rmt.DATASET_MAP:
        ds_cls = rmt.DATASET_MAP[primary_eval_domain]
        ds_kwargs = dict(
            tokenizer=tokenizer,
            subsample=rmt.SUB_SAMPLE_MAP.get(primary_eval_domain, 100),
            num_examples=0,
            add_reasoning=True,
        )
        if is_b1_eval and primary_eval_domain in _B1_BLOCK_PROMPT_DOMAINS:
            try:
                primary_val_dataset = ds_cls(**ds_kwargs)
                sp0 = getattr(primary_val_dataset, "system_prompt", "")
                setattr(primary_val_dataset, "system_prompt", _with_block_instruction(sp0))
            except Exception:
                ds_kwargs["system_prompt"] = _with_block_instruction("")
                primary_val_dataset = ds_cls(**ds_kwargs)
        else:
            primary_val_dataset = ds_cls(**ds_kwargs)

    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )

    try:
        import trl.trainer.grpo_trainer as _grpo_mod

        orig = getattr(_grpo_mod.GRPOTrainer, "_get_train_sampler", None)
        if orig is not None:
            def _wrapped_get_train_sampler(self, train_dataset=None):
                try:
                    return orig(self, train_dataset)
                except TypeError:
                    return orig(self)

            _grpo_mod.GRPOTrainer._get_train_sampler = _wrapped_get_train_sampler
    except Exception:
        pass

    R1TrainerClass = rmt.get_r1_trainer_class(r1_tt)
    if rmt.is_main_process():
        print(f"[Block-R1] Using trainer: {R1TrainerClass.__name__}")

    eval_callbacks.append(rmt.R1ControllerSaveCallback(controller))

    trainer = R1TrainerClass(
        args=grpo_config,
        model=model,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        eval_dataset=primary_val_dataset,
        callbacks=eval_callbacks,
        block_size_controller=controller,
    )

    if rmt.is_main_process():
        if wandb.run is None:
            wandb.init(project=grpo_config.wandb_project, name=grpo_config.run_name)

    trainer.train()

    if rmt.is_main_process():
        meta = {
            "mode": "block_r1_dataset",
            "train_jsonl": jsonl_path,
            "domains_in_file": sorted(domains_in_file),
            "domains_for_reward": sorted(domains_for_reward),
            "domain_names_config": domain_names,
            "trainer_type_requested": tt_raw,
            "trainer_type_r1": r1_tt,
        }
        final_path = os.path.join(grpo_config.output_dir, "r1.json")
        os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[Block-R1] Wrote metadata to {final_path} (no bandit controller).")


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    grpo_config.remove_unused_columns = False
    grpo_config.label_names = ["completion_ids"]
    main(grpo_config=grpo_config, model_config=model_config)
