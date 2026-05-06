"""
Acknowledgment: This code is adapted from the diffu-grpo (d1) official implementation.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import TrainingArguments

TrainerType = str
MaddaMaskingSchedule = Literal["cosine", "linear", "pow", "sigmoid"]


@dataclass
class DiffuGRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation

        temperature (`float`, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int` or `None`, *optional*, defaults to `50`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled.
        min_p (`float` or `None`, *optional*, defaults to `None`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        cache_implementation (`str` or `None`, *optional*, defaults to `None`):
            Implementation of the cache method for faster generation when use_vllm is set to False.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
            automatically select the next available GPU after the last one used for training. This assumes that
            training has not already occupied all available GPUs. If only one device is available, the device will be
            shared between both training and vLLM.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        vllm_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        vllm_enable_prefix_caching (`bool`, *optional*, defaults to `True`):
            Whether to enable prefix caching in vLLM. If set to `True` (default), ensure that the model and the hardware
            support this feature.
        vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving training
            speed, but may be numerically unstable for long training runs.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originites from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is
            installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    model_path: Optional[str] = field(
        default="",
    )
    sft_path: Optional[str] = field(
        default="",
    )
    trainer_type: Optional[TrainerType] = field(
        default="d1",
        metadata={"help": "Training method that we use"},
    )
    madda_maskingSchedule: Optional[MaddaMaskingSchedule] = field(
        default="linear",
        metadata={"help": "Masking schedule for MADDA. Used for random samples"},
    )

    use_p_ref_weight: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use pi ref in the weight of rev_grpo_trainer_d1clip. Not the same as using it in importance sampling"
        },
    )
    wandb_project: Optional[str] = field(
        default="var-diff",
        metadata={"help": "wandb project to report to"},
    )

    likelihood_estimation: Optional[str] = field(
        default="mix",
        metadata={"help": "Likelihood estimation method."},
    )

    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )

    # Parameters that control generation, following d1 and rl settings
    temperature: float = field(
        default=0.9,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Implementation of the cache method for faster generation when use_vllm is set to False."
        },
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training. This assumes "
            "that training has not already occupied all available GPUs."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    vllm_enable_prefix_caching: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True` (default), ensure that the model and "
            "the hardware support this feature."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={
            "help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."
        },
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={
            "help": "Number of iterations per batch (denoted as μ in the algorithm)."
        },
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={"help": "Whether to log the completions during training."},
    )

    generation_batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": "Batch size for generation. If not set, the batch size will be equal to the number of generations."
        },
    )

    block_length: Optional[int] = field(
        default=64,
        metadata={"help": "diffusion block length"},
    )
    diffusion_steps: Optional[int] = field(
        default=64,
    )
    cfg_scale: Optional[float] = field(
        default=0.0,
    )
    remasking: Optional["str"] = field(
        default="low_confidence",
    )
    dataset: Optional[str] = field(
        default="gsm8k",
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    p_mask_prompt: float = field(
        default=0.3,
        metadata={"help": "Probability of masking the prompt."},
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id. Default is from Llada"},
    )
    random_masking: bool = field(
        default=True,
        metadata={"help": "Whether to randomly mask tokens."},
    )
    # GDPO likelihood estimation
    logp_estimator: str = field(
        default="gauss-2",
        metadata={
            "help": "Method for likelihood estimation. Options: 'mc', 'gauss-x', etc. "
            "Will be passed to the likelihood estimator."
        },
    )

    # R1: Cross-Domain Dynamic Block Size RL
    use_r1: bool = field(
        default=False,
        metadata={
            "help": "Enable R1 cross-domain dynamic block size adaptation. "
            "When True, block sizes are selected per-domain via a multi-armed bandit."
        },
    )
    r1_domains: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated domain names for R1 multi-domain training. "
            "E.g. 'math,countdown,kodcode,mmlu'. Each domain loads its own dataset and reward."
        },
    )
    r1_block_size_candidates: str = field(
        default="16,32,64,128",
        metadata={
            "help": "Comma-separated block size candidates for R1 bandit selection."
        },
    )
    r1_block_size_lr: float = field(
        default=0.1,
        metadata={"help": "EMA learning rate for block size Q-value updates."},
    )
    r1_exploration_rate: float = field(
        default=0.3,
        metadata={"help": "Epsilon-greedy exploration rate for block size selection."},
    )
    r1_efficiency_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight γ for block size efficiency bias in R1 bandit update: "
            "signal = advantage + γ * b / b_max  (larger block = more parallel = bonus)."
        },
    )
    r1_adaptive_proto: bool = field(
        default=False,
        metadata={
            "help": "R1: use DSCB adaptive prototypes (dynamic centroids + per-proto Q) "
            "instead of fixed per-domain centroids. Eval/test route by nearest prototype."
        },
    )
    r1_max_prototypes: int = field(
        default=64,
        metadata={"help": "Max prototype slots when r1_adaptive_proto is True."},
    )
    r1_proto_gamma: float = field(
        default=2.5,
        metadata={
            "help": "DSCB confidence multiplier γ: new prototype if "
            "s_max < μ - γ·σ (similarity vs nearest centroid)."
        },
    )
    r1_proto_beta_sim: float = field(
        default=0.1,
        metadata={"help": "EMA β for online μ and σ² of per-prototype similarities."},
    )
    r1_proto_min_samples_stat: int = field(
        default=3,
        metadata={
            "help": "Minimum assignments to a prototype before DSCB outlier test applies."
        },
    )

    # Block-R1: offline JSONL with per-row ``br1_best_block_size`` (see ``rl.block_r1`` / ``run_block_r1``)
    use_block_r1_dataset: bool = field(
        default=False,
        metadata={
            "help": "Use each sample's br1_best_block_size from the dataset for generation "
            "(no R1 bandit). Requires train.jsonl from build_block_r1."
        },
    )
    block_r1_train_jsonl: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to train.jsonl (block_r1 multi-domain export). "
            "Used when use_block_r1_dataset is True."
        },
    )
    block_r1_balance_domains: bool = field(
        default=False,
        metadata={
            "help": "Block-R1: balance domains in offline train.jsonl by subsampling "
            "each domain to the same size (min count across domains), then shuffling. "
            "If false, sampling follows the JSONL per-domain frequencies."
        },
    )

    # StableDRL (SPG + ELBO/EUBO mix + optional SNIS; ported from dLLM-DRL)
    # Defaults match dLLM-DRL ``SPGConfig`` / ``training_loop_spg_snis`` (use_snis off, num_mc_samples=1).
    # If False, will not use SNIS and only use ELBO/EUBO for SPG.
    # If True, will use SNIS and ELBO/EUBO for StableDRL.
    stable_drl_use_snis: bool = field(
        default=True,
        metadata={"help": "Enable self-normalized importance sampling stabilization (dLLM-DRL default: false)."},
    )
    stable_drl_ais_clip_iw: float = field(
        default=5.0,
        metadata={"help": "Clip importance weights (linear scale) for SNIS."},
    )
    stable_drl_spg_beta: float = field(
        default=1.5,
        metadata={"help": "EUBO temperature β (eubo_beta in SPGConfig)."},
    )
    stable_drl_spg_omega: float = field(
        default=0.5,
        metadata={"help": "Mix weight for ELBO/EUBO (mix_weight); 0=ELBO, 1=EUBO, else mix."},
    )
    stable_drl_logp_estimation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override: 'mix'|'elbo'|'eubo'|'zero'. If None, derived from stable_drl_spg_omega."
        },
    )
    stable_drl_num_mc_samples: int = field(
        default=1,
        metadata={"help": "MC samples num_t in SPG forward process (dLLM-DRL default: 1)."},
    )
    stable_drl_p_mask_perturb: float = field(
        default=0.15,
        metadata={"help": "Prompt mask probability for SPG forward process."},
    )
    stable_drl_forward_type: str = field(
        default="block_random",
        metadata={"help": "SPG forward_type: block_random | block_all | random | all."},
    )
    stable_drl_anti_short_boxed: bool = field(
        default=True,
        metadata={"help": "Add anti-short-\\boxed-only shaping reward (dLLM-DRL default)."},
    )
    stable_drl_advantage_mode: str = field(
        default="center",
        metadata={
            "help": "Grouped advantages: 'center' = r - mean_g (dLLM-DRL llada_svpo.generate_and_score_completions_spg); "
            "'grpo_std' = (r-mean_g)/(std_g+eps) like d1/wd1 in this repo."
        },
    )

    # ESPO (ported from dLLM-ESPO; ELBO-based sequence-level policy optimization)
    espo_num_mc: int = field(
        default=2,
        metadata={"help": "ESPO: number of Monte Carlo mask seeds per iteration for ELBO estimation."},
    )
    espo_reduce_var: bool = field(
        default=True,
        metadata={"help": "ESPO: use coupled estimator to reduce ELBO variance (matches dLLM-ESPO default)."},
    )
