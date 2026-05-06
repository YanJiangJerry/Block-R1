import torch
import torch.distributed as dist
import wandb
from data_utils import (
    get_countdown_questions,
    get_gsm8k_questions,
    get_math_questions,
    get_sudoku_questions,
    get_mbpp_questions,
    get_humaneval_questions,
    get_kodcode_light_rl_10k,
    get_mmlu_questions,
    get_mmlu_pro_questions,
    get_hellaswag_questions,
    get_arc_c_questions,
    get_arc_e_questions,
    set_random_seed,
)
from peft import LoraConfig
from reward_func import (
    boxed_and_answer_tags_format_reward,
    correctness_reward_func,
    correctness_reward_func_math,
    countdown_reward_func,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    sudoku_reward_func,
    xmlcount_reward_func,
    code_reward_func,
    get_code_format_reward,
    code_reward,
    mc_reward_func,
)
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import ModelConfig, TrlParser

from rl.trainers.diffu_grpo_config import DiffuGRPOConfig
from rl.llada2_compat import (
    ensure_transformers_kwargs,
    load_diffusion_model,
    patch_llada2_block_causal_attention,
)

# Custom imports
from rl.trainers.diffu_grpo_trainer import DiffuGRPOTrainer


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main(grpo_config, model_config):
    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    elif grpo_config.dataset == "mbpp":
        # Use MBPP's own training set (task_id 511-974)
        dataset = get_mbpp_questions()
        code_format = get_code_format_reward(language="python")
        reward_functions = [code_format, code_reward]
    elif grpo_config.dataset == "humaneval":
        # Use KodCode as training set (larger), HumanEval only for evaluation
        dataset = get_kodcode_light_rl_10k("train")
        code_format = get_code_format_reward(language="python")
        reward_functions = [code_format, code_reward]
    elif grpo_config.dataset == "kodcode":
        dataset = get_kodcode_light_rl_10k("train")
        code_format = get_code_format_reward(language="python")
        reward_functions = [code_format, code_reward]
    elif grpo_config.dataset == "mmlu":
        dataset = get_mmlu_questions("auxiliary_train")
        reward_functions = [mc_reward_func]
    elif grpo_config.dataset == "mmlu_pro":
        # MMLU-Pro has only 70 validation samples, use MMLU's auxiliary_train for training
        # Evaluation will still be on MMLU-Pro test set
        dataset = get_mmlu_questions("auxiliary_train")
        reward_functions = [mc_reward_func]
    elif grpo_config.dataset == "hellaswag":
        dataset = get_hellaswag_questions("train")
        reward_functions = [mc_reward_func]
    elif grpo_config.dataset == "arc_c":
        dataset = get_arc_c_questions("train")
        reward_functions = [mc_reward_func]
    elif grpo_config.dataset == "arc_e":
        dataset = get_arc_e_questions("train")
        reward_functions = [mc_reward_func]

    # Shuffle dataset with fixed seed for reproducibility (except mbpp which has fixed train split)
    if grpo_config.dataset != "mbpp":
        dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(
            range(0, len(dataset) - 500)
        )  # Leave last 500 for evaluation
    elif grpo_config.dataset == "mbpp":
        # MBPP training set: task_id 511-974 (indices 500-973 in 0-indexed)
        train_set = dataset.select(range(500, 974))
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    ensure_transformers_kwargs()
    model = load_diffusion_model(
        grpo_config.model_path,
        torch_dtype=torch.bfloat16,
        device=device,
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        grpo_config.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    patch_llada2_block_causal_attention(model, getattr(grpo_config, "block_length", 32))

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Initialize and run trainer
    trainer = DiffuGRPOTrainer(
        args=grpo_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
    )
    if is_main_process():
        wandb.init(project="d1", name="diff-grpo")
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
