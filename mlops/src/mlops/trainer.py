import os
import time

# Set environment variables BEFORE any HuggingFace imports
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"
os.environ["HF_DATASETS_CACHE"] = "./outputs/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ["NCCL_TIMEOUT"] = "7200"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEFAULT_TIMEOUT"] = "7200"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:true"
)

# NCCL: prefer loopback for single-node multi-GPU
_fc_nccl_ifname = os.environ.get("FC_NCCL_SOCKET_IFNAME")
if _fc_nccl_ifname:
    os.environ["NCCL_SOCKET_IFNAME"] = _fc_nccl_ifname
else:
    try:
        _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        _world_size = 1
    _nccl_ifname = (os.environ.get("NCCL_SOCKET_IFNAME") or "").strip().lower()
    if _world_size > 1 and (_nccl_ifname in {"", "eth0"}):
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

import base64
import json
import argparse
import logging
import pathlib
import glob
import torch
import psutil
from datetime import datetime
from pydantic import BaseModel
from accelerate.data_loader import DataLoaderShard
from datasets import load_dataset, disable_caching
from transformers import (
    PrinterCallback,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    _PEFT_AVAILABLE = True
    # Compatibility: newer peft versions reference float8_e8m0fnu which
    # only exists in PyTorch 2.6+. Provide a fallback for torch 2.5.x.
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.float32
except ImportError:
    _PEFT_AVAILABLE = False

try:
    import deepspeed  # noqa: F401
except Exception:
    deepspeed = None


_MODEL_LOAD_SFT_KWARGS = {
    "load_in_4bit",
    "load_in_8bit",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_quant_type",
    "bnb_4bit_use_double_quant",
}

_TRAINER_ONLY_SFT_KWARGS = _MODEL_LOAD_SFT_KWARGS | {
    "chat_template",
}


_SIMPLE_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}

{% elif message['role'] == 'user' %}User: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% if not loop.last %}

{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"""


from mlops.training_artifacts import persist_training_artifacts
from mlops.token_batching import TokenBudgetBatchSampler, TokenBudgetConfig

disable_caching()


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_memory(stage: str):
    """Log memory usage at different stages for debugging."""
    try:
        process = psutil.Process()
        mem = process.memory_info()
        vm = psutil.virtual_memory()
        logger.info(
            f"[MEMORY - {stage}] Process RAM: {mem.rss / 1024**3:.2f}GB | "
            f"System: {vm.used / 1024**3:.2f}GB / {vm.total / 1024**3:.2f}GB ({vm.percent}% used)"
        )
    except Exception as e:
        logger.warning(f"Failed to log memory at {stage}: {e}")


class TrainerConfig(BaseModel):
    """Configuration for model training.

    NOTE: Only single-node, multi-GPU training is supported (all GPUs on one
    machine). Multi-node distributed training is not supported.
    """

    model: str
    datapath: str
    epochs: int = 2
    eval_strategy: str = "steps"
    eval_steps: float = 0.1
    save_strategy: str = "no"
    save_steps: float | None = None
    ft_model_name: str = ""
    sft_kwargs: dict = {}
    use_deepspeed: bool = False
    deepspeed_config: dict = None
    deepspeed_stage: int = 2
    deepspeed_cpu_offload: bool = False
    deepspeed_save_full_model: bool = True
    resume_from_checkpoint: str | None = None
    auto_resume: bool | None = None
    model_cache_dir: str = "./.model_cache"
    # External experiment trackers for the HF Trainer (e.g. ["wandb"],
    # ["tensorboard"]). Empty disables all reporting -- the standalone default.
    report_to: list[str] = []
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    save_lora_adapter_only: bool = False
    # Loss masking: when True, labels for prompt tokens (system + user
    # messages) are set to -100 so only the assistant completion contributes
    # to the loss. This matters for chat-SFT because the prompt can be 50-100x
    # longer than the completion; without masking, the model optimizes
    # prompt-regurgitation instead of the task.
    mask_prompt_loss: bool = True


def _torch_dtype_from_name(dtype_name: str):
    normalized = str(dtype_name).lower().replace("torch.", "")
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype for quantization: {dtype_name}")


def _retained_prompt_tokens_after_truncation(
    *,
    prompt_len: int,
    full_len: int,
    kept_len: int,
    truncation_side: str,
) -> int:
    """Return how many retained tokens still belong to the prompt.

    When prompt-loss masking is enabled we need the prompt boundary *after*
    truncation, not before it. Otherwise long prompts can mask the entire
    retained sequence and silently turn the sample into a zero-loss example.
    """

    truncation_side = (truncation_side or "right").lower()
    if truncation_side == "left":
        dropped_tokens = max(0, full_len - kept_len)
        return max(0, prompt_len - dropped_tokens)
    return min(prompt_len, kept_len)


def _build_tokenize_fn(tokenizer, max_len: int, mask_prompt_loss: bool):
    """Return a HF datasets-compatible tokenize_function.

    When mask_prompt_loss=True, `labels` is emitted with prompt tokens masked
    to -100 so the loss is computed only over the assistant completion. The
    prompt boundary is found by tokenizing the message list without the final
    assistant turn (with add_generation_prompt=True) and using its length as
    the boundary.
    """

    def tokenize_function(examples):
        full_texts = []
        prompt_lens: list[int] | None = [] if mask_prompt_loss else None
        full_lens: list[int] | None = [] if mask_prompt_loss else None
        truncation_side = getattr(tokenizer, "truncation_side", "right")

        for messages in examples["messages"]:
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_texts.append(full_text)
            if mask_prompt_loss:
                # Find assistant-turn boundary by rendering the messages up to
                # (but not including) the final assistant reply, with a
                # generation prompt suffix.
                if not messages or messages[-1]["role"] != "assistant":
                    # Degenerate sample: no assistant turn. Mark entire
                    # sequence as prompt (everything masked — sample
                    # contributes nothing, which is correct).
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                else:
                    prompt_text = tokenizer.apply_chat_template(
                        messages[:-1], tokenize=False, add_generation_prompt=True
                    )
                # Token length only — no truncation here, we just need count.
                prompt_ids = tokenizer(
                    prompt_text, truncation=False, padding=False
                )["input_ids"]
                prompt_lens.append(len(prompt_ids))
                full_ids = tokenizer(
                    full_text, truncation=False, padding=False
                )["input_ids"]
                full_lens.append(len(full_ids))

        batch = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        if mask_prompt_loss:
            labels = []
            supervised_tokens = []
            for ids, p_len, full_len in zip(batch["input_ids"], prompt_lens, full_lens):
                lbl = list(ids)
                cutoff = _retained_prompt_tokens_after_truncation(
                    prompt_len=int(p_len),
                    full_len=int(full_len),
                    kept_len=len(lbl),
                    truncation_side=truncation_side,
                )
                for i in range(cutoff):
                    lbl[i] = -100
                labels.append(lbl)
                supervised_tokens.append(len(lbl) - cutoff)
            batch["labels"] = labels
            batch["supervised_tokens"] = supervised_tokens

        # Length column enables HF Trainer's group_by_length=True to bucket
        # similar-length sequences into the same batch, drastically reducing
        # padding waste on variable-length data.
        batch["length"] = [len(ids) for ids in batch["input_ids"]]

        return batch

    return tokenize_function


def _log_and_filter_supervision(dataset, split_name: str):
    if not hasattr(dataset, "column_names") or "supervised_tokens" not in dataset.column_names:
        return dataset

    supervised_tokens = [int(x) for x in dataset["supervised_tokens"]]
    if not supervised_tokens:
        logger.warning(f"{split_name}: no supervised-token metadata found after tokenization")
        return dataset

    total = len(supervised_tokens)
    zero_count = sum(tokens <= 0 for tokens in supervised_tokens)
    short_count = sum(tokens < 8 for tokens in supervised_tokens)
    sorted_tokens = sorted(supervised_tokens)
    p50 = sorted_tokens[len(sorted_tokens) // 2]
    p90 = sorted_tokens[min(len(sorted_tokens) - 1, int(len(sorted_tokens) * 0.9))]
    logger.info(
        f"{split_name}: supervised tokens median={p50}, p90={p90}, "
        f"zero={zero_count}/{total}, lt8={short_count}/{total}"
    )

    if zero_count:
        logger.warning(
            f"{split_name}: dropping {zero_count}/{total} zero-supervision samples "
            "caused by truncation or missing assistant replies"
        )
        dataset = dataset.filter(
            lambda example: int(example["supervised_tokens"]) > 0,
            desc=f"Filtering {split_name} zero-supervision samples",
        )

    return dataset


class TokenBudgetSFTTrainer(SFTTrainer):
    def __init__(self, *args, token_budget_config: TokenBudgetConfig, **kwargs):
        self.token_budget_config = token_budget_config
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if not hasattr(train_dataset, "column_names"):
            raise TypeError(
                "Token-budget batching requires a Hugging Face dataset with "
                "accessible column names."
            )

        length_column_name = self.token_budget_config.length_column_name
        if length_column_name not in train_dataset.column_names:
            raise ValueError(
                "Token-budget batching requires a tokenized dataset with a "
                f"'{length_column_name}' column."
            )

        lengths = train_dataset[length_column_name]
        dataset = self._remove_unused_columns(train_dataset, description="Training")
        batch_sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            rank=max(0, int(getattr(self.args, "process_index", 0))),
            world_size=max(1, int(getattr(self.args, "world_size", 1))),
            config=self.token_budget_config,
            seed=int(
                self.args.data_seed
                if self.args.data_seed is not None
                else self.args.seed
            ),
        )

        dataloader_kwargs = {
            "collate_fn": self.data_collator,
            "batch_sampler": batch_sampler,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = (
                self.args.dataloader_persistent_workers
            )
            if self.args.dataloader_prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = (
                    self.args.dataloader_prefetch_factor
                )

        return DataLoaderShard(
            dataset,
            _drop_last=batch_sampler.config.drop_last,
            **dataloader_kwargs,
        )


class Trainer:
    def _build_model_load_kwargs(self, requested_attn_impl: str | None = None) -> dict:
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": self.trust_remote_code,
        }
        if requested_attn_impl:
            load_kwargs["attn_implementation"] = requested_attn_impl

        sft_kwargs = self.config.sft_kwargs or {}
        load_in_4bit = bool(sft_kwargs.get("load_in_4bit", False))
        load_in_8bit = bool(sft_kwargs.get("load_in_8bit", False))
        if load_in_4bit and load_in_8bit:
            raise ValueError("Only one of load_in_4bit/load_in_8bit may be true")

        if load_in_4bit or load_in_8bit:
            bnb_kwargs = {
                "load_in_4bit": load_in_4bit,
                "load_in_8bit": load_in_8bit,
            }
            if load_in_4bit:
                bnb_kwargs.update(
                    {
                        "bnb_4bit_compute_dtype": _torch_dtype_from_name(
                            sft_kwargs.get("bnb_4bit_compute_dtype", "bfloat16")
                        ),
                        "bnb_4bit_quant_type": str(
                            sft_kwargs.get("bnb_4bit_quant_type", "nf4")
                        ),
                        "bnb_4bit_use_double_quant": bool(
                            sft_kwargs.get("bnb_4bit_use_double_quant", True)
                        ),
                    }
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
            if torch.cuda.is_available():
                device_index = self.local_rank if self.local_rank >= 0 else 0
                load_kwargs["device_map"] = {"": device_index}
            if self.is_main_process:
                quantization = "4-bit" if load_in_4bit else "8-bit"
                logger.info("Loading base model with %s quantization", quantization)

        return load_kwargs

    def _load_causal_lm(self, **load_kwargs):
        try:
            return AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        except TypeError:
            load_kwargs = dict(load_kwargs)
            load_kwargs.pop("trust_remote_code", None)
            load_kwargs.pop("attn_implementation", None)
            return AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

    def _configure_chat_template(self, tokenizer) -> None:
        if getattr(tokenizer, "chat_template", None):
            return

        requested_template = (self.config.sft_kwargs or {}).get("chat_template")
        if not requested_template:
            return

        if str(requested_template).lower() != "simple":
            raise ValueError(f"Unsupported chat_template: {requested_template}")

        tokenizer.chat_template = _SIMPLE_CHAT_TEMPLATE
        logger.info("Tokenizer had no chat template; using simple fallback template")

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.is_main_process = self.local_rank in [-1, 0]

        # Enforce single-node: all communication must stay on loopback
        if self.world_size > 1:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        try:
            self.trust_remote_code = bool(
                self.config.sft_kwargs.get("trust_remote_code", False)
            )
        except Exception:
            self.trust_remote_code = False

        try:
            self.rank = int(
                os.environ.get(
                    "RANK",
                    0
                    if self.is_main_process
                    else (self.local_rank if self.local_rank >= 0 else 0),
                )
            )
        except Exception:
            self.rank = 0
        try:
            self.world_size_env = int(
                os.environ.get("WORLD_SIZE", self.world_size)
            )
        except Exception:
            self.world_size_env = self.world_size

        # The model is loaded directly from config.model, a Hugging Face hub id
        # or a local path.
        self.model_path = self.config.model
        self.model_family_hint = self.config.model
        if self.is_main_process:
            logger.info(f"Using model path: {self.model_path}")

        if self.is_distributed:
            logger.info(
                f"Single-node multi-GPU training: WORLD_SIZE={self.world_size}, LOCAL_RANK={self.local_rank}"
            )
        elif self.world_size > 1:
            logger.info(f"Single-node multi-GPU: {self.world_size} GPUs available")
        else:
            logger.info("Single GPU/CPU training mode")

        effective_save_steps = (
            self.config.save_steps
            if self.config.save_steps is not None
            else self.config.eval_steps
        )

        if self.world_size > 1:
            logger.info("Configuring for multi-GPU training")
            sft_params = {
                "num_train_epochs": self.config.epochs,
                "bf16": True,
                "eval_strategy": self.config.eval_strategy,
                "eval_steps": self.config.eval_steps,
                "save_strategy": self.config.save_strategy,
                "save_steps": effective_save_steps,
                "report_to": self.config.report_to,
                "max_seq_length": 1024,
                "dataset_num_proc": None,
                "dataloader_num_workers": 0,
                "dataloader_prefetch_factor": None,
                "packing": False,
                "eval_accumulation_steps": 4,
                "ddp_find_unused_parameters": True if self.config.use_lora else False,
                "ddp_backend": "nccl",
                "gradient_checkpointing": True,
                "per_device_eval_batch_size": 1,
                "torch_compile": False,
            }
        else:
            logger.info("Configuring for single-GPU training")
            sft_params = {
                "num_train_epochs": self.config.epochs,
                "bf16": True,
                "eval_strategy": self.config.eval_strategy,
                "eval_steps": self.config.eval_steps,
                "save_strategy": self.config.save_strategy,
                "save_steps": effective_save_steps,
                "report_to": self.config.report_to,
                "max_seq_length": 1024,
                "dataset_num_proc": 1,
                "dataloader_num_workers": 2,
                "dataloader_prefetch_factor": 2,
                "packing": False,
                "eval_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "per_device_eval_batch_size": 1,
                "torch_compile": False,
            }

        _override_ok = {
            "dataloader_num_workers",
            "dataloader_prefetch_factor",
            "logging_steps",
            "packing",
            "gradient_checkpointing",
            "ddp_find_unused_parameters",
            "torch_compile",
            "max_seq_length",
        }

        for key, value in self.config.sft_kwargs.items():
            if key in _TRAINER_ONLY_SFT_KWARGS:
                continue
            if key not in sft_params:
                sft_params[key] = value
                continue
            if key in _override_ok:
                sft_params[key] = value
                continue
            logger.warning(
                f"sft_kwargs key '{key}' conflicts with explicit parameter"
            )

        self.token_budget_config: TokenBudgetConfig | None = None
        if sft_params.get("train_sampling_strategy") == "token_budget":
            max_tokens_per_device = sft_params.pop("max_tokens_per_device", None)
            if max_tokens_per_device is None:
                raise ValueError(
                    "train_sampling_strategy='token_budget' requires "
                    "'max_tokens_per_device' in sft_kwargs."
                )

            max_examples_per_device = sft_params.pop(
                "max_examples_per_device", None
            )
            if max_examples_per_device is None:
                max_examples_per_device = sft_params.get("per_device_train_batch_size")
            if max_examples_per_device is None:
                raise ValueError(
                    "Token-budget batching requires either "
                    "'max_examples_per_device' or 'per_device_train_batch_size'."
                )

            self.token_budget_config = TokenBudgetConfig(
                max_tokens_per_device=int(max_tokens_per_device),
                max_examples_per_device=int(max_examples_per_device),
                length_column_name=str(
                    sft_params.get("length_column_name", "length")
                ),
                megabatch_mult=int(sft_params.pop("token_budget_megabatch_mult", 50)),
                drop_last=bool(sft_params.pop("token_budget_drop_last", False)),
            )
            sft_params.pop("train_sampling_strategy", None)

        if sft_params.get("train_sampling_strategy") == "length":
            logger.info(
                "Normalizing train_sampling_strategy='length' to "
                "'group_by_length'"
            )
            sft_params["train_sampling_strategy"] = "group_by_length"

        self.tokenization_num_proc = sft_params.pop("tokenization_num_proc", None)
        self.tokenization_batch_size = int(
            sft_params.pop(
                "tokenization_batch_size",
                os.environ.get("FC_TOKENIZATION_BATCH_SIZE", 1024),
            )
        )
        self.tokenization_writer_batch_size = int(
            sft_params.pop(
                "tokenization_writer_batch_size",
                os.environ.get(
                    "FC_TOKENIZATION_WRITER_BATCH_SIZE",
                    self.tokenization_batch_size,
                ),
            )
        )
        self.tokenization_wait_timeout = int(
            sft_params.pop(
                "tokenization_wait_timeout",
                os.environ.get("FC_TOKENIZATION_WAIT_TIMEOUT", 14400),
            )
        )

        legacy_group_by_length = sft_params.pop("group_by_length", None)
        if legacy_group_by_length is not None:
            logger.warning(
                "sft_kwargs key 'group_by_length' is unsupported by the "
                "installed TRL version; use "
                "train_sampling_strategy='group_by_length' instead"
            )
            if legacy_group_by_length:
                sft_params["train_sampling_strategy"] = "group_by_length"

        outputs_dir = "./outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        sft_params["output_dir"] = outputs_dir + "/model-checkpoints"
        sft_params["logging_dir"] = outputs_dir + "/logs"

        sft_params.pop("trust_remote_code", None)

        if self.config.use_deepspeed:
            ds_config = self._load_deepspeed_config()
            ds_config_path = os.path.join(outputs_dir, "deepspeed_config_runtime.json")
            with open(ds_config_path, "w") as f:
                json.dump(ds_config, f, indent=2)
            sft_params["deepspeed"] = ds_config_path
            logger.info(
                f"DeepSpeed config saved: ZeRO stage {ds_config['zero_optimization']['stage']}"
            )

        self.sft_params = sft_params
        self.sft_config = None

    def _ensure_sft_config(self) -> SFTConfig:
        """Build SFTConfig once with the same max-length normalization everywhere."""
        if self.sft_config is not None:
            return self.sft_config

        # attn_implementation is consumed at model load time, not by SFTConfig.
        self.sft_params.pop("attn_implementation", None)

        # Newer TRL versions removed max_seq_length in favor of max_length.
        max_seq_length = int(self.sft_params.pop("max_seq_length", 1024))
        self.sft_params.setdefault("max_length", max_seq_length)

        self.sft_config = SFTConfig(**self.sft_params)
        self.sft_config._max_seq_length = max_seq_length
        return self.sft_config

    def _build_trainer(
        self,
        *,
        model,
        train_dataset,
        eval_dataset,
        args,
        tokenizer,
    ):
        trainer_cls = SFTTrainer
        trainer_extra_kwargs = {}

        if self.token_budget_config is not None:
            if self.config.use_deepspeed:
                raise NotImplementedError(
                    "Token-budget batching is currently implemented only for "
                    "the DDP/_train_with_transformers path."
                )
            trainer_cls = TokenBudgetSFTTrainer
            trainer_extra_kwargs["token_budget_config"] = self.token_budget_config

        trainer_kwargs = dict(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            processing_class=tokenizer,
        )
        if self.config.mask_prompt_loss:
            trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                label_pad_token_id=-100,
                return_tensors="pt",
            )

        trainer = trainer_cls(**trainer_kwargs, **trainer_extra_kwargs)
        trainer.remove_callback(PrinterCallback)
        return trainer

    def _find_latest_checkpoint_dir(self, output_dir: str) -> str | None:
        def _parse_checkpoint_dirname(dirname: str) -> int | None:
            if dirname == "checkpoint-last":
                return 10**18
            if not dirname.startswith("checkpoint-"):
                return None
            suffix = dirname.split("checkpoint-", 1)[-1]
            try:
                return int(suffix)
            except Exception:
                return None

        def _scan(base: pathlib.Path) -> list[tuple[int, float, pathlib.Path]]:
            out: list[tuple[int, float, pathlib.Path]] = []
            if not base.exists() or not base.is_dir():
                return out
            try:
                for child in base.iterdir():
                    if not child.is_dir():
                        continue
                    step = _parse_checkpoint_dirname(child.name)
                    if step is None:
                        continue
                    try:
                        mtime = child.stat().st_mtime
                    except Exception:
                        mtime = 0.0
                    out.append((step, mtime, child))
            except Exception:
                return out
            return out

        try:
            base = pathlib.Path(output_dir)
            candidates = _scan(base)

            if not candidates:
                root_outputs = pathlib.Path("./outputs")
                retry_dirs = []
                try:
                    retry_dirs = [
                        p for p in root_outputs.glob("retry_*") if p.is_dir()
                    ]
                except Exception:
                    retry_dirs = []

                try:
                    parents = list(base.parents)[:4]
                except Exception:
                    parents = []
                for parent in parents:
                    try:
                        retry_dirs.extend(
                            [p for p in parent.glob("retry_*") if p.is_dir()]
                        )
                    except Exception:
                        pass

                seen = set()
                unique_retry_dirs: list[pathlib.Path] = []
                for p in retry_dirs:
                    sp = str(p.resolve())
                    if sp in seen:
                        continue
                    seen.add(sp)
                    unique_retry_dirs.append(p)

                for retry_dir in unique_retry_dirs:
                    candidates.extend(_scan(retry_dir / "model-checkpoints"))
                    candidates.extend(
                        _scan(retry_dir / "outputs" / "model-checkpoints")
                    )
                    candidates.extend(_scan(retry_dir / "model" / "checkpoints"))

            if not candidates:
                return None

            candidates.sort(key=lambda t: (t[0], t[1]))
            return str(candidates[-1][2])
        except Exception as e:
            logger.warning(f"Failed to scan for checkpoints: {e}")
            return None

    def _resolve_resume_checkpoint(
        self, *, rank: int, world_size: int
    ) -> str | None:
        resume = (self.config.resume_from_checkpoint or "").strip()
        auto_resume = self.config.auto_resume
        if auto_resume is None:
            auto_resume = False

        explicit: str | None = None
        if resume:
            if resume.lower() in {"auto", "latest"}:
                auto_resume = True
            else:
                explicit = resume

        if explicit:
            return explicit
        if not auto_resume:
            return None

        output_dir = self.sft_params.get("output_dir") or "./outputs/model-checkpoints"
        sync_dir = pathlib.Path("./outputs")
        sync_dir.mkdir(parents=True, exist_ok=True)
        sync_file = sync_dir / "resume_checkpoint.txt"

        if world_size > 1:
            if rank == 0:
                ckpt = self._find_latest_checkpoint_dir(output_dir) or ""
                try:
                    sync_file.write_text(ckpt, encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Failed to write resume checkpoint file: {e}")
                return ckpt or None

            start = time.time()
            timeout = 300
            while not sync_file.exists():
                if time.time() - start > timeout:
                    logger.warning("Timed out waiting for resume checkpoint file")
                    return None
                time.sleep(2)
            try:
                ckpt = sync_file.read_text(encoding="utf-8").strip()
            except Exception:
                ckpt = ""
            return ckpt or None

        return self._find_latest_checkpoint_dir(output_dir)

    def _load_deepspeed_config(self) -> dict:
        """Load and customize DeepSpeed configuration."""
        config_path = pathlib.Path(__file__).parent / "deepspeed_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                ds_config = json.load(f)
        else:
            # Default DeepSpeed config if file not found
            ds_config = {
                "zero_optimization": {
                    "stage": self.config.deepspeed_stage,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 5e8,
                    "allgather_bucket_size": 5e8,
                },
                "bf16": {"enabled": True},
                "gradient_clipping": 1.0,
                "train_micro_batch_size_per_gpu": "auto",
                "train_batch_size": "auto",
                "gradient_accumulation_steps": "auto",
            }

        ds_config["zero_optimization"]["stage"] = self.config.deepspeed_stage
        if not self.config.deepspeed_save_full_model:
            ds_config["zero_optimization"][
                "stage3_gather_16bit_weights_on_model_save"
            ] = False

        if getattr(self.config, "deepspeed_cpu_offload", False):
            zero = ds_config.setdefault("zero_optimization", {})
            stage = int(zero.get("stage", self.config.deepspeed_stage or 2))

            offload_opt = zero.get("offload_optimizer") or {}
            offload_opt.update({"device": "cpu", "pin_memory": True})
            zero["offload_optimizer"] = offload_opt

            if stage >= 3:
                offload_param = zero.get("offload_param") or {}
                offload_param.update({"device": "cpu", "pin_memory": True})
                zero["offload_param"] = offload_param

        if self.config.deepspeed_config:
            ds_config.update(self.config.deepspeed_config)

        logger.info(
            f"DeepSpeed config: ZeRO Stage {ds_config['zero_optimization']['stage']}"
        )
        return ds_config

    def _train_with_deepspeed(self):
        """Training method using DeepSpeed with HuggingFace Trainer integration."""
        if deepspeed is None:
            raise ImportError(
                "DeepSpeed is not installed. Install with: pip install deepspeed"
            )

        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Enforce single-node: all communication on loopback
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        if torch.cuda.is_available() and local_rank is not None and local_rank >= 0:
            try:
                torch.cuda.set_device(local_rank)
            except Exception as e:
                logger.warning(f"Failed to set CUDA device for local_rank={local_rank}: {e}")

        if rank == 0:
            logger.info(
                f"DeepSpeed single-node multi-GPU training: {world_size} GPUs, RANK={rank}, LOCAL_RANK={local_rank}"
            )

        # Load dataset
        if rank == 0:
            logger.info(f"Loading dataset from {self.config.datapath}")
        dataset = load_dataset(
            "json", data_files=self.config.datapath, split="train"
        )

        dataset_size = len(dataset)
        if rank == 0:
            logger.info(f"Dataset loaded: {dataset_size} samples")
            log_memory("After dataset load")

        eval_size = min(5000, int(dataset_size * 0.05))
        dataset_split = dataset.train_test_split(test_size=eval_size, seed=42)

        if rank == 0:
            print(
                f"Training: {len(dataset_split['train'])}, Eval: {len(dataset_split['test'])}"
            )
            print(f"Model: {self.config.model}, Epochs: {self.config.epochs}")
            print(f"DeepSpeed ZeRO Stage: {self.config.deepspeed_stage}")

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._configure_chat_template(tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if self.config.mask_prompt_loss:
            tokenizer.truncation_side = "left"
            logger.info(
                "Masking prompt loss: using left truncation to preserve assistant targets"
            )
        sft_config = self._ensure_sft_config()
        max_len = self.sft_config._max_seq_length

        # Pre-tokenize dataset
        if rank == 0:
            logger.info("Pre-tokenizing dataset")

        cache_dir = "./outputs/tokenized_cache"
        os.makedirs(cache_dir, exist_ok=True)

        tokenize_function = _build_tokenize_fn(
            tokenizer,
            max_len=max_len,
            mask_prompt_loss=self.config.mask_prompt_loss,
        )

        if rank == 0:
            try:
                cpu_total = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 8
            except Exception:
                cpu_total = 8
            try:
                util_target = float(os.environ.get("FC_TOKENIZE_CPU_UTIL_TARGET", "0.85"))
            except Exception:
                util_target = 0.85
            util_target = min(max(util_target, 0.5), 0.95)
            target_workers = int(max(1, min(int(cpu_total * util_target), cpu_total - 2)))
            if self.tokenization_num_proc is not None:
                cpu_workers = max(1, int(self.tokenization_num_proc))
            else:
                cpu_workers = max(4, min(target_workers, 32))
            eval_workers = max(2, cpu_workers // 2)
            logger.info(
                "Rank 0: Tokenizing with num_proc=%d, batch_size=%d, "
                "writer_batch_size=%d",
                cpu_workers,
                self.tokenization_batch_size,
                self.tokenization_writer_batch_size,
            )
            os.environ["DATASETS_MP_START_METHOD"] = os.environ.get(
                "DATASETS_MP_START_METHOD", "spawn"
            )

            def _map_with_fallback(ds, num_proc_target, cache_name, desc):
                tries = []
                for candidate in (num_proc_target, max(1, num_proc_target // 2), 1):
                    if candidate not in tries:
                        tries.append(candidate)
                last_err = None
                for np in tries:
                    try:
                        return ds.map(
                            tokenize_function,
                            batched=True,
                            batch_size=self.tokenization_batch_size,
                            num_proc=np,
                            remove_columns=ds.column_names,
                            desc=desc,
                            cache_file_name=f"{cache_dir}/{cache_name}",
                            writer_batch_size=self.tokenization_writer_batch_size,
                            keep_in_memory=False,
                        )
                    except RuntimeError as e:
                        last_err = e
                        if "abruptly died during map operation" in str(e):
                            logger.warning(f"map failed with num_proc={np}; retrying")
                            continue
                        else:
                            raise
                raise last_err

            train_dataset = _map_with_fallback(
                dataset_split["train"],
                cpu_workers,
                "train_tokenized.arrow",
                "Tokenizing training data",
            )
            eval_dataset = _map_with_fallback(
                dataset_split["test"],
                eval_workers,
                "eval_tokenized.arrow",
                "Tokenizing eval data",
            )
            try:
                flag_path = os.path.join(cache_dir, "tokenization_done.flag")
                with open(flag_path, "w") as f:
                    f.write(
                        json.dumps(
                            {
                                "train_samples": len(train_dataset),
                                "eval_samples": len(eval_dataset),
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                            }
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to write tokenization flag: {e}")

        if world_size > 1 and rank != 0:
            flag_path = os.path.join(cache_dir, "tokenization_done.flag")
            start = time.time()
            timeout = self.tokenization_wait_timeout
            logger.info("Rank %d: waiting up to %ds for tokenization", rank, timeout)
            while not os.path.exists(flag_path):
                if time.time() - start > timeout:
                    raise TimeoutError("Timed out waiting for tokenization")
                time.sleep(5)

        if rank != 0:
            train_dataset = dataset_split["train"].map(
                tokenize_function,
                batched=True,
                batch_size=self.tokenization_batch_size,
                num_proc=1,
                remove_columns=dataset_split["train"].column_names,
                cache_file_name=f"{cache_dir}/train_tokenized.arrow",
                writer_batch_size=self.tokenization_writer_batch_size,
                load_from_cache_file=True,
            )
            eval_dataset = dataset_split["test"].map(
                tokenize_function,
                batched=True,
                batch_size=self.tokenization_batch_size,
                num_proc=1,
                remove_columns=dataset_split["test"].column_names,
                cache_file_name=f"{cache_dir}/eval_tokenized.arrow",
                writer_batch_size=self.tokenization_writer_batch_size,
                load_from_cache_file=True,
            )

        train_dataset = _log_and_filter_supervision(train_dataset, "train")
        eval_dataset = _log_and_filter_supervision(eval_dataset, "eval")

        if rank == 0:
            log_memory("After tokenization")

        # Load model
        if rank == 0:
            logger.info("Loading model for DeepSpeed training")

        requested_attn_impl = None
        try:
            requested_attn_impl = self.config.sft_kwargs.get("attn_implementation")
        except Exception:
            pass

        if "attn_implementation" in self.sft_params:
            self.sft_params.pop("attn_implementation", None)

        if "phi" in self.model_family_hint.lower():
            config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
            if hasattr(config, "rope_scaling") and config.rope_scaling:
                if "short_factor" in config.rope_scaling:
                    short_factor = config.rope_scaling["short_factor"]
                    if len(short_factor) == 48:
                        config.rope_scaling["short_factor"] = (
                            short_factor + short_factor[:16]
                        )
            if requested_attn_impl:
                config.attn_implementation = str(requested_attn_impl)
            else:
                config.attn_implementation = "eager"
            load_kwargs = self._build_model_load_kwargs()
            load_kwargs["config"] = config
            model = self._load_causal_lm(**load_kwargs)
        else:
            config = None
            if requested_attn_impl:
                try:
                    config = AutoConfig.from_pretrained(
                        self.model_path, trust_remote_code=self.trust_remote_code
                    )
                    config.attn_implementation = str(requested_attn_impl)
                except Exception:
                    config = None
            load_kwargs = self._build_model_load_kwargs()
            load_kwargs["config"] = config
            model = self._load_causal_lm(**load_kwargs)

        # Apply LoRA if enabled
        model = self._apply_lora(model)

        requested_gradient_checkpointing = bool(
            self.sft_params.get("gradient_checkpointing", True)
        )
        try:
            if hasattr(model, "config"):
                model.config.use_cache = not requested_gradient_checkpointing
            if requested_gradient_checkpointing:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
            elif hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
        except Exception as e:
            logger.warning(
                "Failed to configure gradient checkpointing "
                f"(requested={requested_gradient_checkpointing}): {e}"
            )

        try:
            num_workers = self.sft_params.get("dataloader_num_workers")
            prefetch = self.sft_params.get("dataloader_prefetch_factor")
            if prefetch is not None and (
                num_workers is None or int(num_workers) <= 0
            ):
                self.sft_params["dataloader_prefetch_factor"] = None
        except Exception:
            pass

        # Synchronize ranks before training
        ready_dir = os.path.join("./outputs", "ds_ready")
        os.makedirs(ready_dir, exist_ok=True)
        ready_file = os.path.join(ready_dir, f"rank_{rank}_ready")
        try:
            with open(ready_file, "w") as f:
                f.write(datetime.utcnow().isoformat() + "Z")
        except Exception:
            pass

        if world_size > 1:
            start_sync = time.time()
            timeout_sync = 1800
            while True:
                try:
                    files = glob.glob(os.path.join(ready_dir, "rank_*_ready"))
                except Exception:
                    files = []
                if len(files) >= world_size:
                    break
                if time.time() - start_sync > timeout_sync:
                    raise TimeoutError(
                        "Timed out waiting for all ranks before DS init"
                    )
                time.sleep(2)

        if rank == 0:
            log_memory("Before training")

        try:
            if rank == 0:
                logger.info(
                    f"Rendezvous: MASTER_ADDR={os.environ.get('MASTER_ADDR', '')}, "
                    f"MASTER_PORT={os.environ.get('MASTER_PORT', '')}"
                )
            trainer = self._build_trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=sft_config,
                tokenizer=tokenizer,
            )
        except Exception as e:
            msg = str(e)
            if any(
                tok in msg
                for tok in [
                    "EADDRINUSE",
                    "address already in use",
                    "server socket has failed",
                ]
            ):
                raise RuntimeError(
                    "Distributed rendezvous port in use. "
                    f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
                    f"MASTER_PORT={os.environ.get('MASTER_PORT')}."
                ) from e
            raise

        resume_ckpt = self._resolve_resume_checkpoint(
            rank=rank, world_size=world_size
        )
        if resume_ckpt:
            if rank == 0:
                logger.info(f"Resuming from checkpoint: {resume_ckpt}")
            train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            train_result = trainer.train()

        # All ranks must participate in save for ZeRO-3 weight gathering
        print(f"Training completed with DeepSpeed. (rank {rank})")
        output_dir = self.sft_params.get("output_dir", "./outputs/model-checkpoints")
        self._merge_and_save(trainer, tokenizer, output_dir)
        if rank == 0:
            persist_training_artifacts(
                trainer=trainer,
                output_dir=output_dir,
                logging_dir=self.sft_params.get("logging_dir"),
                train_metrics=getattr(train_result, "metrics", None),
                run_config=self.config.model_dump(mode="json"),
            )

    def start_training(self):
        """Route to appropriate training method."""
        if self.config.use_deepspeed:
            logger.info("Using DeepSpeed training")
            self._train_with_deepspeed()
        else:
            logger.info("Using HuggingFace Transformers training")
            self._train_with_transformers()

    def _apply_lora(self, model):
        """Wrap model with LoRA adapters if enabled."""
        if not self.config.use_lora:
            logger.info("LoRA disabled, training all parameters (full SFT)")
            return model

        if not _PEFT_AVAILABLE:
            raise ImportError(
                "LoRA training requires `peft`. Install with: pip install peft"
            )

        if bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
            getattr(model, "is_loaded_in_8bit", False)
        ):
            use_gradient_checkpointing = bool(
                self.sft_params.get("gradient_checkpointing", True)
            )
            try:
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                )
            except TypeError:
                model = prepare_model_for_kbit_training(model)
            logger.info("Prepared quantized model for k-bit LoRA training")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if bool(self.sft_params.get("gradient_checkpointing", True)):
            try:
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                elif hasattr(model, "get_input_embeddings"):
                    input_embeddings = model.get_input_embeddings()
                    if input_embeddings is not None and not getattr(
                        input_embeddings, "_fc_require_grads_hook", False
                    ):
                        def _require_grads(_module, _inputs, output):
                            if isinstance(output, torch.Tensor):
                                output.requires_grad_(True)
                            elif isinstance(output, (tuple, list)):
                                for value in output:
                                    if isinstance(value, torch.Tensor):
                                        value.requires_grad_(True)

                        input_embeddings.register_forward_hook(_require_grads)
                        input_embeddings._fc_require_grads_hook = True
            except Exception as e:
                logger.warning(
                    "Failed to enable PEFT input grads for gradient checkpointing: "
                    f"{e}"
                )

        if self.is_main_process:
            model.print_trainable_parameters()
            logger.info(
                f"LoRA applied: r={self.config.lora_r}, alpha={self.config.lora_alpha}, "
                f"dropout={self.config.lora_dropout}, targets={self.config.lora_target_modules}"
            )

        return model

    def _merge_and_save(self, trainer, tokenizer, output_dir: str):
        """Merge LoRA weights into base model and save as standard HF model."""
        dist_ready = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.world_size > 1
        )
        if dist_ready:
            torch.distributed.barrier()

        if self.config.use_lora and self.config.save_lora_adapter_only:
            if self.is_main_process:
                logger.info("Saving LoRA adapter only (skipping merge)...")
                trainer.model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"LoRA adapter saved to {output_dir}")
        elif self.config.use_lora:
            # Under DeepSpeed ZeRO-3 the base model params are sharded across
            # ranks (rank 0's slice has shape (0, ...)), so merge_and_unload()
            # tries to add a full delta into an empty tensor and crashes. We
            # must gather all params (collective: every rank participates)
            # before merging, then save on rank 0 only.
            if self.config.use_deepspeed:
                logger.info(
                    f"Gathering ZeRO-3 params for LoRA merge (rank {self.local_rank})..."
                )
                import deepspeed
                params_to_gather = list(trainer.model.parameters())
                with deepspeed.zero.GatheredParameters(
                    params_to_gather, modifier_rank=0
                ):
                    if self.is_main_process:
                        logger.info("Merging LoRA weights into base model...")
                        merged_model = trainer.model.merge_and_unload()
                        merged_model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info(f"Merged model saved to {output_dir}")
            elif self.is_main_process:
                logger.info("Merging LoRA weights into base model...")
                merged_model = trainer.model.merge_and_unload()
                merged_model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Merged model saved to {output_dir}")
        elif not self.config.use_deepspeed:
            if self.is_main_process:
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Model saved to {output_dir}")
        else:
            logger.info(
                f"Skipping merge/save on worker process (rank {self.local_rank})"
            )

        if not self.config.use_lora and self.config.use_deepspeed:
            if not self.config.deepspeed_save_full_model:
                checkpoint_root = pathlib.Path(output_dir) / "deepspeed-sharded-checkpoint"
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                tag = f"global_step{getattr(trainer.state, 'global_step', 0) or 'final'}"
                engine = getattr(trainer, "model_wrapped", None) or getattr(
                    trainer, "deepspeed", None
                )
                if engine is None or not hasattr(engine, "save_checkpoint"):
                    raise RuntimeError(
                        "DeepSpeed sharded save requested, but no DeepSpeed "
                        "engine with save_checkpoint() is available."
                    )

                logger.info(
                    "Saving DeepSpeed ZeRO checkpoint shards to %s with tag %s",
                    checkpoint_root,
                    tag,
                )
                engine.save_checkpoint(
                    str(checkpoint_root),
                    tag=tag,
                    client_state={
                        "global_step": getattr(trainer.state, "global_step", None),
                        "save_mode": "deepspeed_sharded",
                        "requires_offline_consolidation": True,
                    },
                )
                if self.is_main_process:
                    model_config = getattr(getattr(trainer, "model", None), "config", None)
                    if model_config is not None:
                        model_config.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    metadata_path = pathlib.Path(output_dir) / "deepspeed_sharded_checkpoint.json"
                    metadata_path.write_text(
                        json.dumps(
                            {
                                "checkpoint_dir": str(checkpoint_root),
                                "tag": tag,
                                "save_mode": "deepspeed_sharded",
                                "requires_offline_consolidation": True,
                                "note": (
                                    "This artifact intentionally avoids ZeRO-3 "
                                    "full-model consolidation during training. "
                                    "Run DeepSpeed zero_to_fp32 consolidation "
                                    "as a separate post-training step before "
                                    "registering for inference."
                                ),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    logger.info("DeepSpeed sharded checkpoint metadata saved")
                if dist_ready:
                    torch.distributed.barrier()
                return

            # ZeRO-3 16-bit consolidation is a collective: every rank must enter
            # trainer.save_model(...) so rank 0 can gather full weights.
            trainer.save_model(output_dir)
            if self.is_main_process:
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Model saved to {output_dir}")

        if dist_ready:
            torch.distributed.barrier()

    def _train_with_transformers(self):
        """Training method using HuggingFace SFTTrainer."""
        self._ensure_sft_config()

        logger.info(f"Loading dataset from {self.config.datapath}")
        dataset = load_dataset(
            "json", data_files=self.config.datapath, split="train"
        )

        dataset_size = len(dataset)
        logger.info(f"Dataset loaded: {dataset_size} samples")
        log_memory("After dataset load")

        eval_size = min(5000, int(dataset_size * 0.05))
        dataset_split = dataset.train_test_split(test_size=eval_size, seed=42)

        print(
            f"Training: {len(dataset_split['train'])}, Eval: {len(dataset_split['test'])}"
        )
        print(f"Model: {self.config.model}, Epochs: {self.config.epochs}")

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"GPUs available: {num_gpus}")
            if self.world_size > 1:
                print(f"Single-node multi-GPU: Using {self.world_size} GPUs with DDP")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._configure_chat_template(tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if self.config.mask_prompt_loss:
            tokenizer.truncation_side = "left"
            logger.info(
                "Masking prompt loss: using left truncation to preserve assistant targets"
            )

        logger.info("Pre-tokenizing dataset")
        cache_dir = "./outputs/tokenized_cache"
        os.makedirs(cache_dir, exist_ok=True)

        tokenize_function = _build_tokenize_fn(
            tokenizer,
            max_len=self.sft_config._max_seq_length,
            mask_prompt_loss=self.config.mask_prompt_loss,
        )

        if self.is_main_process:
            logger.info("Main process: Tokenizing and caching datasets")
            train_dataset = dataset_split["train"].map(
                tokenize_function,
                batched=True,
                batch_size=50000,
                num_proc=4,
                remove_columns=dataset_split["train"].column_names,
                desc="Tokenizing train dataset",
                cache_file_name=f"{cache_dir}/train_tokenized.arrow",
                writer_batch_size=50000,
                keep_in_memory=False,
            )
            eval_dataset = dataset_split["test"].map(
                tokenize_function,
                batched=True,
                batch_size=50000,
                remove_columns=dataset_split["test"].column_names,
                desc="Tokenizing eval dataset",
                cache_file_name=f"{cache_dir}/eval_tokenized.arrow",
                writer_batch_size=50000,
            )

        if self.world_size > 1:
            torch.distributed.barrier()
            if not self.is_main_process:
                train_dataset = dataset_split["train"].map(
                    tokenize_function,
                    batched=True,
                    batch_size=1000,
                    remove_columns=dataset_split["train"].column_names,
                    cache_file_name=f"{cache_dir}/train_tokenized.arrow",
                    load_from_cache_file=True,
                )
                eval_dataset = dataset_split["test"].map(
                    tokenize_function,
                    batched=True,
                    batch_size=1000,
                    remove_columns=dataset_split["test"].column_names,
                    cache_file_name=f"{cache_dir}/eval_tokenized.arrow",
                    load_from_cache_file=True,
                )
        else:
            train_dataset = dataset_split["train"].map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                remove_columns=dataset_split["train"].column_names,
                num_proc=self.sft_config.dataset_num_proc,
                desc="Tokenizing train dataset",
            )
            eval_dataset = dataset_split["test"].map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                remove_columns=dataset_split["test"].column_names,
                num_proc=self.sft_config.dataset_num_proc,
                desc="Tokenizing eval dataset",
            )

        train_dataset = _log_and_filter_supervision(train_dataset, "train")
        eval_dataset = _log_and_filter_supervision(eval_dataset, "eval")

        log_memory("After tokenization")

        if self.world_size > 1:
            logger.info("Loading model for DDP training")
            if "phi" in self.model_family_hint.lower():
                config = AutoConfig.from_pretrained(
                    self.model_path, trust_remote_code=self.trust_remote_code
                )
                if hasattr(config, "rope_scaling") and config.rope_scaling:
                    if "short_factor" in config.rope_scaling:
                        short_factor = config.rope_scaling["short_factor"]
                        if len(short_factor) == 48:
                            config.rope_scaling["short_factor"] = (
                                short_factor + short_factor[:16]
                            )
                config.attn_implementation = "eager"
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, config=config, torch_dtype=torch.bfloat16
                )
            else:
                requested_attn_impl = self.config.sft_kwargs.get("attn_implementation")
                load_kwargs = self._build_model_load_kwargs(requested_attn_impl)
                model = self._load_causal_lm(**load_kwargs)
        else:
            if self.config.use_lora:
                # LoRA requires an explicit model object, not a path string
                requested_attn_impl = self.config.sft_kwargs.get("attn_implementation")
                load_kwargs = self._build_model_load_kwargs(requested_attn_impl)
                model = self._load_causal_lm(**load_kwargs)
            else:
                model = self.model_path

        # Apply LoRA if enabled (wraps model with trainable adapters)
        if self.config.use_lora and not isinstance(model, str):
            model = self._apply_lora(model)

        log_memory("Before training")
        trainer = self._build_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=self.sft_config,
            tokenizer=tokenizer,
        )

        rank = int(
            os.environ.get(
                "RANK", 0 if self.is_main_process else self.local_rank
            )
        )
        world_size = int(os.environ.get("WORLD_SIZE", self.world_size))
        resume_ckpt = self._resolve_resume_checkpoint(
            rank=rank, world_size=world_size
        )
        if resume_ckpt:
            if self.is_main_process:
                logger.info(f"Resuming from checkpoint: {resume_ckpt}")
            train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            train_result = trainer.train()
        print("Training completed.")

        self._merge_and_save(trainer, tokenizer, self.sft_config.output_dir)
        if self.is_main_process:
            persist_training_artifacts(
                trainer=trainer,
                output_dir=self.sft_config.output_dir,
                logging_dir=self.sft_config.logging_dir,
                train_metrics=getattr(train_result, "metrics", None),
                run_config=self.config.model_dump(mode="json"),
            )


def train():
    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default="./.model_cache",
        help="Local cache directory for fine-tuned models.",
    )
    parser.add_argument(
        "--datapath", type=str, required=True, help="Path to training dataset."
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs."
    )
    parser.add_argument(
        "--eval_strategy", type=str, default="steps", help="Evaluation strategy."
    )
    parser.add_argument(
        "--eval_steps", type=float, default=0.1, help="Evaluation steps."
    )
    parser.add_argument(
        "--save_strategy", type=str, default="no", help="Model saving strategy."
    )
    parser.add_argument(
        "--save_steps", type=float, default=None, help="Checkpoint save steps."
    )
    parser.add_argument(
        "--ft_model_name", type=str, default="", help="Fine-tuned model name."
    )
    parser.add_argument(
        "--sft_kwargs_base64",
        type=str,
        default="",
        help="Base64-encoded JSON of additional SFT config parameters.",
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Enable DeepSpeed training."
    )
    parser.add_argument(
        "--deepspeed_stage", type=int, default=2, help="DeepSpeed ZeRO stage."
    )
    parser.add_argument(
        "--deepspeed_cpu_offload",
        action="store_true",
        help="Enable DeepSpeed ZeRO CPU offload.",
    )
    parser.add_argument(
        "--deepspeed_save_full_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For DeepSpeed full-SFT, gather and save a consolidated HF model "
            "at the end. Disable to save ZeRO checkpoint shards for offline "
            "consolidation."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Checkpoint path or 'auto'/'latest'.",
    )
    parser.add_argument(
        "--auto_resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto-resume from latest checkpoint if found.",
    )
    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA training (default: True). Use the generated --no- form to disable.",
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA rank (default: 16)."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)."
    )
    parser.add_argument(
        "--save_lora_adapter_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save only LoRA adapter weights instead of merging into the base model.",
    )
    parser.add_argument(
        "--mask_prompt_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mask loss on prompt tokens so only the assistant completion "
            "contributes to training loss (default: True). Use the generated "
            "--no- form to revert to pre-fix behaviour (loss "
            "over entire chat, including the prompt)."
        ),
    )

    args = parser.parse_args()

    sft_kwargs = {}
    if args.sft_kwargs_base64:
        try:
            sft_kwargs_json = base64.b64decode(
                args.sft_kwargs_base64.encode("ascii")
            ).decode("utf-8")
            sft_kwargs = json.loads(sft_kwargs_json)
            logger.info(f"Decoded SFT kwargs: {sft_kwargs}")
        except Exception as e:
            logger.error(f"Failed to decode sft_kwargs_base64: {e}")
            raise

    config = TrainerConfig(
        model=args.model,
        model_cache_dir=args.model_cache_dir,
        datapath=args.datapath,
        epochs=args.epochs,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        ft_model_name=args.ft_model_name,
        sft_kwargs=sft_kwargs,
        use_deepspeed=args.use_deepspeed,
        deepspeed_stage=args.deepspeed_stage,
        deepspeed_cpu_offload=bool(args.deepspeed_cpu_offload),
        deepspeed_save_full_model=bool(args.deepspeed_save_full_model),
        resume_from_checkpoint=(args.resume_from_checkpoint or None),
        auto_resume=args.auto_resume,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_lora_adapter_only=args.save_lora_adapter_only,
        mask_prompt_loss=args.mask_prompt_loss,
    )

    trainer = Trainer(config)
    trainer.start_training()


if __name__ == "__main__":
    train()
