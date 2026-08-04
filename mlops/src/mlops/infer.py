import torch
import time
import logging
import os
from contextlib import nullcontext
from pydantic import BaseModel, computed_field
from typing import Optional, List, Union, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

logger = logging.getLogger(__name__)

# Optional DeepSpeed import
try:
    import deepspeed
except ImportError:
    deepspeed = None


class InferenceConfig(BaseModel):
    """Configuration for inference.

    NOTE: Only single-node, multi-GPU is supported. All GPUs must be on the
    same machine. Multi-node distributed inference is not supported.
    """

    max_tokens: int = 512
    max_out_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    num_return_sequences: int = 1
    num_beams: int = 1
    num_beam_groups: int = 1
    do_sample: bool = False
    seed: Optional[int] = None
    prefix: Optional[str] = None
    use_deepspeed: bool = True
    deepspeed_config: Optional[Dict] = None
    generation_autocast: bool = False
    generation_autocast_dtype: Optional[str] = None


class ModelConfig(BaseModel):
    model: str
    device: str = "auto"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    padding_side: str = "left"
    truncation: bool = True
    attn_implementation: Optional[str] = None
    tensor_parallel_size: int = 1
    # Adapter support: when adapter_path is set, `model` is the BASE model
    # path/name and the adapter is applied on top via PEFT.
    adapter_path: Optional[str] = None
    base_model: Optional[str] = None  # informational: the base the adapter targets


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class InferenceResult(BaseModel):
    id: Optional[str] = None
    prompt: Union[str, List[Dict[str, str]]]
    predictions: List[str]
    usage: Usage
    latency_seconds: Optional[float] = None
    status: str = "ok"  # "ok" or "error"
    error: Optional[str] = None  # Error message if status == "error"


class BatchInferenceResult(BaseModel):
    results: List[InferenceResult]
    total_latency_seconds: Optional[float] = None
    batch_size: int


class StopOnTokenSequences(StoppingCriteria):
    """Stop generation when any of the provided stop tokens appears."""

    def __init__(self, stop_token_ids: List[int]):
        super().__init__()
        self.stop_token_ids = set(stop_token_ids)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if not self.stop_token_ids:
            return False
        for b in range(input_ids.shape[0]):
            if input_ids[b, -1].item() in self.stop_token_ids:
                return True
        return False

    @classmethod
    def get(cls, tokenizer: AutoTokenizer) -> "StopOnTokenSequences":
        return cls(cls.get_stop_token_ids(tokenizer))

    @staticmethod
    def get_stop_token_ids(tokenizer: AutoTokenizer) -> List[int]:
        """Get stop token IDs for the tokenizer."""
        stop_token_ids: List[int] = list()
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            possible_end_tokens = [
                "<|im_end|>",
                "<|end|>",
                "<|assistant|>",
                "<|endoftext|>",
            ]
            for token in possible_end_tokens:
                if token in tokenizer.get_vocab():
                    token_ids = tokenizer.encode(token, add_special_tokens=False)
                    if len(token_ids) == 1:
                        stop_token_ids.append(token_ids[0])
                        break
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)
        return stop_token_ids


class InferenceEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: Optional[InferenceConfig] = None,
    ) -> None:
        self.model_config = model_config
        self.inference_config = inference_config or InferenceConfig()
        self._inference_dtype = torch.bfloat16
        self._load_model()

    def _get_default_deepspeed_config(self) -> Dict:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.model_config.dtype.lower(), torch.bfloat16)
        if self.inference_config.max_out_tokens:
            max_total_tokens = self.inference_config.max_out_tokens
        else:
            max_total_tokens = 32768
        config = {
            "tensor_parallel": {"tp_size": num_gpus},
            "dtype": dtype,
            "replace_with_kernel_inject": True,
            "enable_cuda_graph": False,
            "max_out_tokens": max_total_tokens,
            "min_out_tokens": 1,
        }
        return config

    def _get_device_map(self) -> Union[str, Dict[str, int], None]:
        """Automatically determine optimal device mapping based on available GPUs."""
        num_available_gpus = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        if self.inference_config.use_deepspeed:
            logger.info(f"Deepspeed enabled with {num_available_gpus} GPUs")
            return None

        if num_available_gpus > 1:
            logger.info(
                f"Auto-detected {num_available_gpus} GPUs: Using balanced distribution"
            )
            return "balanced"
        elif num_available_gpus == 1:
            logger.info("Single GPU detected: cuda:0")
            return "cuda:0"
        else:
            logger.warning("No CUDA GPUs detected, falling back to CPU")
            return "cpu"

    def _log_device_placement(self) -> None:
        """Log information about where model components are placed."""
        if self.inference_config.use_deepspeed:
            logger.info("DeepSpeed inference engine initialized.")
            if hasattr(self._model, "module"):
                devices = set()
                for param in self._model.module.parameters():
                    devices.add(param.device)
                logger.info(
                    f"Model distributed across devices: {sorted([str(d) for d in devices])}"
                )
            return

        devices = set()
        for param in self._model.parameters():
            devices.add(param.device)

        if len(devices) == 1:
            device = list(devices)[0]
            logger.info(f"Model loaded on single device: {device}")
        else:
            logger.info(
                f"Model distributed across {len(devices)} devices: {sorted([str(d) for d in devices])}"
            )
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    logger.info(
                        f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
                    )

    def _enable_generation_cache(self, model) -> None:
        """Training checkpoints may persist use_cache=False; inference needs KV cache."""
        config = getattr(model, "config", None)
        if config is not None and getattr(config, "use_cache", None) is False:
            logger.info("Model config has use_cache=False; enabling KV cache for inference.")
            config.use_cache = True

        generation_config = getattr(model, "generation_config", None)
        if (
            generation_config is not None
            and getattr(generation_config, "use_cache", None) is False
        ):
            logger.info(
                "Generation config has use_cache=False; enabling KV cache for inference."
            )
            generation_config.use_cache = True

    def _requires_eager_attention(self) -> bool:
        model_names = [
            self.model_config.model,
            self.model_config.base_model or "",
        ]
        return any(
            token in model_name.lower()
            for model_name in model_names
            for token in ("gpt-oss", "gpt_oss")
        )

    def _load_model(self) -> None:
        is_adapter = bool(self.model_config.adapter_path)
        if is_adapter:
            # Load tokenizer from base model — adapter dirs saved by GRPO trainer
            # often omit chat_template (saved tokenizer_config.json strips it).
            tokenizer_source = self.model_config.model
            logger.info(
                f"Loading adapter: {self.model_config.adapter_path} "
                f"on base: {self.model_config.model} (tokenizer from base)"
            )
        else:
            tokenizer_source = self.model_config.model
            logger.info(f"Loading model: {self.model_config.model}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            logger.warning("Tokenizer has no pad_token, using eos_token.")

        self._tokenizer.padding_side = self.model_config.padding_side

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.model_config.dtype.lower(), torch.bfloat16)
        self._inference_dtype = dtype

        if is_adapter and self.inference_config.use_deepspeed:
            logger.warning(
                "DeepSpeed kernel injection is incompatible with PEFT adapters. "
                "Forcing transformers loading path for adapter inference."
            )
            self.inference_config.use_deepspeed = False

        if (
            self.inference_config.use_deepspeed
            and deepspeed is not None
            and not is_adapter
        ):
            try:
                self._load_model_with_deepspeed(dtype)
            except AttributeError as e:
                if "num_heads" not in str(e):
                    raise
                logger.warning(
                    "DeepSpeed kernel injection is incompatible with this model "
                    f"implementation ({e}). Falling back to transformers inference."
                )
                self.inference_config.use_deepspeed = False
                self._load_model_with_transformers(dtype)
        else:
            if (
                self.inference_config.use_deepspeed
                and deepspeed is None
                and not is_adapter
            ):
                logger.warning(
                    "DeepSpeed requested but not installed. Falling back to transformers."
                )
            self._load_model_with_transformers(dtype)

        if self.inference_config.seed is not None:
            self._set_seed(self.inference_config.seed)
            logger.info(f"Random seed set to: {self.inference_config.seed}")

    def _load_model_with_transformers(self, dtype: torch.dtype) -> None:
        device_map = self._get_device_map()
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_config.model,
            "torch_dtype": dtype,
            "trust_remote_code": self.model_config.trust_remote_code,
            "device_map": device_map,
        }

        if self._requires_eager_attention():
            load_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention for GPT-OSS model")
        elif "phi" in self.model_config.model.lower():
            load_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention for Phi model")
        elif self.model_config.attn_implementation:
            load_kwargs["attn_implementation"] = self.model_config.attn_implementation

        self._model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        if self.model_config.adapter_path:
            from peft import PeftModel

            logger.info(
                f"Applying PEFT adapter from {self.model_config.adapter_path}"
            )
            self._model = PeftModel.from_pretrained(
                self._model, self.model_config.adapter_path
            )
            logger.info(
                f"Loaded adapter on top of base {self.model_config.model}"
            )

        self._enable_generation_cache(self._model)
        self._model.eval()
        self._log_device_placement()

    def _load_model_with_deepspeed(self, dtype: torch.dtype) -> None:
        logger.info("Initializing DeepSpeed inference engine...")

        ds_config = (
            self.inference_config.deepspeed_config
            or self._get_default_deepspeed_config()
        )
        tp_size = ds_config["tensor_parallel"]["tp_size"]

        is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

        if tp_size > 1 and not is_distributed:
            logger.warning(
                f"DeepSpeed TP requires launcher. Falling back to transformers device_map for {tp_size} GPUs."
            )
            # Must flip the flag BEFORE loading -- otherwise _get_device_map()
            # returns None (model lands on CPU) while _get_primary_device()
            # still returns cuda:current_device(), and every batch fails with
            # "Expected all tensors to be on the same device". Matches the
            # pattern in the adapter and DeepSpeed-exception fallbacks above.
            self.inference_config.use_deepspeed = False
            self._load_model_with_transformers(dtype)
            return

        # Single-node only: force loopback for all communication
        if not is_distributed:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            logger.info("Single-node DeepSpeed: set loopback env vars")
        else:
            # Even when launched with a distributed launcher, enforce single-node
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        load_kwargs = {
            "pretrained_model_name_or_path": self.model_config.model,
            "torch_dtype": dtype,
            "trust_remote_code": self.model_config.trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        if self._requires_eager_attention():
            load_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention for GPT-OSS model")
        elif "phi" in self.model_config.model.lower():
            load_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention for Phi model")
        elif self.model_config.attn_implementation:
            load_kwargs["attn_implementation"] = self.model_config.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self._enable_generation_cache(model)
        model.eval()

        self._model = deepspeed.init_inference(
            model,
            mp_size=1 if not is_distributed else tp_size,
            dtype=ds_config["dtype"],
            replace_with_kernel_inject=ds_config.get("replace_with_kernel_inject", True),
            enable_cuda_graph=ds_config.get("enable_cuda_graph", False),
            max_out_tokens=ds_config.get("max_out_tokens", 4096),
            min_out_tokens=ds_config.get("min_out_tokens", 1),
        )

        self._log_device_placement()

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_primary_device(self) -> torch.device:
        """Get the primary device for input tensors."""
        if self.inference_config.use_deepspeed and deepspeed is not None:
            if torch.cuda.is_available():
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            return torch.device("cpu")
        return next(self._model.parameters()).device

    def _build_generation_kwargs(self) -> Dict:
        config = self.inference_config
        gen_kwargs = {
            "max_new_tokens": config.max_tokens,
            "do_sample": config.do_sample,
            "num_beams": config.num_beams,
            "num_beam_groups": config.num_beam_groups,
            "num_return_sequences": config.num_return_sequences,
            "pad_token_id": self._tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
        }

        if config.do_sample:
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p
            if config.top_k is not None:
                gen_kwargs["top_k"] = config.top_k

        stop_token_ids = StopOnTokenSequences.get_stop_token_ids(self._tokenizer)
        if stop_token_ids:
            gen_kwargs["eos_token_id"] = stop_token_ids

        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        return gen_kwargs

    def _generation_autocast(self):
        if not self.inference_config.generation_autocast:
            return nullcontext()

        dtype = self._inference_dtype
        if self.inference_config.generation_autocast_dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            dtype = dtype_map.get(
                self.inference_config.generation_autocast_dtype.lower(), dtype
            )

        if dtype not in {torch.bfloat16, torch.float16}:
            return nullcontext()

        primary_device = self._get_primary_device()
        if primary_device.type != "cuda":
            return nullcontext()

        return torch.autocast(device_type="cuda", dtype=dtype)

    def infer(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        inference_id: Optional[str] = None,
    ) -> InferenceResult:
        if isinstance(prompt, list):
            prefix_forcing = (
                prompt[-1]["role"].lower() == "assistant" if prompt else False
            )
            formatted_prompt = self._tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=not prefix_forcing,
                continue_final_message=prefix_forcing,
            )
        else:
            formatted_prompt = prompt

        inputs = self._tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=self.model_config.truncation,
        ).to(self._get_primary_device())

        prompt_tokens = inputs.input_ids.shape[1]

        gen_kwargs = self._build_generation_kwargs()
        start_time = time.time()
        with torch.inference_mode(), self._generation_autocast():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        latency = time.time() - start_time
        generated_ids = outputs.sequences[:, prompt_tokens:]
        completion_tokens = (
            generated_ids.numel() // self.inference_config.num_return_sequences
        )

        predictions = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        predictions = [pred.strip() for pred in predictions]

        result = InferenceResult(
            id=inference_id,
            prompt=formatted_prompt if isinstance(prompt, str) else prompt,
            predictions=predictions,
            usage=Usage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
            latency_seconds=latency,
        )

        logger.info(
            f"Inference complete: {prompt_tokens} prompt tokens, "
            f"{completion_tokens} completion tokens, {latency:.2f}s"
        )
        return result

    def _make_error_result(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        example_id: str,
        error: Exception,
        prompt_tokens: int = 0,
    ) -> InferenceResult:
        """Create a failed InferenceResult that records the error for analysis."""
        error_type = type(error).__name__
        error_msg = str(error)
        # Truncate very long error messages
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return InferenceResult(
            id=example_id,
            prompt=prompt,
            predictions=[],
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=0),
            latency_seconds=0.0,
            status="error",
            error=f"{error_type}: {error_msg}",
        )

    def _format_prompt(self, p: Union[str, List[Dict[str, str]]]) -> str:
        """Format a single prompt (chat messages or raw string) into text."""
        if isinstance(p, list):
            prefix_forcing = p[-1]["role"].lower() == "assistant" if p else False
            return self._tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=not prefix_forcing,
                continue_final_message=prefix_forcing,
            )
        return p

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a formatted prompt string."""
        return self._tokenizer(text, return_tensors="pt")["input_ids"].shape[1]

    def _infer_single_safe(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        formatted: str,
        example_id: str,
    ) -> InferenceResult:
        """Run inference on a single prompt with error handling."""
        prompt_tokens = self._count_tokens(formatted)
        try:
            inputs = self._tokenizer(
                formatted,
                return_tensors="pt",
                truncation=self.model_config.truncation,
            ).to(self._get_primary_device())

            gen_kwargs = self._build_generation_kwargs()
            start_time = time.time()
            with torch.inference_mode(), self._generation_autocast():
                outputs = self._model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )
            latency = time.time() - start_time

            generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
            completion_tokens = (
                generated_ids.numel() // self.inference_config.num_return_sequences
            )
            predictions = self._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            predictions = [pred.strip() for pred in predictions]

            return InferenceResult(
                id=example_id,
                prompt=prompt,
                predictions=predictions,
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
                latency_seconds=latency,
                status="ok",
            )
        except Exception as e:
            logger.warning(
                f"Inference failed for {example_id} ({prompt_tokens} tokens): {e}"
            )
            return self._make_error_result(prompt, example_id, e, prompt_tokens)

    def infer_batch(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        batch_ids: Optional[List[str]] = None,
        batch_size: int = 4,
    ) -> BatchInferenceResult:
        total_prompts = len(prompts)

        if batch_ids is None:
            batch_ids = [f"example_{i}" for i in range(total_prompts)]

        logger.info(
            f"Running batch inference: {total_prompts} prompts, batch size: {batch_size}"
        )

        results: List[InferenceResult] = []
        failed_count = 0
        total_start_time = time.time()
        num_batches = (total_prompts + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_prompts)

            batch_prompts = prompts[start_idx:end_idx]
            batch_id_slice = batch_ids[start_idx:end_idx]

            formatted_batch = [self._format_prompt(p) for p in batch_prompts]
            prompt_lens = [self._count_tokens(fp) for fp in formatted_batch]
            batch_number = batch_idx + 1
            short_ids = [
                example_id if len(example_id) <= 120 else f"{example_id[:117]}..."
                for example_id in batch_id_slice
            ]
            logger.info(
                "Progress: starting batch %d/%d; rows %d-%d/%d; "
                "completed=%d, failed=%d; prompt_tokens=%s; ids=%s",
                batch_number,
                num_batches,
                start_idx + 1,
                end_idx,
                total_prompts,
                len(results),
                failed_count,
                prompt_lens,
                short_ids,
            )

            try:
                inputs = self._tokenizer(
                    formatted_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=self.model_config.truncation,
                ).to(self._get_primary_device())

                padded_len = inputs.input_ids.shape[1]

                gen_kwargs = self._build_generation_kwargs()
                batch_start_time = time.time()

                with torch.inference_mode(), self._generation_autocast():
                    outputs = self._model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **gen_kwargs,
                    )

                batch_latency = time.time() - batch_start_time

                num_return = self.inference_config.num_return_sequences
                for i, (prompt, example_id, prompt_len) in enumerate(
                    zip(batch_prompts, batch_id_slice, prompt_lens)
                ):
                    seq_start = i * num_return
                    seq_end = seq_start + num_return
                    example_sequences = outputs.sequences[seq_start:seq_end]
                    generated_ids = example_sequences[:, padded_len:]
                    completion_tokens = generated_ids.numel() // num_return

                    predictions = self._tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    predictions = [pred.strip() for pred in predictions]

                    result = InferenceResult(
                        id=example_id,
                        prompt=prompt,
                        predictions=predictions,
                        usage=Usage(
                            prompt_tokens=prompt_len,
                            completion_tokens=completion_tokens,
                        ),
                        latency_seconds=batch_latency / len(batch_prompts),
                        status="ok",
                    )
                    results.append(result)

                logger.info(
                    "Progress: finished batch %d/%d in %.2fs; "
                    "completed=%d/%d, succeeded=%d, failed=%d",
                    batch_number,
                    num_batches,
                    batch_latency,
                    len(results),
                    total_prompts,
                    len(results) - failed_count,
                    failed_count,
                )

            except Exception as batch_err:
                # Batch failed — fall back to per-item inference so we don't
                # lose the entire batch. Each item gets its own try/except.
                logger.warning(
                    f"Batch {batch_number}/{num_batches} failed ({type(batch_err).__name__}), "
                    f"falling back to per-item inference for {len(batch_prompts)} prompts"
                )
                for prompt, formatted, example_id, prompt_len in zip(
                    batch_prompts, formatted_batch, batch_id_slice, prompt_lens
                ):
                    item_start_time = time.time()
                    result = self._infer_single_safe(prompt, formatted, example_id)
                    if result.status == "error":
                        failed_count += 1
                    results.append(result)
                    logger.info(
                        "Progress: fallback item done for batch %d/%d; "
                        "status=%s; prompt_tokens=%d; latency=%.2fs; "
                        "completed=%d/%d, succeeded=%d, failed=%d; id=%s",
                        batch_number,
                        num_batches,
                        result.status,
                        prompt_len,
                        time.time() - item_start_time,
                        len(results),
                        total_prompts,
                        len(results) - failed_count,
                        failed_count,
                        example_id,
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_latency = time.time() - total_start_time

        batch_result = BatchInferenceResult(
            results=results,
            total_latency_seconds=total_latency,
            batch_size=batch_size,
        )

        ok_count = total_prompts - failed_count
        logger.info(
            f"Batch inference done: {ok_count}/{total_prompts} succeeded, "
            f"{failed_count} failed, {total_latency:.2f}s total"
        )
        return batch_result
