"""Standalone batch inference driver.

Runs single-node (optionally multi-GPU) batch inference with a Hugging Face
causal LM and writes predictions to JSONL/JSON. ``--model`` is a Hugging Face
hub id or a local directory; PEFT/LoRA adapters are applied on top of it via
``--adapter-path``.
"""

import argparse
import json
import os
import logging
import base64
from typing import List, Dict, Any

from mlops.infer import ModelConfig, InferenceEngine, InferenceConfig

# Configure environment variables for model compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_DISABLE_FLASH_ATTN"] = "1"

# DeepSpeed environment variables
os.environ["NCCL_TIMEOUT"] = "7200"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "lo")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # Single-node only: force loopback for all multi-GPU communication
        os.environ["MASTER_ADDR"] = "127.0.0.1"
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_prompts_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load prompts from a JSONL file.

    Supports formats:
    1. {"id": "...", "prompt": "string"} — raw string prompt
    2. {"id": "...", "prompt": "[{...}]"} — JSON string containing messages array
    3. {"id": "...", "messages": [{...}]} — direct messages array
    """
    prompts_data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "prompt" in data and isinstance(data["prompt"], str):
                            try:
                                messages = json.loads(data["prompt"])
                                if isinstance(messages, list):
                                    data["messages"] = messages
                            except json.JSONDecodeError:
                                pass
                        prompts_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON on line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

    logger.info(f"Loaded {len(prompts_data)} prompts from {file_path}")
    return prompts_data


def save_results_to_jsonl(results: List[Dict[str, Any]], file_path: str) -> None:
    """Save results to a JSONL file."""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(results)} results to {file_path}")
    except Exception as e:
        logger.error(f"Error saving results to {file_path}: {e}")
        raise


def save_results_to_json(
    results: List[Dict[str, Any]], file_path: str, model: str, display_name: str
) -> None:
    """Save results to a single JSON file in InferenceResults format."""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        inference_results = []
        for result in results:
            try:
                messages = (
                    json.loads(result["prompt"])
                    if isinstance(result["prompt"], str)
                    else result["prompt"]
                )
            except Exception:
                messages = [{"role": "user", "content": str(result["prompt"])}]

            inference_results.append(
                {
                    "id": result["id"],
                    "messages": messages,
                    "predictions": result["predictions"],
                    "usage": {
                        "prompt_tokens": result["usage"]["prompt_tokens"],
                        "completion_tokens": result["usage"]["completion_tokens"],
                    },
                }
            )

        output_data = {
            "parameters": {
                "model": model.split("/")[-1],
                "prompt": display_name,
                "dataset": None,
            },
            "settings": {},
            "results": inference_results,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(output_data, ensure_ascii=False, indent=2))
        logger.info(f"Saved {len(results)} results to JSON file {file_path}")
    except Exception as e:
        logger.error(f"Error saving results to {file_path}: {e}")
        raise


def predict():
    parser = argparse.ArgumentParser(
        description="Run batch inference using a language model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face hub id or local path of the model to load.",
    )
    parser.add_argument(
        "--prompts", type=str, required=True, help="Path to JSONL file with prompts."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output JSONL file."
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for inference."
    )
    parser.add_argument(
        "--display-name",
        type=str,
        default="inference",
        help="Display name for the job.",
    )
    parser.add_argument(
        "--prefix", type=str, default=None, help="Prefix to force at generation start."
    )
    parser.add_argument(
        "--prefix-base64",
        type=str,
        default=None,
        help="Base64-encoded prefix.",
    )
    parser.add_argument(
        "--inference-kwargs-base64",
        type=str,
        default=None,
        help="Base64-encoded JSON with inference parameters.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help=(
            "Optional path to a PEFT/LoRA adapter directory. When set, --model is "
            "loaded as the base model and the adapter is applied on top."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Optional informational base-model id (used only when --adapter-path is set).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models that require custom remote code.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed inference.",
    )

    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = (
        args.local_rank
        if args.local_rank != -1
        else int(os.environ.get("LOCAL_RANK", 0))
    )
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    if rank == 0:
        logger.info(f"Running single-node multi-GPU inference with {world_size} processes")

    # Decode base64 prefix if provided
    prefix = args.prefix
    if args.prefix_base64:
        try:
            prefix = base64.b64decode(args.prefix_base64.encode("ascii")).decode(
                "utf-8"
            )
            if rank == 0:
                logger.info(f"Decoded prefix from base64: {prefix}")
        except Exception as e:
            if rank == 0:
                logger.error(f"Failed to decode base64 prefix: {e}")
            raise

    # Decode inference kwargs if provided
    inference_kwargs = {}
    if args.inference_kwargs_base64:
        try:
            inference_kwargs_json = base64.b64decode(
                args.inference_kwargs_base64.encode("ascii")
            ).decode("utf-8")
            inference_kwargs = json.loads(inference_kwargs_json)
            if rank == 0:
                logger.info(f"Decoded inference kwargs: {inference_kwargs}")
        except Exception as e:
            if rank == 0:
                logger.error(f"Failed to decode inference kwargs: {e}")
            raise

    if rank == 0:
        logger.info(f"Loading prompts from {args.prompts}")
    prompts_data = load_prompts_from_jsonl(args.prompts)

    if not prompts_data:
        if rank == 0:
            logger.error("No prompts loaded. Exiting.")
        return

    prompts = []
    ids = []
    for i, data in enumerate(prompts_data):
        if "messages" in data:
            prompts.append(data["messages"])
            ids.append(str(data.get("id", f"prompt_{i}")))
        elif "prompt" in data:
            prompts.append(data["prompt"])
            ids.append(str(data.get("id", f"prompt_{i}")))
        else:
            if rank == 0:
                logger.warning(
                    f"Missing 'prompt' or 'messages' field at index {i}"
                )

    if not prompts:
        if rank == 0:
            logger.error("No valid prompts found. Exiting.")
        return

    if rank == 0:
        logger.info(f"Processing {len(prompts)} prompts with model {args.model}")

    # The model is loaded directly from --model (a Hugging Face hub id or local
    # path). Adapters, when supplied, are applied on top of that base model.
    model_path = args.model
    adapter_path = args.adapter_path
    base_model = args.base_model
    if adapter_path and rank == 0:
        logger.info(
            f"Adapter mode: base model '{model_path}', adapter '{adapter_path}'"
        )

    if prefix:
        if rank == 0:
            logger.info(f"Applying prefix forcing: '{prefix}'")
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, list):
                if prompt[-1]["role"].lower() == "assistant":
                    prompt.pop()
                prompt.append({"role": "assistant", "content": prefix})

    # Defensive guard: if any prompt still has a trailing assistant turn whose
    # content is not exactly the configured prefix, the chat template will fall
    # into continue_final_message mode and the model will emit EOS immediately
    # (a silent 0-completion-token failure that occurs when running inference on
    # SFT training prompts that already include the gold answer).
    bad_prompts: list[tuple[int, str]] = []
    for i, prompt in enumerate(prompts):
        if isinstance(prompt, list) and prompt:
            last = prompt[-1]
            if last.get("role", "").lower() == "assistant":
                content = last.get("content", "")
                if not prefix or content != prefix:
                    bad_prompts.append((i, ids[i] if i < len(ids) else f"prompt_{i}"))
    if bad_prompts:
        sample = bad_prompts[:5]
        msg = (
            f"{len(bad_prompts)}/{len(prompts)} prompts have a trailing assistant "
            f"turn that is not the configured prefix. This will cause the chat "
            f"template to use continue_final_message and the model to emit EOS "
            f"immediately (empty predictions). Strip trailing assistant turns "
            f"from the prompts file or supply --prefix. Sample ids: {sample}"
        )
        if rank == 0:
            logger.error(msg)
        raise ValueError(msg)

    model_config = ModelConfig(
        model=model_path,
        adapter_path=adapter_path,
        base_model=base_model,
        trust_remote_code=args.trust_remote_code,
    )

    # Build inference config with optional kwargs
    inference_config_params = {"prefix": prefix}
    if inference_kwargs:
        for key, value in inference_kwargs.items():
            param_key = key.replace("-", "_")
            inference_config_params[param_key] = value
        if rank == 0:
            logger.info(f"Inference config parameters: {inference_config_params}")

    inference_config = InferenceConfig(**inference_config_params)
    inference_engine = InferenceEngine(
        model_config=model_config, inference_config=inference_config
    )

    if rank == 0:
        logger.info(f"Running batch inference with batch size {args.batch_size}")
    batch_results = inference_engine.infer_batch(
        prompts=prompts, batch_ids=ids, batch_size=args.batch_size
    )

    # Only rank 0 saves results
    if rank == 0:
        results_for_output = []
        for i, result in enumerate(batch_results.results):
            predictions = result.predictions
            if prefix and predictions and not any(pred.startswith(prefix) for pred in predictions):
                predictions = [prefix + pred for pred in predictions]

            output_data = {
                "id": result.id,
                "prompt": result.prompt,
                "predictions": predictions,
                "status": result.status,
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                },
                "latency_seconds": result.latency_seconds,
                **prompts_data[i],
            }
            if result.error:
                output_data["error"] = result.error
            results_for_output.append(output_data)

        logger.info(f"Saving results to {args.output}")
        save_results_to_jsonl(results_for_output, args.output)

        json_output = args.output.replace(".jsonl", ".json")
        logger.info(f"Saving JSON format: {json_output}")
        save_results_to_json(
            results_for_output, json_output, args.model, args.display_name
        )

        total_tokens = sum(r["usage"]["total_tokens"] for r in results_for_output)
        ok_results = [r for r in results_for_output if r["status"] == "ok"]
        failed_results = [r for r in results_for_output if r["status"] == "error"]
        avg_latency = (
            sum(r["latency_seconds"] for r in ok_results) / len(ok_results)
            if ok_results
            else 0
        )

        logger.info(
            f"Batch inference completed: "
            f"{len(ok_results)}/{len(results_for_output)} succeeded, "
            f"{len(failed_results)} failed, "
            f"{batch_results.total_latency_seconds:.2f}s total, "
            f"{avg_latency:.2f}s avg, "
            f"{total_tokens} total tokens"
        )

        if failed_results:
            logger.warning(f"=== FAILURE SUMMARY ({len(failed_results)} prompts) ===")
            for r in failed_results:
                logger.warning(
                    f"  {r['id']}: {r.get('error', 'unknown')} "
                    f"(prompt_tokens={r['usage']['prompt_tokens']})"
                )


if __name__ == "__main__":
    predict()
