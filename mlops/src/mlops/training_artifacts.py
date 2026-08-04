from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def persist_training_artifacts(
    *,
    trainer,
    output_dir: str | Path,
    logging_dir: str | Path | None = None,
    train_metrics: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts_dir = output_path / "_training_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if run_config is not None:
        _write_json(artifacts_dir / "run_config.json", run_config)

    if train_metrics is not None:
        _write_json(artifacts_dir / "train_metrics.json", train_metrics)

    trainer_state = getattr(trainer, "state", None)
    if trainer_state is not None:
        trainer_state_path = artifacts_dir / "trainer_state.json"
        save_to_json = getattr(trainer_state, "save_to_json", None)
        if callable(save_to_json):
            save_to_json(str(trainer_state_path))
        else:
            state_payload = (
                trainer_state.to_dict()
                if hasattr(trainer_state, "to_dict")
                else getattr(trainer_state, "__dict__", {})
            )
            _write_json(trainer_state_path, state_payload)

        log_history = getattr(trainer_state, "log_history", None)
        if log_history is not None:
            _write_json(artifacts_dir / "log_history.json", log_history)

        training_summary = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "global_step": getattr(trainer_state, "global_step", None),
            "epoch": getattr(trainer_state, "epoch", None),
            "best_metric": getattr(trainer_state, "best_metric", None),
            "best_global_step": getattr(trainer_state, "best_global_step", None),
            "best_model_checkpoint": getattr(
                trainer_state, "best_model_checkpoint", None
            ),
        }
        _write_json(artifacts_dir / "training_summary.json", training_summary)

    if logging_dir is not None:
        source_logs = Path(logging_dir)
        if source_logs.exists():
            dest_logs = artifacts_dir / "logs"
            if dest_logs.exists():
                shutil.rmtree(dest_logs)
            shutil.copytree(source_logs, dest_logs)

    logger.info("Persisted training artifacts to %s", artifacts_dir)
    return artifacts_dir
