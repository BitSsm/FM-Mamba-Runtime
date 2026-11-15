"""Append-only continuation schedules; never rewrite completed training's LR."""
import json
import math
from pathlib import Path


def load_plan(config):
    if not config.get("training_plan_file"):
        return None
    plan = json.loads(Path(config["training_plan_file"]).read_text(encoding="utf-8"))
    assert plan["initial_lr_horizon_epochs"] == config["epochs"] == 40
    end = config["epochs"]
    for segment in plan["extensions"]:
        assert segment["start_epoch"] == end and segment["end_epoch"] > end
        assert 0 < segment["min_lr"] <= segment["lr"]
        end = segment["end_epoch"]
    assert 0 < plan["target_epochs"] <= end
    return plan


def check_resume_plan(saved, current):
    if current is None:
        return
    assert saved is not None, "New holdout jobs must resume only their own planned checkpoints"
    assert saved["initial_lr_horizon_epochs"] == current["initial_lr_horizon_epochs"]
    assert current["extensions"][:len(saved["extensions"])] == saved["extensions"], "Cannot rewrite an existing LR segment"
    assert current["target_epochs"] >= saved["target_epochs"], "Continuation target cannot decrease"


def planned_lr(config, plan, step, batches, original_fn):
    if plan:
        for segment in plan["extensions"]:
            first, end = segment["start_epoch"] * batches, segment["end_epoch"] * batches
            if first <= step < end:
                progress = (step - first) / max(1, end - first - 1)
                return segment["min_lr"] + .5 * (segment["lr"] - segment["min_lr"]) * (1 + math.cos(math.pi * progress))
    return original_fn(config, step, batches)
