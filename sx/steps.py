import os

from .common import ACTIONS, MODELS


def selected_models():
    only = os.environ.get("ONLY_MODELS", "").strip()
    if not only:
        return list(MODELS)
    want = [m.strip().lower() for m in only.split(",") if m.strip()]
    bad = [m for m in want if m not in MODELS]
    if bad:
        raise SystemExit(f"ONLY_MODELS has unknown models {bad}, choose from {MODELS}")
    return [m for m in MODELS if m in want]


def pipeline_steps():
    steps = ["factory", "baseline", "baseline_short"]
    for m in selected_models():
        steps += [f"{m}:{a}" for a in ACTIONS]
    steps.append("report")
    return steps


def run_step(cfg, step):
    if step == "factory":
        from .data import run_factory

        return run_factory(cfg)
    if step == "baseline":
        from .evaluate import run_baseline

        return run_baseline(cfg)
    if step == "baseline_short":
        from .evaluate import run_short

        return run_short(cfg, "base")
    if step == "report":
        from .report import run_report

        return run_report(cfg)
    if ":" not in step:
        raise SystemExit(f"unknown step {step}")
    name, action = step.split(":", 1)
    if name not in MODELS or action not in ACTIONS:
        raise SystemExit(f"unknown step {step}")
    if action == "train":
        from .train import train_cpt, train_sft

        return train_sft(cfg) if name == "sft" else train_cpt(cfg, name)
    if action == "push":
        from .hub import push_model

        return push_model(cfg, name)
    if action == "verify":
        from .verify import verify_model

        return verify_model(cfg, name)
    if action == "evalgen":
        from .evaluate import eval_model

        return eval_model(cfg, name, "full")
    if action == "shorteval":
        from .evaluate import run_short

        return run_short(cfg, name)
    if action == "publish":
        from .hub import publish

        return publish(cfg, name)
