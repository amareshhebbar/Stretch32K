from .common import ACTIONS, MODELS


def pipeline_steps():
    steps = ["factory", "baseline", "baseline_short"]
    for m in MODELS:
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
