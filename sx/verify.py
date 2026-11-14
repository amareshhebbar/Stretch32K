import json
import shutil

from huggingface_hub import HfApi, snapshot_download

from .common import banner, read_json, repo_id, results_path, wpath, write_json


def verify_model(cfg, name):
    from transformers import AutoConfig

    from .evaluate import eval_model
    from .hub import login

    login()
    repo = repo_id(cfg, name)
    banner(f"verifying {repo} straight from the hub")
    files = HfApi().list_repo_files(repo)
    checks = {
        "config.json": "config.json" in files,
        "tokenizer": "tokenizer.json" in files or "tokenizer.model" in files,
        "safetensors": any(f.endswith(".safetensors") and "/" not in f for f in files),
        "README.md": "README.md" in files,
    }
    for k, v in checks.items():
        print(f"  {k}: {'OK' if v else 'MISSING'}", flush=True)
    if not all(checks.values()):
        raise SystemExit(f"verify {name}: repo is missing files")
    local = wpath(cfg, "verify", name)
    if local.exists():
        shutil.rmtree(local)
    try:
        snapshot_download(repo, local_dir=str(local), ignore_patterns=["training/*", "assets/*", "eval/*"], force_download=True)
        conf = AutoConfig.from_pretrained(str(local))
        rope = {"rope_theta": conf.rope_theta, "rope_scaling": conf.rope_scaling, "max_position_embeddings": conf.max_position_embeddings}
        print(f"  rope settings from the hub config: {rope}", flush=True)
        if conf.max_position_embeddings < cfg["target_len"] and not conf.rope_scaling:
            raise SystemExit(f"verify {name}: hub config has no long context settings")
        res = eval_model(cfg, name, "quick", str(local))
    finally:
        shutil.rmtree(local, ignore_errors=True)
    base = read_json(results_path(cfg, "base_none", "quick"))
    s = res["summary"]
    e = cfg["eval"]
    gates = {}
    if base:
        b = base["summary"]
        gates["short_ppl"] = s["ppl_short"] <= b["ppl_short"] * e["gate_short_ratio"]
        gates["target_ppl"] = s["ppl_target"] <= b["ppl_short"] * e["gate_target_ratio"]
    report = {"repo": repo, "rope": rope, "quick": s, "base_quick": base["summary"] if base else None, "gates": gates, "passed": all(gates.values())}
    write_json(wpath(cfg, "results", f"{name}_verify.json"), report)
    banner(f"verify {name}\n" + json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(f"verify {name} FAILED, see {wpath(cfg, 'results', name + '_verify.json')}")
    print(f"verify {name} PASSED", flush=True)
