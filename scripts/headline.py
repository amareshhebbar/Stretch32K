import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from sx.common import BASELINES, MODELS, dataset_repo, get_meta, read_json, repo_id, results_path, wpath, write_json


def pct(x):
    return "" if x is None else f"{x * 100:.1f}%"


def num(x):
    return "" if x is None else (f"{x:.2f}" if x < 1000 else f"{x:,.0f}")


def row(cfg, r):
    s = dict(r["summary"])
    t = str(cfg["target_len"])
    s["needle_target"] = sum(r["needle"][t].values()) / len(r["needle"][t]) if t in r.get("needle", {}) else None
    s["passkey_target"] = r["passkey"].get(t)
    return s


def build(cfg):
    rows = {}
    for tag in [f"base_{v}" for v in BASELINES] + MODELS:
        r = read_json(results_path(cfg, tag, "full"))
        if r:
            rows[tag] = row(cfg, r)
    if "base_none" not in rows:
        raise SystemExit("base_none results missing, run the baseline step first")
    trained = [m for m in MODELS if m in rows]
    if not trained:
        raise SystemExit("no model results yet, run the pipeline first")
    best = max(trained, key=lambda m: (rows[m]["long_score"], -(rows[m].get("ppl_target") or 1e9)))
    zs = max([t for t in rows if t.startswith("base_") and t != "base_none"] or ["base_none"], key=lambda t: rows[t]["long_score"])
    return rows, best, zs


LABEL = {"base_none": "no scaling", "base_pi": "PI, no training", "base_ntk": "NTK, no training", "base_yarn": "YaRN, no training", "base_dynamic": "dynamic NTK, no training"}


def label(cfg, tag):
    return f"{cfg['base_label']} ({LABEL[tag]})" if tag in LABEL else f"Stretch32K {tag.upper()}"


def sentence(cfg, rows, best, zs):
    b, v, z = rows["base_none"], rows[best], rows[zs]
    k = cfg["target_len"] // cfg["orig_len"]
    return (
        f"Stretch32K {best.upper()} extends {cfg['base_label']} from a {cfg['orig_len']:,} token window to {cfg['target_len']:,} tokens ({k}x) on a single GPU: "
        f"passkey retrieval at {cfg['target_len']:,} tokens {pct(v['passkey_target'])} (base model {pct(b['passkey_target'])}, best no training trick {pct(z['passkey_target'])}), "
        f"needle in a haystack {pct(v['needle_target'])} at {cfg['target_len']:,} tokens and {pct(v['needle_mean'])} across all lengths and depths, "
        f"and perplexity at {cfg['target_len']:,} tokens {num(v['ppl_target'])} versus {num(b['ppl_target'])} for the base model."
    )


def markdown(cfg, rows, best, zs):
    coll = get_meta(cfg, "collection_slug")
    T = f"{cfg['target_len']:,}"
    lines = [
        f"# Stretch32K results ({cfg['base_label']})",
        "",
        f"**Headline:** {sentence(cfg, rows, best, zs)}",
        "",
        f"The base model was trained with a {cfg['orig_len']:,} token window. Passkey hides a five digit number in filler text. Needle in a haystack hides a seven digit number inside real book text at every depth and length. "
        "The RULER subset adds multiple keys, multiple values and variable tracking. Perplexity is measured on PG19 test books (lower is better).",
        "",
        f"| Model | Passkey at {T} | Needle at {T} | Needle, all lengths | Passkey, all lengths | RULER subset | Long score | Perplexity at {cfg['eval']['lengths'][0]:,} | Perplexity at {T} |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = [f"base_{x}" for x in BASELINES] + sorted([m for m in MODELS if m in rows], key=lambda m: -rows[m]["long_score"])
    for tag in [t for t in order if t in rows]:
        v = rows[tag]
        name = label(cfg, tag)
        if tag in MODELS:
            name = f"[{name}](https://huggingface.co/{repo_id(cfg, tag)})"
        if tag == best:
            name = f"**{name}**"
        lines.append(f"| {name} | {pct(v['passkey_target'])} | {pct(v['needle_target'])} | {pct(v['needle_mean'])} | {pct(v['passkey_mean'])} | {pct(v['ruler_mean'])} | {pct(v['long_score'])} | {num(v['ppl_short'])} | {num(v.get('ppl_target'))} |")
    lines += [
        "",
        "The headline model is the trained model with the highest long score (mean of passkey, needle and RULER).",
        "",
        "## Check it yourself",
        "",
        f"* Every model card has the needle heatmap, perplexity curve and training settings: [collection](https://huggingface.co/collections/{coll})" if coll else "* Every model card has the needle heatmap, perplexity curve and training settings.",
        f"* Raw evaluation files: [{best.upper()} eval](https://huggingface.co/{repo_id(cfg, best)}/tree/main/eval).",
        f"* Training data: [{dataset_repo(cfg)}](https://huggingface.co/datasets/{dataset_repo(cfg)}).",
        "* Code to reproduce every number: [github.com/amareshhebbar/Stretch32K](https://github.com/amareshhebbar/Stretch32K).",
        "",
        f"Generated {time.strftime('%Y-%m-%d')} from the pipeline's own result files.",
        "",
    ]
    return "\n".join(lines)


def main():
    from huggingface_hub import HfApi

    from sx.common import load_cfg
    from sx.hub import login

    cfg = load_cfg()
    rows, best, zs = build(cfg)
    for tag, v in rows.items():
        print(f"{tag:15s} passkey@target {pct(v['passkey_target']):>7s}  needle@target {pct(v['needle_target']):>7s}  needle all {pct(v['needle_mean']):>7s}  ruler {pct(v['ruler_mean']):>7s}  ppl@target {num(v.get('ppl_target'))}")
    md = markdown(cfg, rows, best, zs)
    wpath(cfg, "report", "HEADLINE.md").write_text(md)
    write_json(wpath(cfg, "report", "headline.json"), {"best": best, "sentence": sentence(cfg, rows, best, zs), "rows": rows})
    print("\n" + sentence(cfg, rows, best, zs) + "\n")
    if "--local" in sys.argv:
        print(f"written to {wpath(cfg, 'report', 'HEADLINE.md')}, not uploaded")
        return
    login()
    api = HfApi()
    ds = dataset_repo(cfg)
    for f in ["HEADLINE.md", "headline.json"]:
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f"report/{f}", repo_id=ds, repo_type="dataset", commit_message=f"Add {f}")
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f, repo_id=repo_id(cfg, best), commit_message=f"Add {f}")
    for png in ["ppl_curve.png", "passkey_curve.png", f"needle_{best}.png"]:
        p = wpath(cfg, "report", png)
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"report/{png}", repo_id=ds, repo_type="dataset", commit_message=f"Add {png}")
    print("proof links:")
    print(f"  https://huggingface.co/{repo_id(cfg, best)}/blob/main/HEADLINE.md")
    print(f"  https://huggingface.co/datasets/{ds}/blob/main/report/HEADLINE.md")


if __name__ == "__main__":
    main()
