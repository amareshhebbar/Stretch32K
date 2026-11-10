from .common import BASELINES, MODELS, banner, dataset_repo, read_json, repo_id, results_path, wpath, write_json


def collect(cfg):
    out = {}
    for tag in [f"base_{v}" for v in BASELINES] + MODELS:
        r = read_json(results_path(cfg, tag, "full"))
        if r:
            out[tag] = r
    return out


def label(tag):
    return tag.replace("base_", "base ") if tag.startswith("base_") else tag.upper()


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def heatmap(cfg, tag, r):
    import numpy as np

    plt = _plt()
    lens = list(r["needle"])
    depths = list(next(iter(r["needle"].values())))
    grid = np.array([[r["needle"][L][d] for L in lens] for d in depths]) * 100
    fig, ax = plt.subplots(figsize=(1.1 * len(lens) + 2, 0.45 * len(depths) + 1.8))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(lens)))
    ax.set_xticklabels([f"{int(L) // 1024}k" for L in lens])
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{float(d) * 100:.0f}%" for d in depths])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Needle depth")
    ax.set_title(f"Needle in a haystack: {label(tag)}")
    for i in range(len(depths)):
        for j in range(len(lens)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="% retrieved")
    fig.tight_layout()
    fig.savefig(wpath(cfg, "report", f"needle_{tag}.png"), dpi=130)
    plt.close(fig)


def plots(cfg, res=None):
    res = res or collect(cfg)
    if not res:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, r in res.items():
        xs = sorted(int(k) for k in r["ppl"])
        ax.plot(xs, [r["ppl"][str(x)] for x in xs], marker="o", linestyle="--" if tag.startswith("base_") else "-", label=label(tag))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.axvline(cfg["orig_len"], color="grey", linewidth=0.8)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Perplexity on PG19 test (log)")
    ax.set_title("Perplexity vs context length")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(wpath(cfg, "report", "ppl_curve.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, r in res.items():
        xs = sorted(int(k) for k in r["passkey"])
        ax.plot(xs, [r["passkey"][str(x)] * 100 for x in xs], marker="o", linestyle="--" if tag.startswith("base_") else "-", label=label(tag))
    ax.set_xscale("log", base=2)
    ax.axvline(cfg["orig_len"], color="grey", linewidth=0.8)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Passkey retrieval %")
    ax.set_title("Passkey retrieval vs context length")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(wpath(cfg, "report", "passkey_curve.png"), dpi=140)
    plt.close(fig)


def run_report(cfg):
    from huggingface_hub import HfApi

    from .card import results_table
    from .hub import login, publish

    login()
    res = collect(cfg)
    if not res:
        raise RuntimeError("no full results to report")
    plots(cfg, res)
    summary = {t: r["summary"] for t, r in res.items()}
    write_json(wpath(cfg, "report", "summary.json"), summary)
    wpath(cfg, "report", "table.html").write_text(results_table(cfg))
    api = HfApi()
    for f in sorted(wpath(cfg, "report").iterdir()):
        if f.suffix in (".png", ".json", ".html") and not f.name.startswith("README"):
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"report/{f.name}", repo_id=dataset_repo(cfg), repo_type="dataset", commit_message=f"Add report {f.name}")
    for m in MODELS:
        if read_json(results_path(cfg, m, "full")):
            publish(cfg, m)
    banner("report done\n" + "\n".join(f"{k}: long score {v.get('long_score', 0):.3f} passkey {v['passkey_mean']:.3f} ppl short {v['ppl_short']:.2f} ppl target {v.get('ppl_target') or 0:.2f}" for k, v in summary.items()))
    for m in MODELS:
        print(f"https://huggingface.co/{repo_id(cfg, m)}", flush=True)
