from pathlib import Path

from .common import BASELINES, MODELS, dataset_repo, get_meta, read_json, repo_id, results_path, wpath

METHOD = {
    "pi": "Position Interpolation. Every position is divided by the stretch factor, so the longest new position lands inside the range the model was trained on. Then the whole model is finetuned on long PG19 books so it adapts to the denser positions.",
    "ntk": "NTK aware scaling. Instead of squeezing all positions equally, the RoPE base frequency is raised, which leaves the fast rotating dimensions (local word order) almost untouched and stretches the slow ones (long distance). Then full finetuning on long PG19 books.",
    "yarn": "YaRN. RoPE frequencies are split into bands: high frequencies are left alone, low frequencies are fully interpolated, with a smooth ramp between, plus an attention temperature correction for long sequences. Then full finetuning on long PG19 books.",
    "longlora": "LongLoRA with Position Interpolation. Training uses shifted sparse attention: the sequence is cut into groups and tokens attend inside their group, while half the heads use groups shifted by half a group so information flows across boundaries. Only LoRA adapters on the attention projections plus the embedding and normalization layers are trained. Inference uses normal full attention.",
    "yarn_longlora": "YaRN positions combined with LongLoRA training: shifted sparse attention, LoRA on the attention projections, and trainable embedding and normalization layers. Inference uses normal full attention.",
    "sft": "The best stretched model, then fully finetuned on LongAlpaca long document instructions so it answers questions about long inputs instead of only reading them.",
}

SHORT = ("ppl_short", "ppl_target", "passkey_mean", "passkey_beyond", "needle_mean", "ruler_mean", "long_score", "longqa_nll")


def _pct(x):
    return "" if x is None else f"{x * 100:.1f}"


def _f(x, fmt="{:.2f}"):
    return "" if x is None else fmt.format(x)


def _rows(cfg):
    from .evaluate import BASE_LABEL

    rows = []
    for v in BASELINES:
        r = read_json(results_path(cfg, f"base_{v}", "full"))
        if r:
            rows.append((f"base_{v}", f"{cfg['base_label']} ({BASE_LABEL[v]})", r, read_json(wpath(cfg, "results", "base_none_short.json"), {}).get("scores", {}) if v == "none" else {}))
    for m in MODELS:
        r = read_json(results_path(cfg, m, "full"))
        if r:
            rows.append((m, f"Stretch32K {m.upper()}", r, read_json(wpath(cfg, "results", f"{m}_short.json"), {}).get("scores", {})))
    return rows


def results_table(cfg, highlight=None):
    rows = _rows(cfg)
    if not rows:
        return "<p>Evaluation results are added after the full evaluation step.</p>"
    short_tasks = cfg["eval"].get("short_tasks") or []
    head = (
        f"<tr><th>Model</th><th>PPL @ {cfg['eval']['lengths'][0]}</th><th>PPL @ {cfg['target_len']}</th><th>Passkey (all lengths)</th>"
        f"<th>Passkey beyond {cfg['orig_len']}</th><th>Needle (grid mean)</th><th>RULER subset</th><th>Long score</th><th>LongAlpaca answer NLL</th>"
        + "".join(f"<th>{t}</th>" for t in short_tasks)
        + "</tr>"
    )
    body = []
    for tag, label, r, short in rows:
        s = r["summary"]
        cells = [label, _f(s.get("ppl_short")), _f(s.get("ppl_target")), _pct(s.get("passkey_mean")), _pct(s.get("passkey_beyond")), _pct(s.get("needle_mean")), _pct(s.get("ruler_mean")), _pct(s.get("long_score")), _f(s.get("longqa_nll"), "{:.3f}")]
        cells += [_pct(short.get(t)) for t in short_tasks]
        if tag == highlight:
            cells = [f"<b>{c}</b>" for c in cells]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table>\n" + head + "\n" + "\n".join(body) + "\n</table>"


def ppl_table(cfg):
    rows = _rows(cfg)
    if not rows:
        return ""
    lens = cfg["eval"]["lengths"]
    head = "<tr><th>Model</th>" + "".join(f"<th>{L}</th>" for L in lens) + "</tr>"
    body = ["<tr><td>" + label + "</td>" + "".join(f"<td>{_f(r['ppl'].get(str(L)))}</td>" for L in lens) + "</tr>" for _, label, r, _ in rows]
    return "<table>\n" + head + "\n" + "\n".join(body) + "\n</table>"


def ruler_table(cfg, name):
    r = read_json(results_path(cfg, name, "full"))
    if not r:
        return ""
    lens = cfg["eval"]["lengths"]
    head = "<tr><th>Task</th>" + "".join(f"<th>{L}</th>" for L in lens) + "</tr>"
    rows = [("passkey", r["passkey"])] + list(r["ruler"].items())
    body = ["<tr><td>" + t + "</td>" + "".join(f"<td>{_pct(v.get(str(L)))}</td>" for L in lens) + "</tr>" for t, v in rows]
    return "<table>\n" + head + "\n" + "\n".join(body) + "\n</table>"


def hyperparams(cfg, name):
    t = dict(cfg["train"])
    t["stretch_factor"] = cfg["target_len"] / cfg["orig_len"]
    if name in ("longlora", "yarn_longlora"):
        t.update({f"lora_{k}": v for k, v in cfg["lora"].items()})
        t["trainable"] = "LoRA on attention, embeddings, norms"
    elif name == "sft":
        t = dict(cfg["sft"])
        t["source"] = read_json(wpath(cfg, "models", "sft", "source.json"), {}).get("source")
    else:
        t["trainable"] = "all parameters"
    rope = read_json(wpath(cfg, "models", name, "rope.json"), {})
    t.update({f"saved_{k}": v for k, v in rope.items()})
    return "<table>\n" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in t.items()) + "\n</table>"


def card(cfg, name):
    repo = repo_id(cfg, name)
    base = cfg["base_model"]
    if name == "sft":
        src = read_json(wpath(cfg, "models", "sft", "source.json"), {}).get("source")
        base = repo_id(cfg, src) if src else base
    slug = get_meta(cfg, "collection_slug")
    coll = f"https://huggingface.co/collections/{slug}" if slug else f"https://huggingface.co/{cfg['hf_user']}"
    imgs = ""
    for f, alt in [(f"needle_{name}.png", "Needle in a haystack"), ("ppl_curve.png", "Perplexity vs length"), ("passkey_curve.png", "Passkey vs length")]:
        if Path(wpath(cfg, "report", f)).exists():
            imgs += f"\n![{alt}](assets/{f})\n"
    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
  - {base}
base_model_relation: finetune
datasets:
  - {dataset_repo(cfg)}
language:
  - en
tags:
  - stretch32k
  - long_context
  - rope_scaling
  - tinyllama
  - {name}
---

# Stretch32K {name.upper()} ({cfg['base_label']})

`{cfg['base_model']}` was trained with a {cfg['orig_len']} token window. This model reads {cfg['target_len']} tokens, a {cfg['target_len'] // cfg['orig_len']}x stretch, trained on a single GPU. It is one of six models in the [Stretch32K collection]({coll}), all trained on the same PG19 chunks with a different context extension method and evaluated on the same tests.

## Method

{METHOD[name]}

Training data: [{dataset_repo(cfg)}](https://huggingface.co/datasets/{dataset_repo(cfg)}). Memory tricks: PyTorch FlashAttention kernels, gradient checkpointing, chunked loss over the vocabulary, and packed sequences with no padding.

## Why finetune

Retrieval feeds a model a few chunks; it cannot reason across a whole book or contract in one pass. The base model rows below show what happens past {cfg['orig_len']} tokens with no training, and with the popular training free scaling tricks applied at load time.

## Results

Perplexity is measured on PG19 test books. Passkey hides a five digit key in repeated filler. Needle hides a seven digit number in real book text across a grid of lengths and depths. The RULER subset has multiple keys, multiple values and variable tracking. Long score is the mean of passkey, needle and RULER. LongAlpaca answer NLL is the loss on answers to held out long document questions (lower is better).

{results_table(cfg, highlight=name)}

### Perplexity by context length

{ppl_table(cfg)}

### Retrieval by context length (this model)

{ruler_table(cfg, name)}
{imgs}
## Training hyperparameters

{hyperparams(cfg, name)}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}", torch_dtype="bfloat16", attn_implementation="sdpa").cuda()
text = open("long_document.txt").read()
ids = tok(text + "\\n\\nSummary:", return_tensors="pt").input_ids[:, -{cfg['target_len'] - 256}:].cuda()
out = model.generate(ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

The RoPE scaling settings are stored in `config.json`, so the model loads with the right positions in transformers and in vLLM. This is a base model, not a chat model{"; it was instruction tuned with the template `### Instruction:` then `### Response:`" if name == "sft" else ""}.
"""


def write_card(cfg, name, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(card(cfg, name))
