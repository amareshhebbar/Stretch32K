import glob
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

from .common import BASELINES, banner, read_json, read_jsonl, resolve_model, results_path, wpath, write_json, write_jsonl
from .data import haystack, ppl_docs
from .modeling import DEV, load_for_eval, token_nll

CITIES = ["Lisbon", "Nairobi", "Kyoto", "Quito", "Oslo", "Hanoi", "Dakar", "Perth", "Lima", "Tbilisi", "Cusco", "Bergen", "Mysore", "Tallinn", "Rabat", "Hobart"]
PASS_INTRO = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information there.\n"
PASS_FILLER = " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."


class Ctx:
    def __init__(self, cfg, model, tok):
        self.cfg, self.model, self.tok = cfg, model, tok
        self.hay = haystack(cfg)
        self.filler = self.enc(PASS_FILLER)

    def enc(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def gen(self, ids, max_new):
        x = torch.tensor([ids], device=DEV)
        out = self.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, len(ids) :], skip_special_tokens=True)


def place(body, inserts):
    out, last = [], 0
    for pos, ids in sorted(inserts, key=lambda x: x[0]):
        out.extend(body[last:pos])
        out.extend(ids)
        last = pos
    out.extend(body[last:])
    return out


def build(ctx, L, head, inserts, tail, filler=None, depths=None):
    fixed = 1 + len(head) + len(tail) + sum(len(i) for i in inserts) + 4
    n = max(0, L - fixed)
    if filler is not None:
        reps = filler * (n // len(filler) + 1)
        body = reps[:n]
        step = len(filler)
        pos = [min(n, int(round(d * n / step)) * step) for d in depths]
    else:
        start = random.randrange(0, max(1, len(ctx.hay) - n - 1))
        body = [int(x) for x in ctx.hay[start : start + n]]
        pos = [int(d * n) for d in depths]
    return [ctx.tok.bos_token_id] + head + place(body, list(zip(pos, inserts))) + tail


def passkey(ctx, L, depth, rng):
    key = rng.randint(10000, 99999)
    needle = ctx.enc(f" The pass key is {key}. Remember it. {key} is the pass key.")
    ids = build(ctx, L, ctx.enc(PASS_INTRO), [needle], ctx.enc("\nWhat is the pass key? The pass key is"), ctx.filler, [depth])
    return float(str(key) in ctx.gen(ids, 10))


def needle(ctx, L, depth, rng):
    city, num = rng.choice(CITIES), rng.randint(1000000, 9999999)
    ins = ctx.enc(f" The special magic number for {city} is: {num}.")
    q = ctx.enc(f"\nWhat is the special magic number for {city} mentioned in the provided text? The special magic number for {city} mentioned in the provided text is")
    return float(str(num) in ctx.gen(build(ctx, L, [], [ins], q, depths=[depth]), 12))


def multikey(ctx, L, rng):
    cities = rng.sample(CITIES, 4)
    nums = [rng.randint(1000000, 9999999) for _ in cities]
    ins = [ctx.enc(f" The special magic number for {c} is: {n}.") for c, n in zip(cities, nums)]
    k = rng.randrange(4)
    q = ctx.enc(f"\nWhat is the special magic number for {cities[k]} mentioned in the provided text? The special magic number for {cities[k]} mentioned in the provided text is")
    ids = build(ctx, L, [], ins, q, depths=sorted(rng.random() for _ in ins))
    return float(str(nums[k]) in ctx.gen(ids, 12))


def multivalue(ctx, L, rng):
    city = rng.choice(CITIES)
    nums = [rng.randint(1000000, 9999999) for _ in range(3)]
    ins = [ctx.enc(f" One of the special magic numbers for {city} is: {n}.") for n in nums]
    q = ctx.enc(f"\nWhat are all the special magic numbers for {city} mentioned in the provided text? The special magic numbers for {city} mentioned in the provided text are")
    out = ctx.gen(build(ctx, L, [], ins, q, depths=sorted(rng.random() for _ in ins)), 40)
    return sum(str(n) in out for n in nums) / len(nums)


def vartrack(ctx, L, rng):
    names = ["".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) for _ in range(4)]
    val = rng.randint(10000, 99999)
    ins = [ctx.enc(f" VAR {names[0]} = {val}.")] + [ctx.enc(f" VAR {names[i]} = VAR {names[i - 1]}.") for i in range(1, 4)]
    q = ctx.enc(f"\nQuestion: Find all variables that are assigned the value {val} in the text above. Answer: According to the chain of variable assignments in the text above, 4 variables are assigned the value {val}, they are:")
    out = ctx.gen(build(ctx, L, [], ins, q, depths=sorted(rng.random() for _ in ins)), 30)
    return sum(n in out for n in names) / len(names)


RULER = {"multikey": multikey, "multivalue": multivalue, "vartrack": vartrack}


@torch.no_grad()
def perplexity(cfg, model, lengths):
    docs = ppl_docs(cfg)
    out = {}
    for L in lengths:
        tot, n = 0.0, 0.0
        for d in docs:
            ids = torch.tensor([d[:L].astype(np.int64)], device=DEV)
            nll, k = token_nll(model, ids, ids, cfg["train"]["logits_chunk"])
            tot += nll.item()
            n += k.item()
        out[str(L)] = math.exp(tot / max(1.0, n))
        print(f"  ppl @ {L}: {out[str(L)]:.3f}", flush=True)
    return out


@torch.no_grad()
def longqa_nll(cfg, model, tok):
    rows = read_jsonl(wpath(cfg, "data", "longalpaca_heldout.jsonl"))
    tot, n = 0.0, 0.0
    for r in rows:
        p = [tok.bos_token_id] + tok(r["prompt"], add_special_tokens=False)["input_ids"]
        o = tok(r["output"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = torch.tensor([p + o], device=DEV)
        labels = torch.tensor([[-100] * len(p) + o], device=DEV)
        nll, k = token_nll(model, ids, labels, cfg["train"]["logits_chunk"])
        tot += nll.item()
        n += k.item()
    return tot / max(1.0, n) if rows else None


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def run_suite(cfg, model, tok, tag, mode, label=None):
    e = cfg["eval"]
    rng = random.Random(cfg["seed"])
    random.seed(cfg["seed"])
    ctx = Ctx(cfg, model, tok)
    t0 = time.time()
    out = {"tag": tag, "label": label, "mode": mode, "rope": {"rope_theta": model.config.rope_theta, "rope_scaling": model.config.rope_scaling, "max_position_embeddings": model.config.max_position_embeddings}}
    lengths = e["lengths"] if mode == "full" else [e["lengths"][0], cfg["target_len"]]
    banner(f"[{tag} | {mode}] perplexity at {lengths}")
    out["ppl"] = perplexity(cfg, model, lengths)
    trials = e["passkey_trials"] if mode == "full" else e["quick_trials"]
    out["passkey"] = {}
    for L in lengths:
        out["passkey"][str(L)] = mean(passkey(ctx, L, (i + 0.5) / trials, rng) for i in range(trials))
        print(f"  passkey @ {L}: {out['passkey'][str(L)]:.2f}", flush=True)
    if mode == "full":
        out["needle"] = {}
        for L in lengths:
            out["needle"][str(L)] = {str(d): mean(needle(ctx, L, d, rng) for _ in range(e["needle_trials"])) for d in e["needle_depths"]}
            print(f"  needle @ {L}: {mean(out['needle'][str(L)].values()):.2f}", flush=True)
        out["ruler"] = {}
        for task, fn in RULER.items():
            out["ruler"][task] = {str(L): mean(fn(ctx, L, rng) for _ in range(e["ruler_trials"])) for L in lengths}
            print(f"  {task}: {out['ruler'][task]}", flush=True)
        out["longqa_nll"] = longqa_nll(cfg, model, tok)
    s = {"ppl_short": out["ppl"][str(lengths[0])], "ppl_target": out["ppl"].get(str(cfg["target_len"])), "passkey_mean": mean(out["passkey"].values())}
    beyond = [L for L in lengths if L > cfg["orig_len"]]
    s["passkey_beyond"] = mean(out["passkey"][str(L)] for L in beyond)
    if mode == "full":
        s["needle_mean"] = mean(v for row in out["needle"].values() for v in row.values())
        s["ruler_mean"] = mean(v for t in out["ruler"].values() for v in t.values())
        s["long_score"] = mean([s["passkey_mean"], s["needle_mean"], s["ruler_mean"]])
        s["longqa_nll"] = out["longqa_nll"]
    out["summary"] = s
    out["minutes"] = (time.time() - t0) / 60
    write_json(results_path(cfg, tag, mode), out)
    banner(f"[{tag}] summary {json.dumps(s)}")
    if mode == "full":
        from .report import heatmap

        heatmap(cfg, tag, out)
    return out


def tokenizer(cfg, path=None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path or cfg["base_model"])


BASE_LABEL = {
    "none": "no scaling",
    "pi": "position interpolation, no training",
    "ntk": "NTK aware scaling, no training",
    "yarn": "YaRN, no training",
    "dynamic": "dynamic NTK, no training",
}


def run_baseline(cfg):
    tok = tokenizer(cfg)
    for v in BASELINES:
        tag = f"base_{v}"
        if read_json(results_path(cfg, tag, "full")):
            print(f"skip {tag}, already evaluated", flush=True)
            continue
        model = load_for_eval(cfg, cfg["base_model"], v)
        if v == "none":
            run_suite(cfg, model, tok, tag, "quick", BASE_LABEL[v])
        run_suite(cfg, model, tok, tag, "full", BASE_LABEL[v])
        del model
        torch.cuda.empty_cache()


def eval_model(cfg, name, mode="full", path=None):
    path = path or resolve_model(cfg, name)
    model = load_for_eval(cfg, path)
    return run_suite(cfg, model, tokenizer(cfg, path), name, mode)


def run_short(cfg, name):
    tasks = cfg["eval"].get("short_tasks") or []
    if not tasks:
        print("short_tasks empty in config, skipping", flush=True)
        return
    tag = "base_none" if name == "base" else name
    path = cfg["base_model"] if name == "base" else resolve_model(cfg, name)
    out_dir = wpath(cfg, "results", f"short_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    lm_eval_bin = os.path.join(os.path.dirname(sys.executable), "lm_eval")
    cmd = [lm_eval_bin, "--model", "hf", "--model_args", f"pretrained={path},dtype=bfloat16,attn_implementation=sdpa", "--tasks", ",".join(tasks), "--batch_size", "16", "--device", "cuda", "--output_path", str(out_dir)]
    banner("running " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    files = sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True), key=os.path.getmtime)
    res = json.loads(open(files[-1]).read())["results"]
    scores = {t: res[t].get("acc_norm,none", res[t].get("acc,none")) for t in tasks if t in res}
    write_json(wpath(cfg, "results", f"{tag}_short.json"), {"tag": tag, "scores": scores})
    banner(f"[{tag}] short context scores {scores}")
