import json
import math
import random
import shutil
import time

import numpy as np
import torch

from .common import banner, model_dir, read_json, read_jsonl, resolve_model, results_path, wpath, write_json, POLICIES
from .modeling import DEV, token_nll
from .rope import describe, load_config

LORA_METHODS = ("longlora", "yarn_longlora")


def build(cfg, name, init_path, method):
    from transformers import AutoModelForCausalLM

    lora = name in LORA_METHODS
    attn = "sdpa"
    if lora:
        from .s2attn import register

        attn = register(cfg["train"]["s2_group"])
    conf = load_config(cfg, init_path, method)
    conf.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(init_path, config=conf, torch_dtype=torch.float32, attn_implementation=attn).to(DEV)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if lora:
        from peft import LoraConfig, get_peft_model

        lc = cfg["lora"]
        model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=0.0, target_modules=lc["targets"], task_type="CAUSAL_LM"))
        for n, p in model.named_parameters():
            if "embed_tokens" in n or "norm" in n:
                p.requires_grad_(True)
    model.train()
    return model


def param_groups(cfg, model, name, lr):
    lora_lr = cfg["train"]["lora_lr"]
    g_lora, g_rest = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (g_lora if "lora_" in n else g_rest).append(p)
    groups = [{"params": g_rest, "lr": lr}]
    if g_lora:
        groups.append({"params": g_lora, "lr": lora_lr})
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for g in groups for p in g["params"])
    print(f"{name}: training {train:,} of {total:,} parameters ({100 * train / total:.2f}%)", flush=True)
    return groups


def loop(cfg, name, model, batches, lr, accum, warmup, total_steps, chunk):
    from transformers import get_cosine_schedule_with_warmup

    out = model_dir(cfg, name) / "training"
    out.mkdir(parents=True, exist_ok=True)
    groups = param_groups(cfg, model, name, lr)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, fused=DEV == "cuda")
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    log = open(out / "log.jsonl", "w")
    t0, step, acc, agg_loss, agg_tok = time.time(), 0, 0, 0.0, 0
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)
    for ids, labels in batches:
        with torch.autocast(device_type="cuda" if DEV == "cuda" else "cpu", dtype=torch.bfloat16):
            nll, n = token_nll(model, ids, labels, chunk)
        loss = nll / n.clamp(min=1)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: non finite loss at step {step}")
        (loss / accum).backward()
        agg_loss += loss.item()
        agg_tok += int(ids.numel())
        acc += 1
        if acc == accum:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            el = time.time() - t0
            rec = {
                "step": step,
                "loss": agg_loss / acc,
                "lr": sched.get_last_lr()[0],
                "grad_norm": gn,
                "tokens_seen": agg_tok,
                "tok_s": agg_tok / el,
                "elapsed_min": el / 60,
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0,
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step == 1 or step % 5 == 0 or step == total_steps:
                eta = el / step * (total_steps - step) / 60
                print(f"[{name}] step {step}/{total_steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gn {gn:.3f} tok/s {rec['tok_s']:.0f} peak {rec['peak_gib']:.1f}GiB eta_min {eta:.1f}", flush=True)
            agg_loss, acc = 0.0, 0
            if step >= total_steps:
                break
    log.close()
    plot(out / "log.jsonl", out / "loss.png", name)
    write_json(out / "summary.json", {"steps": step, "minutes": (time.time() - t0) / 60, "tokens": agg_tok})


def plot(log_path, png, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_jsonl(log_path)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([r["step"] for r in rows], [r["loss"] for r in rows])
    ax.set_title(f"{name} training loss")
    ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def save(cfg, name, model, init_path):
    from transformers import AutoTokenizer, GenerationConfig

    out = model_dir(cfg, name) / "merged"
    if out.exists():
        shutil.rmtree(out)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = True
    model = model.to(torch.bfloat16)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(init_path).save_pretrained(str(out))
    try:
        GenerationConfig.from_pretrained(init_path).save_pretrained(str(out))
    except Exception:
        pass
    write_json(model_dir(cfg, name) / "rope.json", describe(model.config))
    banner(f"saved {name} to {out} with {describe(model.config)}")


def cpt_batches(cfg):
    from .data import train_chunks_path

    arr = np.load(train_chunks_path(cfg), mmap_mode="r")
    for i in range(arr.shape[0]):
        ids = torch.from_numpy(arr[i].astype(np.int64))[None].to(DEV)
        yield ids, ids


def train_cpt(cfg, name):
    t = cfg["train"]
    torch.manual_seed(cfg["seed"])
    base = cfg["base_model"]
    banner(f"{name}: continued pretraining at {t['seq_len']} tokens, rope {name}, {t['steps']} steps x {t['grad_accum']} sequences")
    model = build(cfg, name, base, name)
    lr = t["lr"] if name not in LORA_METHODS else t["embed_norm_lr"]
    loop(cfg, name, model, cpt_batches(cfg), lr, t["grad_accum"], t["warmup_steps"], t["steps"], t["logits_chunk"])
    model.eval()
    save(cfg, name, model, base)


def pick_sft_source(cfg):
    src = cfg["sft"].get("init")
    if src and src != "auto":
        return src
    scored = []
    for n in POLICIES:
        r = read_json(results_path(cfg, n, "full"))
        if r:
            scored.append((r["summary"]["long_score"], -r["summary"].get("ppl_target", 1e9), n))
    if not scored:
        raise RuntimeError("no evaluated models to pick the SFT source from")
    scored.sort(reverse=True)
    print("SFT source candidates (long score, minus ppl, name):", scored, flush=True)
    return scored[0][2]


def sft_batches(cfg, tok, rows, epochs):
    rng = random.Random(cfg["seed"])
    for ep in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in order:
            r = rows[i]
            p = [tok.bos_token_id] + tok(r["prompt"], add_special_tokens=False)["input_ids"]
            o = tok(r["output"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = torch.tensor([p + o], device=DEV)
            labels = torch.tensor([[-100] * len(p) + o], device=DEV)
            yield ids, labels


def train_sft(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    s = cfg["sft"]
    src = pick_sft_source(cfg)
    path = resolve_model(cfg, src)
    banner(f"sft: full finetune of {src} ({path}) on LongAlpaca")
    tok = AutoTokenizer.from_pretrained(path)
    rows = read_jsonl(wpath(cfg, "data", "longalpaca_train.jsonl"))
    conf = load_config(cfg, path)
    conf.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(path, config=conf, torch_dtype=torch.float32, attn_implementation="sdpa").to(DEV)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    total = max(1, len(rows) * s["epochs"] // s["grad_accum"])
    write_json(model_dir(cfg, "sft") / "source.json", {"source": src, "examples": len(rows)})
    loop(cfg, "sft", model, sft_batches(cfg, tok, rows, s["epochs"]), s["lr"], s["grad_accum"], s["warmup_steps"], total, cfg["train"]["logits_chunk"])
    model.eval()
    save(cfg, "sft", model, path)
