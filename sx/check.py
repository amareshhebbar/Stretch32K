import importlib
import platform
import shutil
import subprocess
import sys
import time
import uuid

PARTS = ["sys", "hub", "data", "rope", "s2", "train"]

PACKAGES = ["torch", "transformers", "peft", "accelerate", "datasets", "huggingface_hub", "lm_eval", "safetensors", "sentencepiece", "yaml", "numpy", "matplotlib", "hf_transfer"]


def ok(msg):
    print(f"  [PASS] {msg}", flush=True)


def warn(msg):
    print(f"  [WARN] {msg}", flush=True)


def fail(msg):
    print(f"  [FAIL] {msg}", flush=True)
    raise SystemExit(1)


def check_sys(cfg):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print(f"  python {platform.python_version()} at {sys.executable}", flush=True)
    print(f"  config {cfg['config_path']}, stretch {cfg['orig_len']} to {cfg['target_len']}, train seq_len {cfg['train']['seq_len']}", flush=True)
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            ok(f"import {name} {getattr(m, '__version__', '')}")
        except Exception as e:
            fail(f"import {name}: {e!r}")
    if shutil.which("nvidia-smi"):
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(), flush=True)
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 2**30
    ok(f"GPU 0: {p.name}, {vram:.1f} GiB, compute capability {p.major}.{p.minor}, torch CUDA {torch.version.cuda}")
    if p.major < 8 or not torch.cuda.is_bf16_supported():
        fail("bf16 and FlashAttention need an Ampere or newer GPU (A100, A6000, L40S, H100, RTX 3090/4090)")
    n = cfg["train"]["seq_len"]
    q = torch.randn(1, 32, n, 64, device="cuda", dtype=torch.bfloat16)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.cuda.synchronize()
            t = time.time()
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.cuda.synchronize()
        ok(f"FlashAttention kernel runs causal attention over {n} tokens in {time.time() - t:.2f}s")
    except Exception as e:
        fail(f"FlashAttention kernel not available: {e!r}")
    if vram < 40:
        warn(f"{vram:.0f} GiB VRAM. Configs target 48 GiB. Lower train.seq_len and target_len if the train check runs out of memory.")
    free = shutil.disk_usage(cfg["paths"]["work"]).free / 2**30
    print(f"  work dir {cfg['paths']['work']}, free disk {free:.0f} GiB", flush=True)
    if free < 50:
        warn("under 50 GiB free. Six models of about 2.2 GiB, training chunks and caches need room.")
    else:
        ok("disk space")


def check_hub(cfg):
    from huggingface_hub import HfApi

    from .hub import login

    try:
        login()
    except Exception as e:
        fail(f"login: {e}")
    api = HfApi()
    me = api.whoami()
    name = me.get("name")
    ok(f"logged in as {name}")
    if name != cfg["hf_user"] and cfg["hf_user"] not in [o.get("name") for o in me.get("orgs", [])]:
        fail(f"token belongs to {name} but config hf_user is {cfg['hf_user']}. Run: export HF_USER={name}")
    role = me.get("auth", {}).get("accessToken", {}).get("role")
    print(f"  token role: {role}", flush=True)
    if role == "read":
        fail("token is read only. Create a write token at huggingface.co/settings/tokens")
    tag = uuid.uuid4().hex[:6]
    repo = f"{cfg['hf_user']}/stretch32k_write_test_{tag}"
    try:
        api.create_repo(repo, private=True)
        api.upload_file(path_or_fileobj=b"ok", path_in_repo="probe.txt", repo_id=repo, commit_message="write probe")
        ok(f"create repo and upload file ({repo})")
        c = api.create_collection(title=f"stretch32k write test {tag}", namespace=cfg["hf_user"], private=True)
        api.add_collection_item(c.slug, repo, "model", note="probe")
        api.delete_collection(c.slug)
        ok("create, fill and delete a collection")
    finally:
        try:
            api.delete_repo(repo)
            ok("delete test repo")
        except Exception as e:
            warn(f"could not delete {repo}: {e}")


def check_data(cfg):
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    from .data import LONGALPACA, stream_books

    conf = AutoConfig.from_pretrained(cfg["base_model"])
    if conf.max_position_embeddings != cfg["orig_len"]:
        warn(f"base model max_position_embeddings is {conf.max_position_embeddings}, config orig_len is {cfg['orig_len']}")
    ok(f"base {cfg['base_model']}: {conf.num_hidden_layers} layers, hidden {conf.hidden_size}, heads {conf.num_attention_heads}/{conf.num_key_value_heads}, rope_theta {conf.rope_theta}, window {conf.max_position_embeddings}")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if len(tok) > 65535:
        fail("vocab too large for uint16 token storage")
    ok(f"tokenizer vocab {len(tok)}, bos {tok.bos_token}, eos {tok.eos_token}")
    for split in ["train", "test"]:
        t = time.time()
        text = next(iter(stream_books(split)))
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        ok(f"PG19 {split} streams: first book {n:,} tokens ({time.time() - t:.0f}s)")
    ds = load_dataset(LONGALPACA, split="train")
    cols = ds.column_names
    if "instruction" not in cols or "output" not in cols:
        fail(f"LongAlpaca columns changed: {cols}")
    ok(f"LongAlpaca: {len(ds)} rows, columns {cols}")


def check_rope(cfg):
    import torch

    from .data import eval_books
    from .evaluate import Ctx, passkey, perplexity
    from .modeling import load_for_eval

    eval_books(cfg)
    ok("PG19 test books cached for evaluation")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    short = cfg["eval"]["lengths"][0]
    import random

    for method in ["none", "pi", "ntk", "yarn", "dynamic"]:
        model = load_for_eval(cfg, cfg["base_model"], method)
        sc = model.config.rope_scaling
        p = perplexity(cfg, model, [short, cfg["target_len"]])
        rng = random.Random(0)
        ctx = Ctx(cfg, model, tok)
        pk_short = sum(passkey(ctx, short, d, rng) for d in (0.2, 0.5, 0.8)) / 3
        print(f"  {method:8s} rope_scaling {sc} theta {model.config.rope_theta:.0f}: ppl {p}, passkey @ {short} {pk_short:.2f}", flush=True)
        if not all(v == v and v < 1e6 for v in [p[str(short)]]):
            fail(f"{method}: short perplexity is broken")
        del model, ctx
        torch.cuda.empty_cache()
    ok("all five RoPE settings load, score perplexity and run passkey generation")


def check_s2(cfg):
    import torch

    from .s2attn import s2_core

    torch.manual_seed(0)
    g = cfg["train"]["s2_group"]
    n = g * 4
    q = torch.randn(1, 8, n, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    full = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    same = s2_core(q, k, v, n)
    if not torch.allclose(full, same, atol=2e-2):
        fail("S2 attention with one group does not match full causal attention")
    ok("S2 attention with one group equals full causal attention")
    t = n // 2 + 7
    out = s2_core(q, k, v, g)
    out[:, :, t].float().sum().backward()
    leak = max(x.grad[:, :, t + 1 :].abs().max().item() for x in (k, v))
    if leak != 0.0:
        fail(f"S2 attention leaks information from future tokens ({leak})")
    ok(f"S2 attention is causal: no gradient from future tokens (group {g})")


def check_train(cfg):
    import torch

    from .common import model_dir
    from .modeling import DEV, load_for_eval, token_nll
    from .train import build, param_groups, save

    t = cfg["train"]
    L = t["seq_len"]
    est = {}
    for name in ["pi", "longlora"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build(cfg, name, cfg["base_model"], name)
        lr = t["lr"] if name == "pi" else t["embed_norm_lr"]
        opt = torch.optim.AdamW(param_groups(cfg, model, name, lr), fused=True)
        ids = torch.randint(100, 30000, (1, L), device=DEV)
        times, losses = [], []
        for i in range(3):
            torch.cuda.synchronize()
            s = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nll, n = token_nll(model, ids, ids, t["logits_chunk"])
            loss = nll / n
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - s)
            losses.append(loss.item())
        tok_s = L / (sum(times[1:]) / len(times[1:]))
        peak = torch.cuda.max_memory_allocated() / 2**30
        hours = t["steps"] * t["grad_accum"] * L / tok_s / 3600
        est[name] = hours
        print(f"  {name}: {L} token sequence, losses {[round(x, 3) for x in losses]}, {tok_s:.0f} tok/s, peak {peak:.1f} GiB, about {hours:.1f} h per model at this config", flush=True)
        if not all(x == x for x in losses):
            fail(f"{name}: loss is NaN")
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        if peak > total * 0.95:
            warn(f"{name}: peak memory is close to the GPU limit")
        ok(f"{name}: full {L} token training step fits in memory")
        if name == "longlora":
            model.eval()
            save(cfg, "_smoke", model, cfg["base_model"])
            m = load_for_eval(cfg, str(model_dir(cfg, "_smoke") / "merged"))
            if (m.config.rope_scaling or {}).get("factor") != cfg["target_len"] / cfg["orig_len"]:
                fail(f"saved model lost its rope settings: {m.config.rope_scaling}")
            if getattr(m.config, "_attn_implementation", "sdpa") != "sdpa":
                fail("saved model kept the training attention")
            ok(f"LoRA merge, bf16 save and reload keep rope settings {m.config.rope_scaling}")
            del m
            shutil.rmtree(model_dir(cfg, "_smoke"), ignore_errors=True)
        del model, opt
        torch.cuda.empty_cache()
    full_h = est["pi"] * 3 + est["longlora"] * 2
    print(f"  estimated training time for the five stretched models: about {full_h:.1f} h, plus SFT, evals and uploads", flush=True)


FUNCS = {"sys": check_sys, "hub": check_hub, "data": check_data, "rope": check_rope, "s2": check_s2, "train": check_train}


def run_part(cfg, part):
    print(f"\nCHECK {part}", flush=True)
    FUNCS[part](cfg)
    print(f"CHECK {part} DONE", flush=True)
