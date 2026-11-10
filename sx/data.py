import random
from pathlib import Path

import numpy as np

from .common import banner, read_jsonl, wpath, write_json, write_jsonl

PG19 = "emozilla/pg19"
LONGALPACA = "Yukang/LongAlpaca-12k"
SFT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def stream_books(split):
    from datasets import load_dataset

    for r in load_dataset(PG19, split=split, streaming=True):
        if r.get("text"):
            yield r["text"]


def train_chunks_path(cfg):
    return wpath(cfg, "data", f"pg19_train_{cfg['train']['seq_len']}.npy")


def n_chunks(cfg):
    t = cfg["train"]
    return t["steps"] * t["grad_accum"]


def build_train_chunks(cfg, tok):
    path = train_chunks_path(cfg)
    need = n_chunks(cfg)
    L = cfg["train"]["seq_len"]
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] >= need and arr.shape[1] == L:
            return arr
    banner(f"tokenizing PG19 train books into {need} chunks of {L} tokens")
    buf, chunks, books = [], [], 0
    for text in stream_books("train"):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        buf.extend([tok.bos_token_id] + ids + [tok.eos_token_id])
        books += 1
        while len(buf) >= L and len(chunks) < need:
            chunks.append(np.asarray(buf[:L], dtype=np.uint16))
            buf = buf[L:]
        if books % 25 == 0:
            print(f"  {books} books, {len(chunks)} of {need} chunks", flush=True)
        if len(chunks) >= need:
            break
    arr = np.stack(chunks)
    rng = np.random.default_rng(cfg["seed"])
    arr = arr[rng.permutation(len(arr))]
    np.save(path, arr)
    write_json(wpath(cfg, "data", "train_stats.json"), {"books": books, "chunks": len(arr), "seq_len": L, "tokens": int(arr.size)})
    return np.load(path, mmap_mode="r")


def eval_books(cfg, tok=None):
    d = wpath(cfg, "data", "eval_books")
    need_len = max(cfg["eval"]["lengths"])
    n = cfg["eval"]["ppl_docs"] + cfg["eval"]["haystack_books"]
    files = sorted(d.glob("book_*.npy")) if d.exists() else []
    if len(files) >= n:
        return [np.load(f) for f in files[:n]]
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for text in stream_books("test"):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= need_len + 64:
            arr = np.asarray(ids, dtype=np.int32)
            np.save(d / f"book_{len(out):03d}.npy", arr)
            out.append(arr)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} PG19 test books have {need_len} tokens, need {n}")
    return out


def ppl_docs(cfg):
    return eval_books(cfg)[: cfg["eval"]["ppl_docs"]]


def haystack(cfg):
    books = eval_books(cfg)[cfg["eval"]["ppl_docs"] :]
    return np.concatenate(books)


def longalpaca(cfg, tok):
    tr, ho = wpath(cfg, "data", "longalpaca_train.jsonl"), wpath(cfg, "data", "longalpaca_heldout.jsonl")
    if tr.exists() and ho.exists():
        return read_jsonl(tr), read_jsonl(ho)
    from datasets import load_dataset

    ds = load_dataset(LONGALPACA, split="train")
    s = cfg["sft"]
    rows = []
    for i, r in enumerate(ds):
        inst = (r.get("instruction") or "").strip()
        extra = (r.get("input") or "").strip()
        out = (r.get("output") or "").strip()
        if not inst or not out:
            continue
        prompt = SFT_TEMPLATE.format(instruction=inst + ("\n\n" + extra if extra else ""))
        np_ = len(tok(prompt, add_special_tokens=False)["input_ids"])
        no = len(tok(out, add_special_tokens=False)["input_ids"])
        rows.append({"id": i, "prompt": prompt, "output": out, "prompt_tokens": np_, "output_tokens": no})
    rng = random.Random(cfg["seed"])
    rng.shuffle(rows)
    limit = cfg["train"]["seq_len"]
    eval_max = max(cfg["eval"]["lengths"])
    fit = [r for r in rows if r["prompt_tokens"] + r["output_tokens"] + 2 <= limit]
    held = [r for r in fit if r["prompt_tokens"] + r["output_tokens"] + 2 <= eval_max and r["prompt_tokens"] > cfg["orig_len"]][: s["heldout_n"]]
    held_ids = {r["id"] for r in held}
    train = [r for r in fit if r["id"] not in held_ids][: s["train_n"]]
    write_jsonl(tr, train)
    write_jsonl(ho, held)
    return train, held


def run_factory(cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    arr = build_train_chunks(cfg, tok)
    books = eval_books(cfg, tok)
    train, held = longalpaca(cfg, tok)
    stats = {
        "train_chunks": int(arr.shape[0]),
        "seq_len": int(arr.shape[1]),
        "train_tokens": int(arr.shape[0] * arr.shape[1]),
        "eval_books": len(books),
        "longalpaca_train": len(train),
        "longalpaca_heldout": len(held),
        "longalpaca_mean_tokens": sum(r["prompt_tokens"] + r["output_tokens"] for r in train) / max(1, len(train)),
    }
    write_json(wpath(cfg, "data", "stats.json"), stats)
    banner("factory stats\n" + "\n".join(f"{k}: {v}" for k, v in stats.items()))
    if stats["longalpaca_train"] < 50 or stats["longalpaca_heldout"] < 5:
        raise RuntimeError(f"too little LongAlpaca data for seq_len {cfg['train']['seq_len']}: {stats}")
    push_dataset(cfg, stats)


def dataset_card(cfg, stats):
    return f"""
# Stretch32K Data ({cfg['base_label']})

Data used to stretch `{cfg['base_model']}` from {cfg['orig_len']} to {cfg['target_len']} tokens of context.

<table>
<tr><th>Config</th><th>Rows</th><th>What it holds</th></tr>
<tr><td>pg19_{stats['seq_len']}</td><td>{stats['train_chunks']}</td><td>Packed chunks of {stats['seq_len']} token ids from PG19 train books, shared by every continued pretraining model ({stats['train_tokens']:,} tokens)</td></tr>
<tr><td>longalpaca_sft</td><td>{stats['longalpaca_train']}</td><td>LongAlpaca long document instructions used for the SFT model</td></tr>
<tr><td>longalpaca_heldout</td><td>{stats['longalpaca_heldout']}</td><td>Held out LongAlpaca examples used to score answer loss</td></tr>
</table>

Token ids use the `{cfg['base_model']}` tokenizer. Sources: `{PG19}` (public domain books) and `{LONGALPACA}`.
"""


def push_dataset(cfg, stats):
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    from .common import dataset_repo
    from .hub import ensure_collection, login

    login()
    api = HfApi()
    repo = dataset_repo(cfg)
    api.create_repo(repo, repo_type="dataset", private=cfg["private"], exist_ok=True)
    arr = np.load(train_chunks_path(cfg), mmap_mode="r")
    name = f"pg19_{stats['seq_len']}"

    def gen():
        for i in range(arr.shape[0]):
            yield {"input_ids": arr[i].astype(np.int32).tolist()}

    banner(f"pushing {name}: {arr.shape[0]} rows to {repo}")
    Dataset.from_generator(gen).push_to_hub(repo, config_name=name, private=cfg["private"], commit_message=f"Add {name}")
    for cname, fname in [("longalpaca_sft", "longalpaca_train"), ("longalpaca_heldout", "longalpaca_heldout")]:
        Dataset.from_list(read_jsonl(wpath(cfg, "data", f"{fname}.jsonl"))).push_to_hub(repo, config_name=cname, private=cfg["private"], commit_message=f"Add {cname}")
    readme = Path(hf_hub_download(repo, "README.md", repo_type="dataset", force_download=True)).read_text()
    front = readme[: readme.find("\n---", 3) + 4] if readme.startswith("---") else ""
    api.upload_file(path_or_fileobj=(front + "\n" + dataset_card(cfg, stats)).encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset", commit_message="Update dataset card")
    slug = ensure_collection(cfg)
    api.add_collection_item(slug, repo, "dataset", note="Training data shared by every model in this collection", exists_ok=True)
    for cname, key in [(name, "train_chunks"), ("longalpaca_sft", "longalpaca_train"), ("longalpaca_heldout", "longalpaca_heldout")]:
        n = len(load_dataset(repo, cname, split="train", download_mode="force_redownload"))
        if n != stats[key]:
            raise RuntimeError(f"{cname}: hub has {n} rows, expected {stats[key]}")
        print(f"{cname}: {n} rows OK", flush=True)
