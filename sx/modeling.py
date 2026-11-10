import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def core(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model, m.lm_head


def _chunk_nll(h, w, t):
    logits = F.linear(h, w).float()
    return torch.logsumexp(logits, dim=-1) - logits.gather(-1, t.clamp(min=0).unsqueeze(-1)).squeeze(-1)


def token_nll(model, ids, labels, chunk=1024):
    body, head = core(model)
    hidden = body(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    parts = []
    for s in range(0, hidden.size(1), chunk):
        h, t = hidden[:, s : s + chunk], targets[:, s : s + chunk]
        if torch.is_grad_enabled():
            parts.append(checkpoint(_chunk_nll, h, head.weight, t, use_reentrant=False))
        else:
            parts.append(_chunk_nll(h, head.weight, t))
    nll = torch.cat(parts, dim=1)
    mask = (targets != -100).to(nll.dtype)
    return (nll * mask).sum(), mask.sum()


def load_for_eval(cfg, path, method=None):
    from transformers import AutoModelForCausalLM

    from .rope import load_config

    conf = load_config(cfg, path, method)
    model = AutoModelForCausalLM.from_pretrained(path, config=conf, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    return model
