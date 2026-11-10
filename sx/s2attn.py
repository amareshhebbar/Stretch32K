import torch
import torch.nn.functional as F

NAME = "s2_sdpa"
GROUP = {"size": 0}


def _sdpa(q, k, v, scale):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def _grouped(q, k, v, g, scale):
    b, h, n, d = q.shape
    ng = n // g

    def split(x):
        return x.reshape(b, h, ng, g, d).transpose(1, 2).reshape(b * ng, h, g, d)

    o = _sdpa(split(q), split(k), split(v), scale)
    return o.reshape(b, ng, h, g, d).transpose(1, 2).reshape(b, h, n, d)


def s2_core(q, k, v, g, scale=None):
    b, h, n, d = q.shape
    if g <= 0 or n <= g or n % g or g % 2 or h < 2:
        return _sdpa(q, k, v, scale)
    half, s = h // 2, g // 2
    o1 = _grouped(q[:, :half], k[:, :half], v[:, :half], g, scale)
    q2, k2, v2 = q[:, half:], k[:, half:], v[:, half:]
    head = _sdpa(q2[:, :, :s], k2[:, :, :s], v2[:, :, :s], scale)
    mid = _grouped(q2[:, :, s : n - s], k2[:, :, s : n - s], v2[:, :, s : n - s], g, scale)
    tail = _sdpa(q2[:, :, n - s :], k2[:, :, n - s :], v2[:, :, n - s :], scale)
    return torch.cat([o1, torch.cat([head, mid, tail], dim=2)], dim=1)


def repeat_kv(x, rep):
    if rep == 1:
        return x
    b, h, n, d = x.shape
    return x[:, :, None].expand(b, h, rep, n, d).reshape(b, h * rep, n, d)


def s2_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    rep = query.shape[1] // key.shape[1]
    o = s2_core(query, repeat_kv(key, rep), repeat_kv(value, rep), GROUP["size"], scaling)
    return o.transpose(1, 2).contiguous(), None


def register(group):
    from transformers import AttentionInterface

    GROUP["size"] = int(group)
    AttentionInterface.register(NAME, s2_attention)
    try:
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

        AttentionMaskInterface.register(NAME, sdpa_mask)
    except Exception:
        pass
    return NAME
