from transformers import AutoConfig

METHOD_ROPE = {
    "none": "none",
    "pi": "pi",
    "ntk": "ntk",
    "yarn": "yarn",
    "dynamic": "dynamic",
    "longlora": "pi",
    "yarn_longlora": "yarn",
}


def factor(cfg):
    return cfg["target_len"] / cfg["orig_len"]


def head_dim(conf):
    return getattr(conf, "head_dim", None) or conf.hidden_size // conf.num_attention_heads


def apply(conf, cfg, method):
    kind = METHOD_ROPE[method]
    s = factor(cfg)
    orig = cfg["orig_len"]
    if kind == "none":
        return conf
    if kind == "pi":
        conf.rope_scaling = {"rope_type": "linear", "factor": s}
        conf.max_position_embeddings = cfg["target_len"]
    elif kind == "ntk":
        d = head_dim(conf)
        conf.rope_theta = float(conf.rope_theta) * s ** (d / (d - 2))
        conf.rope_scaling = None
        conf.max_position_embeddings = cfg["target_len"]
    elif kind == "yarn":
        conf.rope_scaling = {"rope_type": "yarn", "factor": s, "original_max_position_embeddings": orig}
        conf.max_position_embeddings = cfg["target_len"]
    elif kind == "dynamic":
        conf.rope_scaling = {"rope_type": "dynamic", "factor": s}
        conf.max_position_embeddings = orig
    return conf


def load_config(cfg, path, method=None):
    conf = AutoConfig.from_pretrained(path)
    if method:
        apply(conf, cfg, method)
    conf.use_cache = True
    return conf


def describe(conf):
    return {
        "rope_theta": getattr(conf, "rope_theta", None),
        "rope_scaling": getattr(conf, "rope_scaling", None),
        "max_position_embeddings": getattr(conf, "max_position_embeddings", None),
    }
