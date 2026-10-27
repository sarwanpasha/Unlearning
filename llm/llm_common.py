"""
Shared machinery for the language-model experiments: data encoding, LoRA model
construction, losses, exact Hessian-vector products on the LoRA parameters and
a matrix-free conjugate-gradient solver.

All unlearning and the default relearning adversary act in the LoRA subspace,
so curvature quantities are exact Hessian-vector products over a few million
parameters rather than diagonal approximations.
"""
import json, math, os, random
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------- data
def tofu(split):
    from datasets import load_dataset
    return load_dataset("locuslab/TOFU", split)["train"]


def encode(tok, rows, maxlen=192):
    """Question/answer rows -> input_ids with the prompt masked out of the loss."""
    ids, labs = [], []
    for r in rows:
        q = f"Question: {r['question']}\nAnswer:"
        a = " " + r["answer"]
        qi = tok(q, add_special_tokens=False)["input_ids"]
        ai = tok(a, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        x = (qi + ai)[:maxlen]
        y = ([-100] * len(qi) + ai)[:maxlen]
        ids.append(x); labs.append(y)
    return ids, labs


def batches(ids, labs, bs, pad, shuffle=True, seed=0):
    idx = list(range(len(ids)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        n = max(len(ids[j]) for j in sel)
        x = torch.full((len(sel), n), pad, dtype=torch.long)
        y = torch.full((len(sel), n), -100, dtype=torch.long)
        m = torch.zeros((len(sel), n), dtype=torch.long)
        for k, j in enumerate(sel):
            L = len(ids[j])
            x[k, :L] = torch.tensor(ids[j]); y[k, :L] = torch.tensor(labs[j]); m[k, :L] = 1
        yield x.to(DEV), y.to(DEV), m.to(DEV)


# --------------------------------------------------------------- model
def build(model_name, r=16, alpha=32, dropout=0.0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # eager attention: the fused SDPA kernels have no double-backward, which the
    # Hessian-vector products need.
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32,
                                             attn_implementation="eager").to(DEV)
    m.config.use_cache = False
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(m, cfg)
    return m, tok


def lora_params(m):
    return [p for n, p in m.named_parameters() if p.requires_grad]


def flat(vs):
    return torch.cat([v.reshape(-1) for v in vs])


def unflat(x, ref):
    out, i = [], 0
    for p in ref:
        n = p.numel(); out.append(x[i:i + n].view_as(p)); i += n
    return out


@torch.no_grad()
def add_(ps, vs, scale=1.0):
    for p, v in zip(ps, vs):
        p.add_(v, alpha=scale)


# --------------------------------------------------------------- losses
def nll(m, ids, labs, bs, pad, cap=None, seed=0):
    """Mean token NLL over a dataset (no grad)."""
    m.eval(); tot = n = 0.0
    with torch.no_grad():
        for i, (x, y, msk) in enumerate(batches(ids, labs, bs, pad, False, seed)):
            if cap is not None and i >= cap:
                break
            out = m(input_ids=x, attention_mask=msk).logits[:, :-1]
            t = y[:, 1:]
            sel = t != -100
            l = F.cross_entropy(out[sel], t[sel], reduction="sum")
            tot += l.item(); n += sel.sum().item()
    m.train()
    return tot / max(n, 1)


def loss_on(m, batch):
    x, y, msk = batch
    out = m(input_ids=x, attention_mask=msk).logits[:, :-1]
    t = y[:, 1:]
    sel = t != -100
    return F.cross_entropy(out[sel], t[sel])


def grad_of(m, batch, ps, create_graph=False):
    l = loss_on(m, batch)
    g = torch.autograd.grad(l, ps, create_graph=create_graph)
    return l.detach(), g


def hvp(m, batch, ps, vec):
    """Exact Hessian-vector product of the batch loss w.r.t. the LoRA params."""
    l = loss_on(m, batch)
    g = torch.autograd.grad(l, ps, create_graph=True)
    dot = sum((gi * vi).sum() for gi, vi in zip(g, vec))
    h = torch.autograd.grad(dot, ps)
    return [hi.detach() for hi in h]


def cg(matvec, b, iters=25, damp=1e-3, tol=1e-10):
    """Conjugate gradients for (H + damp I) x = b, matrix-free."""
    x = [torch.zeros_like(t) for t in b]
    r = [t.clone() for t in b]
    p = [t.clone() for t in r]
    rs = sum((t * t).sum() for t in r)
    for _ in range(iters):
        Ap = matvec(p)
        Ap = [a + damp * pi for a, pi in zip(Ap, p)]
        al = rs / (sum((pi * ai).sum() for pi, ai in zip(p, Ap)) + 1e-30)
        x = [xi + al * pi for xi, pi in zip(x, p)]
        r = [ri - al * ai for ri, ai in zip(r, Ap)]
        rs2 = sum((t * t).sum() for t in r)
        if rs2 < tol:
            break
        p = [ri + (rs2 / rs) * pi for ri, pi in zip(r, p)]
        rs = rs2
    return x


def seq_logprob(m, batch):
    """Per-example summed log-probability of the answer tokens, shape [B].

    NPO is defined on the sequence-level likelihood ratio, and the distinction
    matters: with a batch-mean scalar the NPO gradient is a constant multiple of
    the gradient-ascent gradient, which AdamW normalises away, making the two
    baselines numerically identical.
    """
    x, y, msk = batch
    out = m(input_ids=x, attention_mask=msk).logits[:, :-1]
    t = y[:, 1:]
    lp = torch.log_softmax(out, dim=-1)
    sel = t != -100
    tok = torch.gather(lp, 2, t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok * sel).sum(dim=1)
