# grid_search.py – grid search with nice tqdm progress bars

import math, time, itertools, random, json, argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from reservoir_sampler import ReservoirLogitsProcessor

# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------

def self_ppl(model, tok, prompts, K, R):
    """Compute *self‑perplexity* of the model when decoded with the
    ReservoirLogitsProcessor(K, R).

    A tqdm progress‑bar shows how many prompts have been evaluated so far so
    you know roughly how long the run will take.
    """
    proc = ReservoirLogitsProcessor(K=K, R=R, device=model.device)
    nll = tok_cnt = 0

    # inner loop progress bar – one iteration per prompt -----------------------
    for p in tqdm(prompts, desc=f"PPL  K={K} R={R}", leave=False):
        enc = tok(p, return_tensors="pt", add_special_tokens=False)
        ids, attn = enc.input_ids.to(model.device), enc.attention_mask.to(model.device)
        with torch.no_grad():
            gen = model.generate(
                ids,
                attention_mask=attn,
                max_new_tokens=64,
                logits_processor=[proc],
                do_sample=True,
                top_p=1.0,
                temperature=1.0,
                use_cache=True,
            )[0]
            logits = model(gen[:-1].unsqueeze(0), attention_mask=attn).logits
            logp = torch.log_softmax(logits, -1)
            tgt = gen[1:]
            nll += (-logp.gather(2, tgt.unsqueeze(0).unsqueeze(-1)).squeeze()).sum().item()
            tok_cnt += tgt.numel()
    return math.exp(nll / tok_cnt)


def throughput(model, tok, K, R, warmup=4, seq_len=512, batch=4):
    """Measure tokens / second for a fixed batch & sequence length."""
    inputs = tok(["Hello world"] * batch, return_tensors="pt", padding=True).to(model.device)
    proc = ReservoirLogitsProcessor(K=K, R=R, device=model.device)
    with torch.no_grad():
        # warm‑up
        model.generate(**inputs, max_new_tokens=warmup, logits_processor=[proc], do_sample=True)
        t0 = time.time()
        model.generate(
            **inputs,
            max_new_tokens=seq_len,
            logits_processor=[proc],
            do_sample=True,
        )
    return batch * seq_len / (time.time() - t0)

# -----------------------------------------------------------------------------
# main grid search
# -----------------------------------------------------------------------------

def run_grid(args):
    K_vals = args.K or [64]
    R_vals = args.R or [8, 16, 32, 64]

    model_id = args.model
    tok = AutoTokenizer.from_pretrained(model_id, token=True, use_fast=False)
    tok.pad_token = tok.eos_token   # silence HF "pad_token_id" warning
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=True, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    model.generation_config.pad_token_id = tok.pad_token_id

    # prompts ------------------------------------------------------------------
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [s["text"] for s in ds if s["text"].strip()]
    random.seed(0)
    prompts = random.sample(texts, args.n_prompts)

    results = []

    combos = list(itertools.product(K_vals, R_vals))
    outer_bar = tqdm(combos, desc="Grid search", unit="combo")

    for K, R in outer_bar:
        outer_bar.set_postfix(K=K, R=R)
        start = time.time()
        ppl = self_ppl(model, tok, prompts, K, R)
        ppl_time = time.time() - start
        tps = throughput(model, tok, K, R)
        results.append({"K": K, "R": R, "ppl": ppl, "ppl_time": ppl_time, "tok_per_sec": tps})
        print(
            f"K={K:3d}  R={R:3d}   perplexity {ppl:6.2f}   (eval {ppl_time:4.1f}s)   "
            f"speed {tps:6.1f} tok/s"
        )

    # sort by perplexity asc then speed desc -----------------------------------
    best = sorted(results, key=lambda d: (d["ppl"], -d["tok_per_sec"]))[0]
    print("\n=== best ===")
    print(json.dumps(best, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n⇢ wrote raw results to {args.out}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Grid‑search Reservoir K/R hyper‑params")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--K", type=int, nargs="*", help="list of K values (top‑k proposals)")
    p.add_argument("--R", type=int, nargs="*", help="list of R values (reservoir sizes)")
    p.add_argument("--n_prompts", type=int, default=64, help="number of WikiText prompts to sample")
    p.add_argument("--out", type=str, help="optional JSON path to save results")
    run_grid(p.parse_args())
