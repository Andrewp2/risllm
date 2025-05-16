#!/usr/bin/env python3

import math
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import trange, tqdm

def sample_prefixes(model, tokenizer, prompt, M, k, device, temperature=1.0):
    """
    Stage 1: generate M short prefixes of up to k tokens each,
    recording the total log-prob of each prefix.
    """
    prefixes = []
    # tokenize once
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    input_len = input_ids.shape[1]

    for _ in trange(M, desc="Sampling prefixes"):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=k,
                do_sample=True,
                top_p=1.0,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id
            )
        seq = out.sequences[0]
        scores = out.scores  # list of length = #generated tokens
        # accumulate log‐prob for each generated token
        lp = 0.0
        for token_id, score in zip(seq[input_len:].tolist(), scores):
            log_probs = torch.log_softmax(score, dim=-1)
            lp += log_probs[0, token_id].item()
        text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
        boundaries = [text.rfind(p) for p in [".", "!", "?"]]
        idx = max(boundaries)
        if idx >= 0:
            text = text[:idx+1]
        prefixes.append((text, lp))
    return prefixes


def weighted_reservoir_sampling(prefixes, N):
    """
    Stage 2: from the M (prefix, log-prob) pairs,
    compute normalized weights and perform weighted reservoir sampling of size N.
    """
    lps = [lp for (_, lp) in prefixes]
    max_lp = max(lps)
    # for numerical stability
    raw_ws = [math.exp(lp - max_lp) for lp in lps]
    total_w = sum(raw_ws)
    ws = [w / total_w for w in raw_ws]

    survivors = []
    W = 0.0
    for (pref, _), w in tqdm(zip(prefixes, ws), total=len(prefixes), desc="Reservoir sampling"):
        W += w
        if len(survivors) < N:
            survivors.append((pref, w))
        else:
            if random.random() < w / W:
                idx = random.randrange(N)
                survivors[idx] = (pref, w)
    return survivors


def generate_completions(model, tokenizer, prompt, survivors, max_new_tokens, temperature=1.0):
    """
    Stage 3: for each surviving prefix, generate the full completion.
    """
    completions = []
    device = next(model.parameters()).device

    for prefix, _ in tqdm(survivors, desc="Generating completions"):
        text_in = prompt + prefix
        inputs = tokenizer(text_in, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=1.0,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )
        gen = out[0]
        completion = tokenizer.decode(gen[inputs.input_ids.shape[1]:], skip_special_tokens=True)
        completions.append(completion)
    return completions


class RISampler:
    """High-level interface for three-stage RIS sampling."""
    def __init__(self, model_name, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()
        self.device = device or next(self.model.parameters()).device

    def sample(self, prompt, M=100, k=50, N=10, max_new_tokens=150, temperature=1.0):
        prefixes = sample_prefixes(self.model, self.tokenizer, prompt, M, k, self.device, temperature)
        survivors = weighted_reservoir_sampling(prefixes, N)
        return generate_completions(self.model, self.tokenizer, prompt, survivors, max_new_tokens, temperature)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RISampler: Resampled Importance Sampling for LLMs")
    parser.add_argument("--model", "-m",      type=str,   default="qwen/Qwen3-8B",    help="HuggingFace model name")
    parser.add_argument("--prompt", "-p",     type=str,   required=True,      help="Prompt text")
    parser.add_argument("--M", "-M",          type=int,   default=10,        help="# of prefixes to propose")
    parser.add_argument("--k", "-k",          type=int,   default=25,         help="max tokens per prefix")
    parser.add_argument("--N", "-N",          type=int,   default=3,         help="# of survivors")
    parser.add_argument("--stage3_k", "-s",   type=int,   default=250,        help="max tokens for final completion")
    parser.add_argument("--temperature", "-t",type=float, default=1.0,        help="sampling temperature")
    args = parser.parse_args()

    sampler = RISampler(args.model)
    outputs = sampler.sample(
        prompt=args.prompt,
        M=args.M,
        k=args.k,
        N=args.N,
        max_new_tokens=args.stage3_k,
        temperature=args.temperature
    )
    for i, out in enumerate(outputs, 1):
        print(f"\n=== Completion {i} ===\n{out}")
