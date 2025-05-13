import math, random, time, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from reservoir_sampler import ReservoirLogitsProcessor
from tqdm import tqdm

def self_ppl(model, tok, texts, use_reservoir):
    proc = ReservoirLogitsProcessor(K=64, R=32, device=model.device) if use_reservoir else None
    nll = tok_cnt = 0
    for p in tqdm(texts, leave=False):
        enc  = tok(p, return_tensors="pt", add_special_tokens=False)
        ids  = enc.input_ids.to(model.device)
        attn = enc.attention_mask.to(model.device)

        with torch.no_grad():
            gen = model.generate(
                ids, attention_mask=attn,
                max_new_tokens=64,
                logits_processor=[proc] if proc else None,
                do_sample=True, top_p=1.0, temperature=1.0, use_cache=True
            )[0]

            logits = model(gen[:-1].unsqueeze(0), attention_mask=attn).logits
            logp   = torch.log_softmax(logits, -1)
            tgt    = gen[1:]
            nll   += (-logp.gather(2, tgt.unsqueeze(0).unsqueeze(-1)).squeeze()).sum().item()
            tok_cnt += tgt.numel()
    return math.exp(nll / tok_cnt)


def main():
    model_id = "meta-llama/Llama-3.2-1B"
    tok = AutoTokenizer.from_pretrained(model_id, token=True, use_fast=False)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=True, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [s["text"] for s in ds if s["text"].strip()]
    random.seed(0)
    prompts = random.sample(texts, 256)

    t0 = time.time()
    ppl_base = self_ppl(model, tok, prompts, False)
    t1 = time.time()
    ppl_res = self_ppl(model, tok, prompts, True)
    t2 = time.time()

    print(f"top-p sampling  self-perplexity: {ppl_base:.2f}   ({t1 - t0:.1f}s)")
    print(f"reservoir       self-perplexity: {ppl_res:.2f}   ({t2 - t1:.1f}s)")
    print(f"reservoir / baseline = {ppl_res / ppl_base:.3f}")

if __name__ == "__main__":
    main()
