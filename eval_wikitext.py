import time, math, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    total_nll, total_tokens = 0.0, 0
    start = time.time()

    for sample in ds:
        text = sample["text"]
        if not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt")
        ids = enc.input_ids.to(model.device)
        if ids.size(1) < 2:
            continue
        with torch.no_grad():
            loss = model(ids[:, :-1], labels=ids[:, 1:]).loss
        n = ids.size(1) - 1
        total_nll += loss.item() * n
        total_tokens += n

    elapsed = time.time() - start
    print(f"WikiText-2 cross-entropy: {total_nll / total_tokens:.4f}")
    print(f"Throughput: {total_tokens / elapsed:.1f} tokens/sec")

if __name__ == "__main__":
    main()
