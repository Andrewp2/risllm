import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def measure_generate(batch_size: int = 4, max_new_tokens: int = 512) -> None:
    """
    Measures tokens/sec by calling model.generate() once for the whole sequence.
    """
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    # Prepare the same "Hello world" prompt repeated
    prompts = ["Hello world"] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # Warm up (optional but more stable timing)
    _ = model.generate(**inputs, max_new_tokens=4)

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # greedy; closest to your arg‑max loop
        use_cache=True,            # reuse past_key_values
        return_dict_in_generate=True,
    )
    elapsed = time.time() - start

    tokens_generated = batch_size * max_new_tokens
    print(f"generate(): {tokens_generated/elapsed:.1f} tokens/sec")

def main():
    measure_generate()

if __name__ == "__main__":
    main()
