import time, torch, math
from transformers import AutoTokenizer, AutoModelForCausalLM
from reservoir_sampler import ReservoirLogitsProcessor

def main():
    model_id = "meta-llama/Llama-3.2-1B"
    tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=True, use_fast=False)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    batch, n = 4, 512
    ids = tok(["Hello world"] * batch, return_tensors="pt", padding=True).to(model.device)
    proc = ReservoirLogitsProcessor(K=64, R=32, device=model.device)
    _ = model.generate(**ids, max_new_tokens=4, logits_processor=[proc], do_sample=True, temperature=1.0, top_p=1.0)
    t0 = time.time()
    _ = model.generate(**ids, max_new_tokens=n, logits_processor=[proc], do_sample=True, temperature=1.0, top_p=1.0, use_cache=True)
    dt = time.time() - t0
    print(f"{batch*n/dt:.1f} tokens/sec")

if __name__ == "__main__":
    main()
