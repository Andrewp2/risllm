Below is a **three-phase roadmap** that starts from a clean workstation with only uv, CUDA 12.x drivers, and an RTX 3090, and ends with empirical numbers that tell you whether a ReSTIR/CRIS-style *reservoir sampler* is a meaningful alternative to today’s top-p / nucleus / beam decoders.

---

## 1 Environment & Baseline

First create a fresh virtual-env (`uv venv`) and install a lean stack: Python 3.11, PyTorch 2.3 + cu118 wheels, Hugging Face Transformers v4.41, Datasets, and Evaluate. Pull the smallest open-weight model whose perplexity is already well documented-`mistralai/Mistral-7B-Instruct-v0.2` is a good fit for 24 GB of VRAM; if memory is tight, use a 4-bit quantised checkpoint via Bits-and-Bytes.

note: using `Llama-3.2-1B` instead, faster
actually should try a 7b model "7b is the minimum" https://x.com/kalomaze/status/1919559585213329857
should try Qwen 3 maybe?

Write a *fully explicit* autoregressive loop (`with torch.no_grad(): for t in range(max_len): …`) so you can intercept logits before any built-in sampling. Record two baselines:

* quality → per-token cross-entropy on WikiText-2 and AlpacaEval’s MT-Bench “single-shot” subset;
* speed → tokens / second for 512-token continuations at batch = 4.

These numbers are your reference curve.

---

## 2 Prototype a CRIS-Inspired Sampler

Implement a light version first, focusing on *within-step* reuse:

1. At each decoding step pull the vector of logits, apply temperature, and extract the top-K probabilities (say K = 64).
2. Store those K tokens, their probabilities p₀, and a running maximum weight w_max in a fixed-size reservoir (R ≈ 32). For the very first step every token is accepted.
3. At the next step compute new probabilities p₁ for the *same* tokens under the **new** context; define an importance weight `w = p₁ / p₀`.  Feed each candidate to Talbot-style *resampled importance sampling*: keep the token with probability `min(1, w / w_max)`; if accepted, update `w_max`.  Fill the reservoir back to capacity with fresh top-K candidates that were not previously stored.
4. Draw the next token from the reservoir with probability ∝ w.

That four-line core reproduces the unbiased weight formula from CRIS while avoiding any PDF you cannot evaluate-exactly the trick the paper formalises citeturn1search0.

Once that works and produces text, extend it to *cross-step* reuse: each token in the reservoir carries its own `p_prev`; when you move to step t + 1, you simply update weights with the new `p_curr` and repeat the accept-reject.  That mirrors “temporal reuse” in ReSTIR.  Because every update is ℴ(R) and R is ≲ 64, the CPU cost is negligible compared with one transformer forward pass.

---

## 3 Measure, Tune, and Stress-Test

Use exactly the same scripts as for the baseline but sweep three knobs: reservoir size (8, 16, 32), K (32, 64, 128), and update frequency (every step vs. every 2 steps).  Log perplexity, MT-Bench score, and throughput.  A result is interesting if **either** perplexity drops by ≥ 0.2 on WikiText-2 at the *same* speed **or** throughput rises by ≥ 25 % at *equal* perplexity.

For robustness run TruthfulQA and a 1 000-pair human preference A/B using your favourite annotation service; correlated artifacts will show up there if the sampler collapses to high-probability clichés.

Finally, profile VRAM and wall-clock in a flame graph; if the sampler’s Python loop takes more than ~5 % of total time, port it to a tiny CUDA kernel that updates the weight arrays in parallel-one kernel launch per step is still cheaper than another transformer call.

---

### Expected Effort and Resources

A single RTX 3090 can run all experiments in < 120 GPU-hours (WikiText-2 perplexity is light); calendar-time about ten days: two for baseline, three to code and debug the reservoir, three for the full grid search, and two to analyse and write everything down.

If you see an unmistakable speed-quality Pareto improvement, you have a result that is both novel and publishable; if not, you have a negative result that tells the community where the analogy breaks. Either way the experiment is worth doing.

### Why doesn't this seem to be working?

ReSTIR works primarily because we have millions of parallel, independent but slightly different versions of the same fundamental question. LLMs are serial, and don't have parallel runs occuring simultaneously. So maybe the whole idea is bunk.
