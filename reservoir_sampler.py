import torch
from transformers import LogitsProcessor


class ReservoirLogitsProcessor(LogitsProcessor):
    """Vectorised CRIS‑style reservoir sampler with cross‑step reuse.

    **Key features**
    ---------------
    * Fully‑vectorised acceptance test & refill – no Python loops over batch rows.
    * Safe duplicate‑mask construction – works even when a batch row keeps **zero** tokens.
    * `reset()` method to clear internal state so the same object can be reused for
      multiple *independent* decoding runs (e.g. across prompts during grid‑search).
    """

    def __init__(self, K: int = 64, R: int = 32, device=None):
        super().__init__()
        self.K, self.R = K, R
        self.device = device
        self.reset()

    # ------------------------------------------------------------------
    # public helper -----------------------------------------------------
    # ------------------------------------------------------------------
    def reset(self):
        """Clear the reservoir so the processor can be reused safely.

        Call this **between independent decoding runs** – for instance, once
        per prompt inside a loop.  (If you create a *fresh* processor per
        prompt you don't need this.)
        """
        self.tokens = None  # (B, R) long – token ids in reservoir
        self.p_prev = None  # (B, R) float – probs from previous step
        self.w_max = None   # (B,)   float – max importance weight per batch

    # ------------------------------------------------------------------
    # private helpers ---------------------------------------------------
    # ------------------------------------------------------------------
    def _allocate_state(self, probs: torch.Tensor):
        """Initialise reservoir on the very first decoding step."""
        topk_p, topk_tok = probs.topk(self.K, dim=-1)          # (B, K)
        self.tokens = topk_tok[:, : self.R].clone()            # (B, R)
        self.p_prev = topk_p[:, : self.R].clone()              # (B, R)
        self.w_max = torch.ones(probs.size(0), device=probs.device)

    # ------------------------------------------------------------------
    # main hook ---------------------------------------------------------
    # ------------------------------------------------------------------
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.device is None:
            self.device = scores.device

        probs = torch.softmax(scores, dim=-1)                  # (B, V)
        B, V = probs.shape

        # first step -----------------------------------------------------
        if self.tokens is None:
            self._allocate_state(probs)

        # 1) importance weights & accept/reject --------------------------
        p_curr = probs.gather(1, self.tokens)                   # (B, R)
        w = p_curr / (self.p_prev + 1e-30)                      # (B, R)

        accept_prob = (w / self.w_max.unsqueeze(1)).clamp(max=1.0)
        keep_mask = torch.rand_like(accept_prob).lt(accept_prob)  # (B, R) bool

        # 2) build new reservoir tensors -------------------------------
        new_tokens = torch.full_like(self.tokens, -1)           # (B, R)
        new_p_prev = torch.zeros_like(self.p_prev)              # (B, R)

        if keep_mask.any():
            rows, cols = keep_mask.nonzero(as_tuple=True)
            new_tokens[rows, cols] = self.tokens[rows, cols]
            new_p_prev[rows, cols] = p_curr[rows, cols]

        # 3) duplicate mask – safe even if a row kept zero tokens -------
        sentinel_mask = new_tokens.eq(-1)
        idx = new_tokens.clamp(min=0)                           # replace -1 → 0
        have_mask = torch.zeros((B, V), dtype=torch.bool, device=self.device)
        have_mask.scatter_(1, idx, ~sentinel_mask)

        # 4) refill from top‑K proposals -------------------------------
        topk_p, topk_tok = probs.topk(self.K, dim=-1)           # (B, K)
        for k in range(self.K):
            need_rows = (new_tokens == -1).any(1)               # rows still with holes
            if not need_rows.any():
                break

            cand_tok = topk_tok[need_rows, k]                  # (N,)
            cand_p = topk_p[need_rows, k]
            is_dup = have_mask[need_rows, cand_tok]

            if (~is_dup).any():
                rows = need_rows.nonzero(as_tuple=False).squeeze(1)[~is_dup]
                toks = cand_tok[~is_dup]
                ps = cand_p[~is_dup]

                slot = (new_tokens[rows] == -1).float().argmax(dim=1)
                new_tokens[rows, slot] = toks
                new_p_prev[rows, slot] = ps
                have_mask[rows, toks] = True

        # 5) pad any remaining holes with EOS(0) and tiny prob ----------
        unfilled = new_tokens.eq(-1)
        if unfilled.any():
            new_tokens[unfilled] = 0
            new_p_prev[unfilled] = 1e-9

        # 6) update persistent state ------------------------------------
        self.tokens = new_tokens
        self.p_prev = new_p_prev
        w_now = probs.gather(1, self.tokens) / (self.p_prev + 1e-30)
        self.w_max = w_now.max(dim=1).values.detach()

        # 7) project reservoir back to logits ---------------------------
        new_scores = torch.full_like(scores, float('-inf'))
        new_scores.scatter_(1, self.tokens, torch.log(probs.gather(1, self.tokens) + 1e-30))
        # Use the current importance weights for sampling
        # new_scores.scatter_(1, self.tokens, torch.log(w + 1e-30))
        return new_scores
