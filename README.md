# Resampled Importance Sampling for Large Language Models

## Authors
Andrew Peterson

## Main Idea

Current Large Language Models use naive versions of sampling and search where an assumption is made that different branches
of the same tree of thought are completely independent.

To make the problem apparent, consider the following mate-in-5 chess puzzle:

```
8/2Nb4/pp6/4rp1p/1Pp1pPkP/PpPpR3/1B1P2N1/1K6 w - - 0 1
```

![Chess Puzzle](image.png)

Even on max settings on lichess.org with Stockfish 16 (NNUE) it struggles at first to find a guaranteed mate, and then even more to find the mate in 5. Dumber chess engines can find the mate quickly because they perform a dumber, more brute-force search, and Stockfish 17 can find it realtively quickly based on having a larger NNUE.

Example on how mixing at different levels can produce better outcome

Question: "What is the capital of France?"

Completion 1: "**The largest city in France is the capital.** _Marseille is the largest city in France._ Marseille is the capital of France."

Completion 2: "_The largest city in France is not the capital._ **Paris is the largest city in France.** Marseille is the capital of France."

Correct completion: "Usually the largest city is the capital. Paris is the biggest city in France. Paris is the capital of France."

[1] Tree of Thought https://arxiv.org/abs/2305.10601
