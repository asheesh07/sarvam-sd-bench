# sarvam-sd-bench

Speculative decoding benchmark: Sarvam-1 draft → Sarvam-30B target.

---

## Results

```
Draft  : Sarvam-1 (3B, 4-bit quantized)
Target : Sarvam-30B (via API)
Prompts: 30 Hindi prompts × 5 K values = 150 benchmark runs
GPU    : Kaggle P100 (16GB)
```

**Best at K=1**

```
α (acceptance rate) = 0.73
Speedup             = 1.8× over always-calling-target baseline
Expected latency    = 126ms vs 228ms per token
```

### Benchmark Table

| K  | α overall | Speedup | α general | α technical (10 prompts) | α mixed |
|----|-----------|---------|-----------|--------------------------|---------|
| 1  | 0.73      | 1.80×   | 0.70      | 1.00                     | 0.50    |
| 3  | 0.60      | 1.58×   | 0.30      | 1.00                     | 0.50    |
| 5  | 0.47      | 1.40×   | 0.50      | 0.50                     | 0.40    |
| 7  | 0.57      | 1.53×   | 0.50      | 0.80                     | 0.40    |
| 10 | 0.80      | 1.95×   | 0.80      | 0.80                     | 0.80    |

Speedup formula (output-level SD):
```
speedup = api_ms / (α × draft_per_token_ms + (1−α) × api_ms)
```

This is not the standard token-level SD formula. The standard formula assumes
the target runs one forward pass over K tokens. Here the target is an API call —
fixed cost regardless of K. The formula above reflects that.

---

## Key Finding — Tokenizer Mismatch

Standard token-level SD does not work between Sarvam-1 and Sarvam-30B.

```
Sarvam-1   : 68,096 vocab  (LlamaTokenizerFast)
Sarvam-30B : 262,144 vocab (different tokenizer, different BOS token)
```

Same Hindi sentence, two different tokenizers, completely different token IDs.
Token-level SD requires the draft and target to agree on what a token is.
They don't.

Possible directions: shared tokenizer across the model family, or a cross-vocab
logit projection layer mapping the 68K → 262K space. Both are non-trivial.

This benchmark uses output-level verification as a proxy until that exists.

---

## Domain Analysis

Acceptance rate at K=1:

```
technical Hindi (10 prompts) : 1.00
general Hindi   (10 prompts) : 0.70
mixed prompts   (10 prompts) : 0.50
```

Technical Hindi (ML, infrastructure, API terms) hits α=1.00 at K=1.
One hypothesis: technical prompts have more constrained continuations,
so the small model and large model converge on the same next token more often.
Worth investigating with a larger prompt set before drawing strong conclusions.

---

## Timing

```
Draft inference (Sarvam-1, 4-bit, P100) : 88.8ms per token
Target API (Sarvam-30B, batch=5)        : 227.8ms per prompt
Verification overhead (vs API latency)   : 0.81×
Avg total sampling time                  : 648ms
```

Batch verification sends 5 prompts per API call instead of 1.
Per-prompt API cost dropped from ~1148ms to ~228ms.
Without batching, speedup stays below 1×.

---

## Hardware

```
Draft : Sarvam-1 (3B, 4-bit BitsAndBytes)
GPU   : Kaggle P100 (16GB VRAM)
VRAM  : 1.73GB after loading draft
```

Sarvam-30B needs ~17GB in 4-bit — doesn't fit on a P100.
Target runs via the Sarvam API (`sarvam-30b` endpoint).
That's also why output-level verification is used instead of token-level.

---

## The Tool

`sd_bench.py` — single file, no framework dependencies.

```
sd_bench.py
├── load_draft_model()      — 4-bit BnB, device_map=auto
├── generate_draft()        — local K-token generation
├── sarvam_batch_verify()   — N prompts per API call
├── compute_speedup()       — output-level SD formula
└── save_summary()          — report.csv + summary.csv
```

---

## Install

```bash
git clone https://github.com/asheesh07/sarvam-sd-bench
cd sarvam-sd-bench
pip install -r requirements.txt
```

---

## Run

```bash
# minimal
python sd_bench.py --api-key YOUR_SARVAM_KEY

# full
python sd_bench.py \
  --draft      sarvamai/sarvam-1 \
  --target     sarvam-30b \
  --prompts    prompts.txt \
  --k-values   1 3 5 7 10 \
  --batch-size 5 \
  --output     report.csv \
  --api-key    YOUR_SARVAM_KEY

# Kaggle (add SARVAM_API_KEY to Secrets first)
python sd_bench.py --skip-confirm
```

---

## Prompt File Format

```
# tab-separated: <category>\t<prompt>
general     आज मौसम बहुत अच्छा है
technical   ट्रांसफॉर्मर आर्किटेक्चर में अटेंशन मैकेनिज्म
mixed       इस API को integrate करना आसान है
```

Domain shift is computed between `general` and `technical` if both exist.

---

## Dataset

```
30 Hindi prompts across 3 domains:

general   (10) — conversational everyday Hindi
technical (10) — ML, infrastructure, API, cloud terms
mixed     (10) — Hindi sentences with embedded English

× 5 K values [1, 3, 5, 7, 10] = 150 benchmark runs
```

---

## Output Files

```
report.csv         — raw results, 150 rows
report_summary.csv — per-K aggregates
```

Columns: `prompt, K, category, draft_continuation, accepted,
draft_forward_time_ms, target_forward_time_ms, verification_overhead, sampling_time_ms`

---

## Repo Structure

```
sarvam-sd-bench/
├── sd_bench.py       — benchmark CLI
├── prompts.txt       — 30 Hindi prompts, tab-separated
├── requirements.txt
└── README.md
```

---

## Conclusion

At K=1, Sarvam-1 gets accepted 73% of the time by Sarvam-30B.
Expected latency drops from 228ms to 126ms — 1.8× improvement.
Technical Hindi acceptance is 1.00 at K=1 on 10 prompts, unexpectedly high.

The blocker for real SD is the tokenizer mismatch.
68K vocab on one side, 262K on the other — token IDs don't match.
Until there's a shared tokenizer or a projection layer, token-level SD
between these two models isn't possible.

---

Built by [Asheesh Dhamacharla](https://github.com/asheesh07)
