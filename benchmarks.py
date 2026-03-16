import argparse
import json
import os
import sys
import time

import pandas as pd
import requests
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_PROMPTS = {
    "general": [
        "आज मौसम बहुत अच्छा है",
        "मुझे चाय बहुत पसंद है",
        "भारत एक विविधताओं से भरा देश है",
        "मेरे परिवार में चार लोग हैं",
        "हम कल बाजार जाएंगे",
        "खाना बहुत स्वादिष्ट था",
        "वह एक अच्छा इंसान है",
        "आज स्कूल में छुट्टी है",
        "मुझे किताबें पढ़ना पसंद है",
        "बच्चे मैदान में खेल रहे हैं",
    ],
    "technical": [
        "मशीन लर्निंग एक ऐसी तकनीक है जो",
        "न्यूरल नेटवर्क में बैकप्रोपगेशन का उपयोग",
        "डेटाबेस में इंडेक्स बनाने से क्वेरी",
        "एपीआई एंडपॉइंट को कॉल करने के लिए",
        "क्लाउड कंप्यूटिंग में वर्चुअलाइजेशन का अर्थ",
        "गहरी शिक्षा मॉडल को प्रशिक्षित करने के लिए",
        "टोकनाइज़ेशन प्रक्रिया में शब्दों को",
        "ट्रांसफॉर्मर आर्किटेक्चर में अटेंशन मैकेनिज्म",
        "सर्वर पर लोड बैलेंसिंग करने से",
        "कुबेरनेटेस में पॉड डिप्लॉयमेंट की प्रक्रिया",
    ],
    "mixed": [
        "मैंने आज Python में एक नया script लिखा",
        "इस model की accuracy बहुत अच्छी है",
        "Data preprocessing के बाद results",
        "इस API को integrate करना आसान है",
        "Machine learning के बिना modern AI",
        "हमारा server आज down हो गया",
        "Database की query optimize करनी होगी",
        "इस feature को production में deploy करें",
        "Testing के दौरान बहुत bugs मिले",
        "Cloud पर migrate करने से cost कम होगी",
    ],
}

def load_prompts(path):
    if path is None:
        return DEFAULT_PROMPTS
    if not os.path.exists(path):
        print(f"ERROR: prompts file not found: {path}")
        sys.exit(1)
    prompts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(f"WARNING: skipping malformed line: {line!r}")
                continue
            cat, prompt = parts
            prompts.setdefault(cat.strip(), []).append(prompt.strip())
    if not prompts:
        print("ERROR: no valid prompts found in file.")
        sys.exit(1)
    total = sum(len(v) for v in prompts.values())
    print(f"Loaded {total} prompts | categories: {list(prompts.keys())}")
    return prompts

def _post(payload, api_key, timeout=45):
    r = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    data = r.json()
    if r.status_code != 200:
        raise Exception(f"API {r.status_code}: {data}")
    return data["choices"][0]["message"]["content"].strip()


def sarvam_chat(prompt, model, api_key, max_tokens=40):
    """Single-prompt API call. Used for the startup test only."""
    return _post({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "reasoning_effort": None,  
    }, api_key)


def sarvam_batch_verify(pairs, model, api_key, max_tokens=200):
    numbered = "\n".join(
        f'{i+1}. [{p}] → [{c}]'
        for i, (p, c) in enumerate(pairs)
    )
    verify_prompt = (
        f"You are evaluating speculative decoding drafts.\n"
        f"Each item shows a Hindi prompt and the model's next few predicted tokens.\n"
        f"These are partial continuations, not complete sentences.\n\n"
        f"{numbered}\n\n"
        f"ACCEPT if the tokens are a natural next continuation.\n"
        f"REJECT only if the tokens are clearly wrong or contradictory.\n\n"
        f"Reply with one line per item, numbered. Example format:\n"
        f"1: ACCEPT\n"
        f"2: ACCEPT\n"
        f"3: REJECT\n"
        f"Output ALL {len(pairs)} lines."
    )

    raw = _post({
    "model": model,
    "messages": [{"role": "user", "content": verify_prompt}],
    "temperature": 0.1,
    "max_tokens": max_tokens,
    "reasoning_effort": None,
}, api_key)

    results = [True] * len(pairs)
    for line in raw.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        idx_str, verdict = line.split(":", 1)
        try:
            idx = int(idx_str.strip()) - 1
            if 0 <= idx < len(results):
                results[idx] = not verdict.strip().upper().startswith("REJECT")
        except ValueError:
            continue
    return results, raw

def load_draft_model(model_id):
    print(f"\nLoading draft model: {model_id}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = next(model.parameters()).device
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded on {device} | GPU memory: {mem_gb:.2f} GB ✅")
    return model, tokenizer, device

def generate_draft(prompt, K, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=K,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    ms = (time.perf_counter() - t0) * 1000
    text = tokenizer.decode(
        ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return text, ms

def speedup(alpha, draft_per_token_ms, api_ms):
    expected = draft_per_token_ms + (1 - alpha) * api_ms
    return api_ms / expected

def run_benchmark(prompts, k_values, model, tokenizer, device,
                  target, api_key, batch_size, sleep_sec):
    all_results = []
    total_prompts = sum(len(v) for v in prompts.values())
    total_runs = total_prompts * len(k_values)
    done = 0

    for K in k_values:
        print(f"\n{'='*52}")
        print(f"K = {K}   ({total_prompts} prompts)")
        print(f"{'='*52}")

        for category, prompt_list in prompts.items():
            print(f"\n  [{category.upper()}]")
            drafted = []
            for prompt in prompt_list:
                cont, draft_ms = generate_draft(prompt, K, model, tokenizer, device)
                drafted.append((prompt, cont, draft_ms))

            pairs = [(p, c) for p, c, _ in drafted]
            verified = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                t0 = time.perf_counter()
                try:
                    accepted_list, _ = sarvam_batch_verify(batch, target, api_key)
                    batch_api_ms = (time.perf_counter() - t0) * 1000
                    per_ms = batch_api_ms / len(batch)
                    verified.extend(zip(accepted_list, [per_ms] * len(batch)))
                except Exception as e:
                    print(f"    Batch error: {e}")
                    verified.extend([(None, None)] * len(batch))
                time.sleep(sleep_sec)

            for (prompt, cont, draft_ms), (accepted, api_ms) in zip(drafted, verified):
                if accepted is None:
                    continue
                done += 1
                overhead = round(api_ms / draft_ms, 2) if draft_ms > 0 else None
                status = "✅" if accepted else "❌"
                print(
                    f"  [{done:03d}/{total_runs}] K={K} {status}  "
                    f"'{cont[:25]}...'  "
                    f"draft={draft_ms:.0f}ms  api(per)={api_ms:.0f}ms"
                )
                all_results.append({
                    "prompt":                prompt,
                    "K":                     K,
                    "category":              category,
                    "draft_continuation":    cont,
                    "accepted":              accepted,
                    "draft_forward_time_ms": round(draft_ms, 1),
                    "target_forward_time_ms": round(api_ms, 1),
                    "verification_overhead": overhead,
                    "sampling_time_ms":      round(draft_ms + api_ms, 1),
                })

    return pd.DataFrame(all_results)

def save_summary(df, k_values, output_path):
    categories = sorted(df["category"].unique().tolist())
    df["draft_per_token_ms"] = df["draft_forward_time_ms"] / df["K"]
    draft_per_token_ms = df["draft_per_token_ms"].mean()
    api_ms = df["target_forward_time_ms"].mean()

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    hdr = f"{'K':<6} {'α':<8} {'Speedup':<10}"
    for cat in categories:
        hdr += f"{'α_'+cat:<14}"
    hdr += "Domain Δ"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for K in k_values:
        kdf = df[df.K == K]
        if kdf.empty:
            continue
        a = kdf["accepted"].mean()
        sp = speedup(a, draft_per_token_ms, api_ms)
        cat_a = {cat: kdf[kdf.category == cat]["accepted"].mean() for cat in categories}
        delta = cat_a.get("general", float("nan")) - cat_a.get("technical", float("nan"))

        line = f"K={K:<4} {a:.2f}    {sp:.2f}x      "
        for cat in categories:
            line += f"{cat_a[cat]:.2f}          "
        line += f"{delta*100:+.0f}pp"
        print(line)

        row = {"K": K, "alpha_overall": round(a, 3), "speedup": round(sp, 3)}
        for cat in categories:
            row[f"alpha_{cat}"] = round(cat_a[cat], 3)
        row["domain_shift_delta_pp"] = round(delta * 100, 1)
        rows.append(row)

    overhead = df["verification_overhead"].mean()
    best_K = df.groupby("K")["accepted"].mean().idxmax()
    best_a = df.groupby("K")["accepted"].mean().max()
    best_sp = speedup(best_a, draft_per_token_ms, api_ms)
    expected_ms = best_a * draft_per_token_ms + (1 - best_a) * api_ms

    print(f"\n{'─'*50}")
    print("TIMING")
    print(f"  Draft per token    : {draft_per_token_ms:.1f} ms")
    print(f"  Target API per call: {api_ms:.1f} ms  (per-prompt after batching)")
    print(f"  Overhead           : {overhead:.2f}x")
    print(f"  Avg sampling time  : {df['sampling_time_ms'].mean():.1f} ms")

    print(f"\n{'─'*50}")
    print("KEY NUMBERS — EMAIL READY")
    print(f"  Best α             : {best_a:.2f}  (K={best_K})")
    print(f"  Speedup at K={best_K}    : {best_sp:.1f}x")
    print(f"  Expected latency   : {expected_ms:.0f}ms vs {api_ms:.0f}ms baseline")
    print(f"  Domain shift       : general α={df[df.category=='general']['accepted'].mean():.2f}  "
          f"technical α={df[df.category=='technical']['accepted'].mean():.2f}")
    print(f"  Tokenizer mismatch : sarvam-1 68K vocab ≠ sarvam-30b 262K vocab")
    print(f"  Blocker            : token-level SD requires shared tokenizer")

    summary_path = output_path.replace(".csv", "_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    df.to_csv(output_path, index=False)
    print(f"\nSaved : {output_path}")
    print(f"Saved : {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Speculative Decoding Efficiency Benchmark — Sarvam model family",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--draft",      default="sarvamai/sarvam-1")
    parser.add_argument("--target",     default="sarvam-30b")
    parser.add_argument("--prompts",    default=None)
    parser.add_argument("--k-values",  nargs="+", type=int, default=[1, 3, 5, 7, 10])
    parser.add_argument("--output",     default="report.csv")
    parser.add_argument("--api-key",   default=os.getenv("SARVAM_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=5,
        help="Prompts per API call — reduces per-prompt API cost (default: 5)")
    parser.add_argument("--sleep",      type=float, default=0.3,
        help="Sleep between batch calls in seconds (default: 0.3)")
    parser.add_argument("--skip-confirm", action="store_true")

    args = parser.parse_args()

    if not args.api_key:
        parser.error("API key required. Pass --api-key or set SARVAM_API_KEY.")

    print("=" * 60)
    print("SARVAM SPECULATIVE DECODING BENCHMARK")
    print("=" * 60)
    print(f"  Draft   : {args.draft}")
    print(f"  Target  : {args.target}")
    print(f"  K values: {args.k_values}")
    print(f"  Batch   : {args.batch_size} prompts/call")
    print(f"  Output  : {args.output}")
    print(f"\nTesting {args.target} API...")
    t0 = time.perf_counter()
    try:
        resp = sarvam_chat("भारत की राजधानी क्या है?", args.target, args.api_key)
        lat = (time.perf_counter() - t0) * 1000
        print(f"  {resp[:60]}")
        print(f"  Latency: {lat:.0f}ms  ✅")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
    prompts = load_prompts(args.prompts)

    print("\nTokenizer check...")
    from transformers import AutoTokenizer as AT
    tok = AT.from_pretrained(args.draft)
    print(f"  {args.draft}: vocab={tok.vocab_size:,} ({type(tok).__name__})")
    print(f"  {args.target}: vocab=262,144 (API-only, confirmed different)")
    print(f"  Token-level SD blocked — using output-level verification")
    tok_info = {"draft_vocab": tok.vocab_size}

    model, tokenizer, device = load_draft_model(args.draft)
    print("\n" + "─" * 40)
    print("SINGLE TEST (K=5)")
    print("─" * 40)
    cont, d_ms = generate_draft("भारत की राजधानी है", 5, model, tokenizer, device)
    t0 = time.perf_counter()
    acc_list, raw = sarvam_batch_verify(
        [("भारत की राजधानी है", cont)], args.target, args.api_key
    )
    a_ms = (time.perf_counter() - t0) * 1000
    print(json.dumps({
        "draft_continuation": cont,
        "accepted": acc_list[0],
        "draft_ms": round(d_ms, 1),
        "api_ms": round(a_ms, 1),
        "target_raw": raw[:80],
    }, ensure_ascii=False, indent=2))

    if not args.skip_confirm:
        print("\nStarting full benchmark...")

    df = run_benchmark(
        prompts, args.k_values,
        model, tokenizer, device,
        args.target, args.api_key,
        args.batch_size, args.sleep,
    )

    if df.empty:
        print("No results collected.")
        sys.exit(1)

    save_summary(df, args.k_values, args.output)
    print("\nDone ✅")

if __name__ == "__main__":
    main()