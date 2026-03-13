import argparse
import os
import sys
import time
import json
import requests
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
            category, prompt = parts
            prompts.setdefault(category.strip(), []).append(prompt.strip())

    if not prompts:
        print("ERROR: no valid prompts found in file.")
        sys.exit(1)

    total = sum(len(v) for v in prompts.values())
    print(f"Loaded {total} prompts across {len(prompts)} categories: {list(prompts.keys())}")
    return prompts

def sarvam_chat(prompt, target_model, api_key, max_tokens=80):
    r = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "reasoning_effort": None,    # ← add this line
        },
        timeout=30,
    )
    data = r.json()
    if r.status_code != 200:
        raise Exception(f"API {r.status_code}: {data}")
    return data["choices"][0]["message"]["content"].strip()

def load_draft_model(draft_model_id):
    print(f"\nLoading draft model: {draft_model_id}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
    device = next(model.parameters()).device
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Draft loaded on {device} | GPU memory: {mem_gb:.2f} GB ✅")
    return model, tokenizer, device

def check_tokenizer_compatibility(draft_model_id, target_model_id):
    print("\nChecking tokenizer compatibility...")
    tok_draft  = AutoTokenizer.from_pretrained(draft_model_id)
    tok_target = AutoTokenizer.from_pretrained(target_model_id) if "/" in target_model_id else None

    vocab_draft = tok_draft.vocab_size
    result = {
        "draft_vocab":  vocab_draft,
        "draft_type":   type(tok_draft).__name__,
        "target_vocab": None,
        "match":        None,
    }

    if tok_target:
        vocab_target = tok_target.vocab_size
        match = vocab_draft == vocab_target
        result["target_vocab"] = vocab_target
        result["target_type"]  = type(tok_target).__name__
        result["match"]        = match
        print(f"  {draft_model_id:<30} vocab={vocab_draft:>7,}  ({result['draft_type']})")
        print(f"  {target_model_id:<30} vocab={vocab_target:>7,}  ({result['target_type']})")
        print(f"  Tokenizer match: {match}")
        if not match:
            print("  ⚠  Token-level SD blocked — using output-level verification as proxy")
    else:
        print(f"  {draft_model_id} vocab={vocab_draft:,} ({result['draft_type']})")
        print(f"  Target is API-only — tokenizer check skipped")

    return result

def run_experiment(prompt, K, draft_model, tokenizer, device, target_model, api_key):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        draft_ids = draft_model.generate(
            **inputs,
            max_new_tokens=K,
            do_sample=False,
        )
    draft_forward_time_ms = (time.perf_counter() - t0) * 1000

    draft_continuation = tokenizer.decode(
        draft_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    verify_prompt = (
        f'Hindi text verification task.\n\n'
        f'Original: "{prompt}"\n'
        f'Proposed continuation: "{draft_continuation}"\n\n'
        f'Does this continuation flow naturally? Reply ONLY:\n'
        f'ACCEPT: <continuation>\nor\nREJECT: <better continuation>'
    )

    t1 = time.perf_counter()
    try:
        target_response = sarvam_chat(verify_prompt, target_model, api_key, max_tokens=80)
        target_forward_time_ms = (time.perf_counter() - t1) * 1000

        accepted = target_response.upper().startswith("ACCEPT")
        sampling_time_ms = draft_forward_time_ms + target_forward_time_ms
        overhead = (
            round(target_forward_time_ms / draft_forward_time_ms, 2)
            if draft_forward_time_ms > 0 else None
        )

        return {
            "prompt":                 prompt,
            "K":                      K,
            "draft_continuation":     draft_continuation,
            "target_response":        target_response,
            "accepted":               accepted,
            "draft_forward_time_ms":  round(draft_forward_time_ms, 1),
            "target_forward_time_ms": round(target_forward_time_ms, 1),
            "verification_overhead":  overhead,
            "sampling_time_ms":       round(sampling_time_ms, 1),
        }
    except Exception as e:
        print(f"    API error: {e}")
        return None

def print_and_save_summary(df, k_values, output_path):
    categories = sorted(df["category"].unique().tolist())

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'K':<5} {'α':<8} {'Speedup':<10}", end="")
    for cat in categories:
        print(f"{'α_' + cat:<14}", end="")
    print("Domain Δ (gen-tech)")
    print("-" * (5 + 8 + 10 + 14 * len(categories) + 20))

    summary_rows = []
    for K in k_values:
        kdf    = df[df.K == K]
        a_all  = kdf["accepted"].mean()
        speedup = (1 + a_all * K) / (1 + K)
        cat_alphas = {cat: kdf[kdf.category == cat]["accepted"].mean() for cat in categories}
        delta = cat_alphas.get("general", float("nan")) - cat_alphas.get("technical", float("nan"))

        print(f"K={K:<3}  {a_all:.2f}    {speedup:.2f}x      ", end="")
        for cat in categories:
            print(f"{cat_alphas[cat]:.2f}          ", end="")
        print(f"{delta*100:+.0f}pp")

        row = {"K": K, "alpha_overall": round(a_all, 3), "speedup": round(speedup, 3)}
        for cat in categories:
            row[f"alpha_{cat}"] = round(cat_alphas[cat], 3)
        row["domain_shift_delta_pp"] = round(delta * 100, 1)
        summary_rows.append(row)

    print(f"\n{'─'*40}")
    print("TIMING")
    print(f"  Avg draft forward time :  {df['draft_forward_time_ms'].mean():.1f} ms")
    print(f"  Avg target forward time:  {df['target_forward_time_ms'].mean():.1f} ms")
    print(f"  Avg verification overhead:{df['verification_overhead'].mean():.2f}x")
    print(f"  Avg total sampling time:  {df['sampling_time_ms'].mean():.1f} ms")

    avg_overhead = df["verification_overhead"].mean()
    breakeven    = 1 / avg_overhead if avg_overhead > 0 else None
    best_alpha   = df.groupby("K")["accepted"].mean().max()
    best_K       = df.groupby("K")["accepted"].mean().idxmax()
    print(f"\n  Breakeven α for SD viability: {breakeven:.2f}")
    print(f"  Best α achieved: {best_alpha:.2f}  (K={best_K})")
    if best_alpha and breakeven and best_alpha > breakeven:
        print("  SD IS VIABLE at peak acceptance rate ✅")
    else:
        print("  SD not yet viable — needs shared tokenizer or higher α ⚠")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_path.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Saved: {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Speculative Decoding Efficiency Benchmark for Sarvam model family",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--draft",
        default="sarvamai/sarvam-1",
        help="HuggingFace model ID for the draft model (default: sarvamai/sarvam-1)",
    )
    parser.add_argument(
        "--target",
        default="sarvam-30b",
        help="Sarvam API model ID for the target model (default: sarvam-30b)",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Path to tab-separated prompts file (default: built-in 30 Hindi prompts)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 10],
        metavar="K",
        help="Draft token counts to sweep (default: 1 3 5 7 10)",
    )
    parser.add_argument(
        "--output",
        default="report.csv",
        help="Output CSV filename (default: report.csv)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("SARVAM_API_KEY"),
        help="Sarvam API key (default: $SARVAM_API_KEY env var)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls for rate limiting (default: 0.5)",
    )
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Skip the single-test confirmation prompt and run immediately",
    )

    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "API key required. Pass --api-key or set SARVAM_API_KEY env variable."
        )

    print("=" * 60)
    print("SARVAM SPECULATIVE DECODING BENCHMARK")
    print("=" * 60)
    print(f"  Draft model : {args.draft}")
    print(f"  Target model: {args.target}")
    print(f"  K values    : {args.k_values}")
    print(f"  Output      : {args.output}")

    prompts = load_prompts(args.prompts)

    print(f"\nTesting {args.target} API...")
    try:
        resp = sarvam_chat(
            "भारत की राजधानी क्या है?", args.target, args.api_key, max_tokens=30
        )
        print(f"  Response: {resp}")
        print("  API ✅")
    except Exception as e:
        print(f"  API FAILED: {e}")
        sys.exit(1)

    tok_info = check_tokenizer_compatibility(args.draft, args.target)

    draft_model, tokenizer, device = load_draft_model(args.draft)

    print("\n" + "─" * 40)
    print("SINGLE TEST (K=5)")
    print("─" * 40)
    test = run_experiment(
        "भारत की राजधानी है", 5,
        draft_model, tokenizer, device,
        args.target, args.api_key,
    )
    print(json.dumps(test, ensure_ascii=False, indent=2))

    if not args.skip_confirm:
        try:
            input("\nPress Enter to run full benchmark, Ctrl+C to abort...")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    all_results   = []
    total_prompts = sum(len(v) for v in prompts.values())
    total_runs    = total_prompts * len(args.k_values)
    done          = 0

    for K in args.k_values:
        print(f"\n{'='*50}")
        print(f"K = {K}  ({total_prompts} prompts)")
        print(f"{'='*50}")

        for category, prompt_list in prompts.items():
            print(f"\n  [{category.upper()}]")
            for prompt in prompt_list:
                result = run_experiment(
                    prompt, K,
                    draft_model, tokenizer, device,
                    args.target, args.api_key,
                )
                if result:
                    result["category"] = category
                    all_results.append(result)
                    done += 1
                    status = "✅" if result["accepted"] else "❌"
                    print(
                        f"  [{done:03d}/{total_runs}] K={K} {status}  "
                        f"'{result['draft_continuation'][:25]}...'  "
                        f"draft={result['draft_forward_time_ms']:.0f}ms  "
                        f"api={result['target_forward_time_ms']:.0f}ms"
                    )
                time.sleep(args.sleep)

    if not all_results:
        print("No results collected. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(all_results)

    print(f"\n{'─'*40}")
    print("TOKENIZER FINDING")
    print(f"  {args.draft} vocab : {tok_info['draft_vocab']:,}  ({tok_info['draft_type']})")
    if tok_info.get("target_vocab"):
        print(f"  {args.target} vocab : {tok_info['target_vocab']:,}  ({tok_info.get('target_type', '?')})")
        print(f"  Match: {tok_info['match']}")
    print(f"  Output-level verification used as SD proxy")

    print_and_save_summary(df, args.k_values, args.output)
    print("\nDone ✅")


if __name__ == "__main__":
    main()