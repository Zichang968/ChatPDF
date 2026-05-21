# eval_rag.py — self-contained, no rag.py dependency
# Uses the same imports as webui_thesis.py (known working in chatpdf_env)

import os, sys, json, time, csv, re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from glob import glob
from typing import List, Dict, Optional

import openai
import torch
import PyPDF2
from similarities import BertSimilarity, BM25Similarity, EnsembleSimilarity
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
QA_PER_DOC    = 5
OUTPUT_JSON   = "eval_results.json"
OUTPUT_CSV    = "eval_results.csv"
CACHE_FILE    = "qa_cache.json"

KIMI_API_KEY  = "sk-hIwarOg2FIbCID8MizCHY4x7eeu3v0ceOHztwHFriwXrkiie"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_MODEL    = "moonshot-v1-8k"

SIM_MODEL     = "shibing624/text2vec-base-multilingual"
GEN_MODEL     = "Qwen/Qwen2-0.5B-Instruct"
CHUNK_SIZE    = 250
TOP_K         = 5
MAX_NEW_TOKENS = 150

CONFIGS = {
    "Dense-Only": [1.0, 0.0],
    "BM25-Only":  [0.0, 1.0],
    "Hybrid":     [0.5, 0.5],
}

# ── Kimi client ────────────────────────────────────────────────────────────────
kimi = openai.OpenAI(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL)

def kimi_chat(prompt: str, temperature: float = 0.3, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            resp = kimi.chat.completions.create(
                model=KIMI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  [WARN] Kimi error attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None

# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pdf(path: str) -> str:
    texts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip():
                texts.append(t.strip())
    return "\n".join(texts)

# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
    chunks, cur = [], ''
    for s in sents:
        if len(cur) + len(s) <= size:
            cur += (' ' if cur else '') + s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 20]

# ── QA generation via Kimi ─────────────────────────────────────────────────────
def generate_qa(doc_text: str, doc_name: str, n: int = QA_PER_DOC) -> List[Dict]:
    text = doc_text[:8000]
    prompt = f"""Based on the document below, generate {n} QA pairs covering: direct extraction, numerical, reasoning, comparison types.
Output ONLY a JSON array, no explanation:
[{{"question":"...","answer":"...","evidence":"..."}}]

Document:
{text}"""
    print(f"  [Kimi] Generating {n} QA pairs for {doc_name}...")
    content = kimi_chat(prompt, temperature=0.3)
    if not content:
        return []
    s, e = content.find('['), content.rfind(']') + 1
    if s == -1 or e == 0:
        return []
    try:
        qa = json.loads(content[s:e])
        qa = [q for q in qa if "question" in q and "answer" in q]
        print(f"  [Kimi] Got {len(qa)} pairs.")
        return qa[:n]
    except Exception as ex:
        print(f"  [ERROR] JSON parse: {ex}")
        return []

# ── Scoring via Kimi ───────────────────────────────────────────────────────────
def score_answer(q: str, gt: str, ans: str) -> int:
    prompt = f"""Score this answer 0-3.
3=Correct. 2=Partially correct. 1=Slightly relevant. 0=Wrong/no answer.
Question: {q}
Ground truth: {gt}
System answer: {ans}
Reply with ONE digit only (0/1/2/3):"""
    content = kimi_chat(prompt, temperature=0.0)
    if content:
        for ch in content.strip():
            if ch in "0123":
                return int(ch)
    return 0

# ── Build retrieval index ──────────────────────────────────────────────────────
def build_index(chunks: List[str], weights: List[float]):
    m1 = BertSimilarity(model_name_or_path=SIM_MODEL, device='cpu')
    m2 = BM25Similarity()
    idx = EnsembleSimilarity(similarities=[m1, m2], weights=weights, c=2)
    idx.add_corpus(chunks)
    return idx

def retrieve(idx, query: str, topk: int = TOP_K) -> List[str]:
    results = idx.most_similar(query, topn=topk)
    refs = []
    for group in results:
        for item in group:
            refs.append(item["corpus_doc"])
    return refs[:topk]

# ── Direct LLM inference (no Thread/Streamer) ──────────────────────────────────
@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(
        ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.1,
    )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Collect PDFs
    pdfs = sorted(glob(os.path.join(DATA_DIR, "*.pdf")))
    print(f"\n[INFO] Found {len(pdfs)} PDF(s):")
    for p in pdfs:
        print(f"       {os.path.basename(p)}")

    # 2. QA cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            all_qa = json.load(f)
        print(f"\n[INFO] Loaded {len(all_qa)} QA pairs from cache.")
    else:
        all_qa = []
        for pdf in pdfs:
            name = os.path.basename(pdf)
            print(f"\n[PROCESS] {name}")
            text = extract_pdf(pdf)
            if not text.strip():
                print(f"  [WARN] No text extracted.")
                continue
            print(f"  [INFO] {len(text)} chars extracted.")
            qa_list = generate_qa(text, name)
            for qa in qa_list:
                all_qa.append({
                    "doc_file": pdf, "doc_name": name,
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "evidence": qa.get("evidence", ""),
                })
            time.sleep(1)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(all_qa, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Saved {len(all_qa)} QA pairs to cache.")

    print(f"\n[INFO] Total: {len(all_qa)} QA pairs from {len(set(q['doc_name'] for q in all_qa))} docs")

    # 3. Build corpus
    print("\n[INFO] Extracting and chunking all PDFs...")
    all_chunks = []
    for pdf in pdfs:
        text = extract_pdf(pdf)
        all_chunks.extend(chunk_text(text))
    print(f"[INFO] {len(all_chunks)} chunks total.")

    # 4. Load LLM
    print(f"\n[INFO] Loading {GEN_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, trust_remote_code=True)
    # Force CPU: MPS conflicts with BertSimilarity when both run simultaneously
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()
    print(f"[INFO] Model on cpu (forced for stability)")

    # 5. Evaluate all configs
    all_rows: List[Dict] = []

    for config_name, weights in CONFIGS.items():
        print(f"\n{'='*55}")
        print(f"  Config: {config_name}")
        print(f"{'='*55}")

        if weights is not None:
            print(f"  Building retrieval index (weights={weights})...")
            idx = build_index(all_chunks, weights)

        for i, item in enumerate(all_qa):
            q  = item["question"]
            gt = item["ground_truth"]

            if weights is None:
                prompt = f"Answer the following question in English:\n\n{q}\n\nAnswer:"
            else:
                refs = retrieve(idx, q)
                context = "\n".join([f"[{j+1}] {r}" for j, r in enumerate(refs)])
                prompt = (
                    f"Based on the following references, answer the question in English.\n\n"
                    f"References:\n{context}\n\nQuestion: {q}\n\nAnswer:"
                )

            answer = generate_answer(model, tokenizer, prompt)
            score  = score_answer(q, gt, answer)

            all_rows.append({
                "config": config_name, "doc_name": item["doc_name"],
                "question": q, "ground_truth": gt,
                "answer": answer, "score": score,
            })
            print(f"  [{config_name}] Q{i+1:02d}/{len(all_qa)} score={score}  {q[:50]}...")

    # 6. Save results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Results saved to {OUTPUT_JSON}")

    csv_fields = ["config", "doc_name", "question", "ground_truth", "answer", "score"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[INFO] CSV saved to {OUTPUT_CSV}")

    # 7. Summary
    print("\n\n" + "="*55)
    print("  FINAL RESULTS")
    print("="*55)
    print(f"{'Config':<14} {'N':>4} {'Total':>7} {'Avg':>6} {'Acc%':>8}")
    print("-"*42)
    for cfg in CONFIGS:
        rows = [r for r in all_rows if r["config"] == cfg]
        if not rows: continue
        total = sum(r["score"] for r in rows)
        acc   = total / (len(rows) * 3) * 100
        print(f"{cfg:<14} {len(rows):>4} {total:>7} {total/len(rows):>6.2f} {acc:>7.1f}%")
    print("="*55)

if __name__ == "__main__":
    main()
