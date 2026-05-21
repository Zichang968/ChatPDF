# -*- coding: utf-8 -*-
"""
ChatPDF WebUI — Thesis Demo (Gradio 3.x, GAT paper corpus)
Privacy-Preserving Local PDF Q&A System
"""
import warnings, os
warnings.filterwarnings('ignore')
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys, torch, jieba, re, time
jieba.setLogLevel("ERROR")

from loguru import logger
from similarities import BertSimilarity, BM25Similarity, EnsembleSimilarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2
import gradio as gr

# ── Config ──────────────────────────────────────────────
PDF_PATH       = "/Users/zhenyuanbo/Desktop/wangzichang/submission/test_doc.pdf"
SIM_MODEL      = "shibing624/text2vec-base-multilingual"
GEN_MODEL      = "Qwen/Qwen2-0.5B-Instruct"
CHUNK_SIZE     = 250
TOP_K          = 3
MAX_NEW_TOKENS = 300

PROMPT = """Based on the following reference content, answer the user's question in English.

Reference:
{context}

Question: {question}

Answer:"""

# ── Text Splitter ────────────────────────────────────────
def split_text(text, chunk_size=250):
    if any('一' <= c <= '鿿' for c in text):
        chunks, cur = [], ''
        for word in jieba.cut(text):
            cur += word
            if len(cur) >= chunk_size:
                chunks.append(cur.strip()); cur = ''
        if cur: chunks.append(cur.strip())
        return [c for c in chunks if c]
    else:
        sents = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        chunks, cur = [], ''
        for s in sents:
            if len(cur) + len(s) <= chunk_size: cur += (' ' if cur else '') + s
            else:
                if cur: chunks.append(cur)
                cur = s
        if cur: chunks.append(cur)
        return [c for c in chunks if c]

# ── Load PDF ─────────────────────────────────────────────
def load_pdf(path):
    texts = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip(): texts.append(t.strip())
    return texts

# ── Init system ──────────────────────────────────────────
print("=" * 60)
print("  ChatPDF — Privacy-Preserving Local RAG Q&A System")
print("  Document: Graph Attention Networks (GAT) paper")
print("=" * 60)

print("\n[1/3] Loading PDF and building retrieval index...")
pdf_texts = load_pdf(PDF_PATH)
chunks = []
for t in pdf_texts:
    chunks.extend(split_text(t, CHUNK_SIZE))
print(f"      {len(pdf_texts)} pages -> {len(chunks)} chunks indexed")

m1 = BertSimilarity(model_name_or_path=SIM_MODEL, device='cpu')
m2 = BM25Similarity()
sim_model = EnsembleSimilarity(similarities=[m1, m2], weights=[0.5, 0.5], c=2)
sim_model.add_corpus(chunks)
print(f"      Retrieval index: {len(sim_model.corpus)} entries (BERT + BM25 ensemble)")

print("\n[2/3] Loading Qwen2-0.5B-Instruct language model...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL, dtype=torch.float16, device_map="auto", trust_remote_code=True)
gen_model.eval()
device = next(gen_model.parameters()).device
print(f"      Generator ready on: {device}")

print("\n[3/3] Starting Gradio web interface...\n")

# ── RAG pipeline ─────────────────────────────────────────
def rag_answer(question):
    t0 = time.time()
    results = sim_model.most_similar(question, topn=TOP_K)
    refs = []
    for group in results:
        for item in group:
            refs.append(item["corpus_doc"])
    refs = refs[:TOP_K]
    retrieval_ms = (time.time() - t0) * 1000

    context = "\n".join([f"[{i+1}] {r}" for i, r in enumerate(refs)])
    prompt  = PROMPT.format(context=context, question=question)
    msgs    = [{"role": "user", "content": prompt}]
    ids     = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(device)

    t1 = time.time()
    with torch.inference_mode():
        out = gen_model.generate(
            ids, max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7, do_sample=True, repetition_penalty=1.1)
    gen_ms = (time.time() - t1) * 1000
    answer = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return answer, refs, retrieval_ms, gen_ms

def chat_fn(message, history):
    answer, refs, rt, gt = rag_answer(message)
    ref_text = "\n\n---\n**Retrieved Sources** (retrieval: {:.0f}ms | generation: {:.0f}ms):\n".format(rt, gt)
    for i, r in enumerate(refs):
        snippet = r[:150].replace('\n', ' ')
        ref_text += f"\n> **[{i+1}]** {snippet}{'...' if len(r)>150 else ''}"
    return answer + ref_text

# ── Gradio UI ─────────────────────────────────────────────
with gr.Blocks(title="ChatPDF - GAT Paper Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # ChatPDF — Privacy-Preserving Local PDF Q&A System
    **Document:** Graph Attention Networks (GAT) &emsp;|&emsp;
    **Generator:** Qwen2-0.5B-Instruct &emsp;|&emsp;
    **Retrieval:** BERT + BM25 Ensemble &emsp;|&emsp;
    **Device:** {device}

    *All processing is performed locally — no data is uploaded to any cloud service.*
    """)

    chatbot = gr.Chatbot(height=500, label="Conversation")
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question about the Graph Attention Networks paper...",
            label="Your Question", scale=5, lines=2)
        btn = gr.Button("Submit", variant="primary", scale=1)
    clear = gr.Button("Clear conversation")

    gr.Examples(
        examples=[
            "What is a Graph Attention Network?",
            "How does the attention mechanism work in GAT?",
            "What datasets were used to evaluate GAT?",
            "What is the difference between GAT and GCN?",
            "What accuracy did GAT achieve on the Cora dataset?",
        ],
        inputs=msg,
        label="Example Questions"
    )

    def respond(message, history):
        history = history or []
        reply = chat_fn(message, history)
        history.append([message, reply])
        return history, ""

    btn.click(respond, [msg, chatbot], [chatbot, msg])
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

demo.queue()
demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
