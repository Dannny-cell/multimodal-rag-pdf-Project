# app.py
import os, re, uuid, hashlib
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image
import numpy as np

# PDF → images + OCR
from pdf2image import convert_from_bytes
import pytesseract

# Page captions (BLIP)
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Embeddings + vector store
from sentence_transformers import SentenceTransformer
import chromadb

# LLM (Groq)
from groq import Groq


# ======================== CONFIG ========================
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"   # your requested model
BGE_QUERY_PREFIX = "Represent this query for retrieving relevant passages: "

# Silence Chroma telemetry noise
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

# --- Windows paths (edit if needed, or set via sidebar/env) ---
DEFAULT_POPPLER_PATH = r"C:\Program Files\poppler\Library\bin"
DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Try to auto-configure pytesseract on Windows
if os.path.exists(DEFAULT_TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_CMD

st.set_page_config(page_title="PDF Multimodal RAG", page_icon="🧠", layout="wide")
st.title("🧠 PDF Multimodal RAG Project")
st.write("Upload a PDF → **Build Index** → Ask a question. Answers are grounded strictly on retrieved chunks.")


# ======================== HELPERS ========================
def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]

def clean_text(t: str) -> str:
    t = t.replace("\x0c", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def chunk(text: str, maxlen: int = 1000) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= maxlen:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= maxlen:
                buf = p
            else:
                for i in range(0, len(p), maxlen):
                    chunks.append(p[i:i+maxlen])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks

def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


# ======================== CACHED RESOURCES ========================
@st.cache_resource(show_spinner=False)
def load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return model, device

@st.cache_resource(show_spinner=False)
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, blip, device

@st.cache_resource(show_spinner=False)
def start_chroma(persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = embedder.encode(
        texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False
    )
    return l2norm(embs)

def embed_query(embedder: SentenceTransformer, q: str) -> np.ndarray:
    e = embedder.encode([BGE_QUERY_PREFIX + q], convert_to_numpy=True, normalize_embeddings=False)
    return l2norm(e)


# ======================== INGEST (OCR + CAPTIONS) ========================
def ocr_pdf_bytes(pdf_bytes: bytes, dpi: int = 300, lang: str = "eng", poppler_path: str | None = None
                  ) -> Tuple[List[Dict], int]:
    """Return (docs, page_count). Each doc = {'id','text','metadata'}."""
    kwargs = {}
    if poppler_path and os.path.exists(poppler_path):
        kwargs["poppler_path"] = poppler_path
    pages = convert_from_bytes(pdf_bytes, fmt="png", dpi=dpi, **kwargs)

    docs = []
    for i, page in enumerate(pages, start=1):
        txt = pytesseract.image_to_string(page.convert("RGB"), lang=lang)
        txt = clean_text(txt)
        if not txt:
            continue
        for j, c in enumerate(chunk(txt, 1000), start=1):
            docs.append({"id": _id("text"), "text": c, "metadata": {"type": "text", "page_number": i, "chunk": j}})
    return docs, len(pages)

def render_pages_bytes(pdf_bytes: bytes, dpi: int = 200, poppler_path: str | None = None) -> List[Image.Image]:
    kwargs = {}
    if poppler_path and os.path.exists(poppler_path):
        kwargs["poppler_path"] = poppler_path
    return convert_from_bytes(pdf_bytes, fmt="png", dpi=dpi, **kwargs)

def caption_pages(images: List[Image.Image]) -> List[Dict]:
    processor, blip, device = load_blip()
    docs = []
    for idx, img in enumerate(images, start=1):
        inputs = processor(images=img.convert("RGB"), return_tensors="pt").to(device)
        out = blip.generate(**inputs, max_new_tokens=40)
        cap = processor.decode(out[0], skip_special_tokens=True).strip()
        docs.append({"id": _id("img"), "text": cap, "metadata": {"type": "image_caption", "page_index": idx}})
    return docs


# ======================== INDEX BUILD / LOAD ========================
def build_index_for_file(pdf_bytes: bytes, persist_root: str, use_captions: bool, poppler_path: str
                         ) -> Tuple[chromadb.Collection, SentenceTransformer, int, int, str]:
    file_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    persist_dir = os.path.join(persist_root, file_id)
    client = start_chroma(persist_dir)

    # fresh collection for this file
    col_name = "attention-is-all-you-need"
    try:
        client.delete_collection(col_name)
    except Exception:
        pass
    collection = client.create_collection(name=col_name, metadata={"hnsw:space": "cosine"})

    with st.status("Ingesting (OCR + pages + captions)…", expanded=True) as s:
        st.write("• OCR text …")
        text_docs, page_count = ocr_pdf_bytes(pdf_bytes, dpi=300, lang="eng", poppler_path=poppler_path)
        st.write(f"  → text chunks: {len(text_docs)} from {page_count} pages")

        caption_count = 0
        image_docs = []
        if use_captions:
            st.write("• Rendering pages …")
            imgs = render_pages_bytes(pdf_bytes, dpi=200, poppler_path=poppler_path)
            st.write("• Generating BLIP captions …")
            image_docs = caption_pages(imgs)
            caption_count = len(image_docs)
            st.write(f"  → captions: {caption_count}")

        all_docs = text_docs + image_docs
        texts = [d["text"] for d in all_docs]
        metas = [d["metadata"] for d in all_docs]
        ids   = [d["id"] for d in all_docs]

        embedder, device = load_embedder()
        st.write(f"• Embedding {len(texts)} items on **{device}** …")
        vecs = embed_texts(embedder, texts)
        collection.add(documents=texts, metadatas=metas, ids=ids, embeddings=vecs.tolist())
        st.write("• Stored in Chroma")
        s.update(label="Ingestion complete", state="complete")

    return collection, embedder, len(texts), caption_count, file_id

@st.cache_resource(show_spinner=False)
def load_index_for_hash(file_id: str, persist_root: str) -> Tuple[chromadb.Collection, SentenceTransformer]:
    persist_dir = os.path.join(persist_root, file_id)
    client = start_chroma(persist_dir)
    collection = client.get_collection("attention-is-all-you-need")
    embedder, _ = load_embedder()
    return collection, embedder


# ======================== RETRIEVE + ANSWER ========================
def retrieve(collection: chromadb.Collection, embedder: SentenceTransformer, query: str, k: int = 10):
    qvec = embed_query(embedder, query)
    res = collection.query(query_embeddings=qvec.tolist(), n_results=k)
    hits = []
    for doc, meta, id_, dist in zip(res["documents"][0], res["metadatas"][0], res["ids"][0], res["distances"][0]):
        hits.append({"id": id_, "text": doc, "metadata": meta, "distance": dist})
    return hits

def build_context(hits: List[Dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        meta = h["metadata"] or {}
        page = meta.get("page_number") or meta.get("page_index")
        kind = meta.get("type")
        parts.append(f"### Chunk {i} [{kind} | page={page}]\n{h['text']}")
    return "\n\n".join(parts)

def answer_groq(context: str, question: str, model_name: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY") or st.session_state.get("groq_api_key", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it in the sidebar.")
    client = Groq(api_key=api_key)
    system = (
        "You are a careful, concise technical assistant.\n"
        "Answer ONLY from the provided CONTEXT. If insufficient, say you don't know.\n"
        "Prefer bullets and mention page hints from metadata when present.\n"
    )
    prompt = f"{system}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer:"
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content


# ======================== UI ========================
with st.sidebar:
    st.header("⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    if groq_key:
        st.session_state["groq_api_key"] = groq_key

    model_name = st.text_input("Groq Model", value=DEFAULT_GROQ_MODEL)
    use_captions = st.checkbox("Generate page captions (BLIP)", value=True, help="Turn off for faster indexing.")
    st.markdown("---")

    st.caption("Windows paths (edit here if needed):")
    poppler_path = st.text_input("POPPLER_PATH", value=os.environ.get("POPPLER_PATH", DEFAULT_POPPLER_PATH))
    tesseract_cmd = st.text_input("TESSERACT_CMD", value=os.environ.get("TESSERACT_CMD", DEFAULT_TESSERACT_CMD))

    # Apply Tesseract path live
    if tesseract_cmd and os.path.exists(tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

st.write("1) Upload a PDF. 2) Click **Build Index**. 3) Ask a question.")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
persist_root = os.path.join(os.getcwd(), ".rag_store")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    if st.button("🚀 Build Index", type="primary"):
        try:
            collection, embedder, total_items, cap_count, fid = build_index_for_file(
                pdf_bytes, persist_root, use_captions=use_captions, poppler_path=poppler_path
            )
            st.success(f"Index ready! Items: {total_items} (captions: {cap_count})")
            st.session_state["fid"] = fid
        except Exception as e:
            st.error(str(e))

    st.divider()

    if "fid" not in st.session_state:
        st.info("Build the index first.")
    else:
        # load existing index
        collection, embedder = load_index_for_hash(st.session_state["fid"], persist_root)

        q = st.text_input(
            "Ask a question about the PDF:",
            value="Explain the encoder-decoder architecture and multi-head attention."
        )
        k = st.slider("Top-k context chunks", 3, 20, 10, 1)

        if st.button("💡 Answer"):
            with st.spinner("Retrieving and synthesizing..."):
                hits = retrieve(collection, embedder, q, k=k)
                context = build_context(hits)
                try:
                    ans = answer_groq(context, q, model_name)
                    st.markdown("### ✅ Answer")
                    st.write(ans)
                except Exception as e:
                    st.error(str(e))

                with st.expander("🔎 Sources"):
                    for i, h in enumerate(hits, 1):
                        meta = h["metadata"] or {}
                        page = meta.get("page_number") or meta.get("page_index")
                        kind = meta.get("type")
                        st.markdown(f"**[{i}]** {kind} • page={page} • dist={h['distance']:.4f}")
                        preview = h["text"][:600] + ("…" if len(h["text"]) > 600 else "")
                        st.write(preview)
