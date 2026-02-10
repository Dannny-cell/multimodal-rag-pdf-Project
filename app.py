import os, re, uuid, hashlib, shutil
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

# ======================== 1. DYNAMIC SYSTEM PATHS ========================
# Automatically find Tesseract and Poppler in the system PATH
def get_system_tool_paths():
    # Find Tesseract
    tesseract_bin = shutil.which("tesseract")
    
    # Fallback for standard Windows installation if not in PATH
    if not tesseract_bin:
        win_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(win_tess):
            tesseract_bin = win_tess

    # Find Poppler (pdftoppm)
    poppler_bin = shutil.which("pdftoppm")
    poppler_path = os.path.dirname(poppler_bin) if poppler_bin else None
    
    # Fallback for standard Windows installation
    if not poppler_path:
        win_pop = r"C:\Program Files\poppler\Library\bin"
        if os.path.exists(win_pop):
            poppler_path = win_pop
            
    return tesseract_bin, poppler_path

TESSERACT_EXE, POPPLER_PATH = get_system_tool_paths()

if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# ======================== CONFIG ========================
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
BGE_QUERY_PREFIX = "Represent this query for retrieving relevant passages: "
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

st.set_page_config(page_title="PDF Multimodal RAG", page_icon="🧠", layout="wide")
st.title("🧠 PDF Multimodal RAG Project")

# ======================== HELPERS & RESOURCES ========================
def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def clean_text(t: str) -> str:
    t = t.replace("\x0c", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)
    return t

@st.cache_resource(show_spinner=False)
def load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL_NAME, device=device), device

@st.cache_resource(show_spinner=False)
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

# ======================== UI SIDEBAR (UPGRADED) ========================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Security
    groq_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    if groq_key:
        st.session_state["groq_api_key"] = groq_key

    # Tool Path Validation
    st.subheader("System Tools")
    if not TESSERACT_EXE:
        st.error("Tesseract not found! Please install it or set PATH.")
    else:
        st.success(f"Tesseract: Found")

    if not POPPLER_PATH:
        st.warning("Poppler not found! PDF-to-Image might fail.")
    
    use_captions = st.checkbox("Generate BLIP Captions", value=True)

# ======================== CORE LOGIC (UNCHANGED) ========================
# ... (Keep your ocr_pdf_bytes, build_index, and answer_groq functions from your original code)

# Note: Ensure you use `POPPLER_PATH` variable instead of the hardcoded string
# in your convert_from_bytes calls.