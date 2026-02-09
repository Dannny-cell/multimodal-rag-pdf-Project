# 🧠 PDF Multimodal RAG Project

This is a high-performance Retrieval-Augmented Generation (RAG) system built with **Streamlit** and **Groq**. It allows you to chat with PDF documents by combining traditional OCR with AI-generated image captions.

## 🚀 Key Features
* **Multimodal Extraction:** Uses `pdf2image` and `pytesseract` for text, and **BLIP** for image captioning.
* **Fast Inference:** Powered by the **Llama-3.3-70b-versatile** model via the Groq API.
* **Vector Search:** Utilizes **ChromaDB** with **BGE-large-en-v1.5** embeddings for semantic retrieval.
* **Advanced Architecture:** Implements an encoder-decoder approach with multi-head attention.

## 🛠️ Setup Instructions
1. **Clone the repo:** `git clone https://github.com/Dannny-cell/multimodal-rag-pdf-Project.git`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Configure API:** Add your `GROQ_API_KEY` to your environment variables or the sidebar.
4. **Run App:** `streamlit run app.py`

## 🏗️ Technical Details
* **Embedding Model:** `BAAI/bge-large-en-v1.5`
* **Image Captioning:** `Salesforce/blip-image-captioning-base`
* **Vector Store:** Persistent ChromaDB instance.
