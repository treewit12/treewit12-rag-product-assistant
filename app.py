import os
from pathlib import Path

import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ลำดับโมเดลที่จะลอง (ถ้าตัวแรกใช้ไม่ได้ จะลองตัวถัดไปอัตโนมัติ)
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash"]

# st.set_page_config ต้องเป็นคำสั่ง Streamlit คำสั่งแรกเสมอ
st.set_page_config(page_title="RAG Product Assistant", page_icon="🤖", layout="wide")

st.title("🤖 RAG Product Assistant")
st.caption("ผู้ช่วยตอบคำถามจากคลังเอกสารคู่มือสินค้า ด้วย Retrieval-Augmented Generation")


# ---------------------------------------------------------------- Embedding / KB
@st.cache_resource(show_spinner="กำลังโหลดโมเดล Embedding (ครั้งแรกอาจใช้เวลาสักครู่)...")
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def clean_text(text):
    return " ".join(text.replace("\ufeff", "").split())


def chunk_text(text, chunk_size=700, overlap=120):
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


@st.cache_resource(show_spinner="กำลังสร้างฐานความรู้ (FAISS)...")
def build_knowledge_base():
    model = load_embedding_model()
    chunks, sources = [], []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append(chunk)
            sources.append(f"{file_path.name} — Chunk {i + 1}")

    if not chunks:
        return [], [], None

    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return chunks, sources, index


def retrieve(query, chunks, sources, index, model, top_k=4):
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, ids = index.search(q, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({"score": float(score), "text": chunks[idx], "source": sources[idx]})
    return results


def make_prompt(question, results):
    context = "\n\n".join(f"[SOURCE: {r['source']}]\n{r['text']}" for r in results)

    return f"""คุณคือ RAG Product Assistant
หน้าที่ของคุณคือตอบคำถามโดยใช้ข้อมูลใน CONTEXT เท่านั้น

กฎสำคัญ:
1. ห้ามใช้ความรู้ภายนอก CONTEXT มาแต่งคำตอบ
2. ถ้า CONTEXT ไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบว่า "ไม่พบข้อมูลในเอกสารความรู้"
3. ตอบเป็นภาษาเดียวกับคำถามเมื่อทำได้
4. ตอบให้กระชับ ชัดเจน และเป็นข้อๆ เมื่อเหมาะสม
5. หากมีข้อมูลจากเอกสาร ให้ระบุชื่อแหล่งข้อมูลในคำตอบ
6. ห้ามสร้างรายละเอียด เช่น ราคา ระยะเวลารับประกัน หรือวิธีแก้ปัญหา หากไม่มีใน CONTEXT

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# ---------------------------------------------------------------- API key
def find_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


# ---------------------------------------------------------------- Load
embedding_model = load_embedding_model()
chunks, sources, index = build_knowledge_base()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📚 Knowledge Base")
    st.write(f"เอกสาร: **{len(list(DATA_DIR.glob('*.txt')))} ไฟล์**")
    st.write(f"Chunks: **{len(chunks)}**")
    st.write("Vector DB: **FAISS**")
    st.write("Embedding: **paraphrase-multilingual-MiniLM-L12-v2**")
    st.divider()

    api_key = find_api_key()
    if not api_key:
        api_key = st.text_input("GEMINI_API_KEY", type="password",
                                help="ใส่คีย์ที่นี่ หรือกำหนดใน Secrets")
    if st.button("🗑️ ล้างประวัติแชต"):
        st.session_state.messages = []
        st.rerun()

if not chunks or index is None:
    st.error(f"ไม่พบไฟล์ .txt ในโฟลเดอร์ {DATA_DIR}")
    st.stop()

if not api_key:
    st.warning("ยังไม่ได้ตั้งค่า GEMINI_API_KEY — ใส่ในแถบด้านซ้าย หรือใน Secrets")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------------------------------------------------------- Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📎 เอกสารอ้างอิง"):
                for source in message["sources"]:
                    st.write(f"• {source}")

question = st.chat_input("พิมพ์คำถามเกี่ยวกับสินค้า เช่น วิธีแก้ปัญหา Wi-Fi หลุดบ่อย")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results = retrieve(question, chunks, sources, index, embedding_model, top_k=4)
    relevant = [r for r in results if r["score"] >= 0.30]

    with st.chat_message("assistant"):
        if not relevant:
            answer = "ไม่พบข้อมูลในเอกสารความรู้"
            used_sources = []
            st.markdown(answer)
        else:
            try:
                answer = None
                last_error = None
                for model_name in GEMINI_MODELS:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=make_prompt(question, relevant),
                            config=types.GenerateContentConfig(
                                system_instruction="You are a precise RAG assistant. Follow the user's grounding rules exactly.",
                                temperature=0.1,
                                max_output_tokens=1500,
                                thinking_config=types.ThinkingConfig(thinking_level="low"),
                            ),
                        )
                        answer = (response.text or "ไม่พบข้อมูลในเอกสารความรู้").strip()
                        break
                    except Exception as e:
                        last_error = e
                if answer is None:
                    raise last_error
                used_sources = [f"{r['source']} — similarity {r['score']:.3f}" for r in relevant]
                st.markdown(answer)
                with st.expander("📎 เอกสารอ้างอิง"):
                    for s in used_sources:
                        st.write(f"• {s}")
            except Exception as e:
                answer = f"เรียก Gemini API ไม่สำเร็จ: {e}"
                used_sources = []
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": used_sources}
    )