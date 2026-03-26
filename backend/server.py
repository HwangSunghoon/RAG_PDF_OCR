# backend/server.py
# ---------------------------------------------------------
# FastAPI 백엔드: OCR + 임베딩 + FAISS + 질의응답
# ---------------------------------------------------------

import io
import os
import base64
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import httpx

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import torch
from transformers import pipeline
from huggingface_hub import InferenceClient

app = FastAPI(title="RAG PDF OCR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_LOCAL_MODEL = os.getenv("LLM_LOCAL_MODEL", "google/flan-t5-base")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")
OCR_LANG = os.getenv("OCR_LANG", "kor+eng")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

_embeddings = None
_vectorstore: Optional[FAISS] = None

_local_pipe = None
_hf_client = None

_page_images = {}

IMG_TYPES = {"image/jpeg","image/png","image/webp","image/jpg"}

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings

def get_llm():
    global _local_pipe, _hf_client
    if _local_pipe is None and _hf_client is None:
        if HF_TOKEN:
            _hf_client = InferenceClient(token=HF_TOKEN, model=HF_MODEL)
        else:
            _local_pipe = pipeline(
                "text2text-generation",
                model=LLM_LOCAL_MODEL,
                tokenizer=LLM_LOCAL_MODEL,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
    return _local_pipe, _hf_client

def extract_text_per_page(file_bytes: bytes, ocr_lang: str = OCR_LANG, dpi: int = 250):  # CHANGED
    reader = PdfReader(io.BytesIO(file_bytes))
    num_pages = len(reader.pages)
    texts, raw = [], []
    for i in range(num_pages):
        t = reader.pages[i].extract_text() or ""
        raw.append(t)

    images = convert_from_bytes(file_bytes, dpi=dpi)  # 모든 페이지 렌더

    for i in range(num_pages):
        t = raw[i].strip()
        if len(t) < 50:
            ocr_t = pytesseract.image_to_string(images[i], lang=ocr_lang) or ""
            t = ocr_t.strip() if len(ocr_t.strip()) > len(t) else t
        texts.append(t)
    return texts, images

def to_documents(file: UploadFile, texts: List[str]) -> List[Document]:
    docs = []
    base = file.filename
    for idx, txt in enumerate(texts, start=1):
        if not txt:
            continue
        docs.append(Document(
            page_content=txt,
            metadata={"source": base, "page": idx, "pretty_source": f"{base} p.{idx}"}
        ))
    return docs

def build_or_update_index(docs: List[Document]):
    global _vectorstore
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    if _vectorstore is None:
        _vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        _vectorstore.add_documents(chunks)

def search_context(query: str, k: int):
    if _vectorstore is None:
        return []
    return _vectorstore.similarity_search(query, k=k)

def build_prompt(question: str, ctx_docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(ctx_docs, 1):
        tag = d.metadata.get("pretty_source", d.metadata.get("source", ""))
        parts.append(f"[{i}] {tag}\n{d.page_content}")
    ctx_block = "\n\n".join(parts)
    system = (
        "당신은 대학 강의조교입니다. 아래 '컨텍스트'에 근거해 질문에 답하세요.\n"
        "- 컨텍스트에 없는 내용은 추측하지 말고 '자료에서 답을 찾기 어렵습니다'라고 말하세요.\n"
        "- 핵심 요점은 bullet로 짧게 정리하세요.\n"
    )
    user = f"질문: {question}\n\n컨텍스트:\n{ctx_block}\n"
    return system + "\n" + user

def generate_answer(prompt: str, temperature=0.2, max_new_tokens=512) -> str:
    local_pipe, hf_client = get_llm()
    if hf_client is not None:
        resp = hf_client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            repetition_penalty=1.05,
        )
        return resp.strip()
    else:
        out = local_pipe(
            prompt,
            do_sample=(temperature > 0),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            truncation=True
        )
        return out[0]["generated_text"].strip()

class AskIn(BaseModel):
    question: str
    top_k: int = 4

class AskOut(BaseModel):
    answer: str
    sources: List[str]

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):  # CHANGED
    docs_all: List[Document] = []
    for f in files:
        b = await f.read()
        ctype = (f.content_type or "").lower()
        fname = f.filename

        # 1) PDF
        if ctype == "application/pdf" or fname.lower().endswith(".pdf"):
            texts, images = extract_text_per_page(b, ocr_lang=OCR_LANG)
            # 텍스트 문서화
            docs_all.extend(to_documents(f, texts))
            # 페이지 이미지 보관
            for idx, img in enumerate(images, start=1):
                _page_images[(fname, idx)] = img

        # 2) 단일 이미지 파일
        elif ctype in IMG_TYPES or fname.lower().endswith((".jpg",".jpeg",".png",".webp")):
            img = Image.open(io.BytesIO(b)).convert("RGB")
            t = (pytesseract.image_to_string(img, lang=OCR_LANG) or "").strip()
            if t:
                docs_all.append(Document(
                    page_content=t,
                    metadata={"source": fname, "page": 1, "pretty_source": f"{fname} p.1"}
                ))
            _page_images[(fname, 1)] = img

        else:
            # 스킵(필요시 에러 처리)
            pass

    if docs_all:
        build_or_update_index(docs_all)
    return {"status": "ok", "pages_indexed": len(docs_all)}

# ---- 멀티모달 질문 엔드포인트 추가 ----
def _docs_for_images_temp(files: List[UploadFile]) -> List[Document]:
    docs = []
    for f in files:
        b = f.file.read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        ocr_t = (pytesseract.image_to_string(img, lang=OCR_LANG) or "").strip()
        if ocr_t:
            docs.append(Document(
                page_content=ocr_t,
                metadata={"source": f.filename, "page": 1, "pretty_source": f"{f.filename} p.1"}
            ))
        _page_images[(f.filename, 1)] = img
    return docs

def _encode_images_to_base64(pil_images: List[Image.Image]) -> List[str]:
    arr = []
    for im in pil_images:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        arr.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return arr

async def call_minicpm_llamacpp(prompt: str, pil_images: List[Image.Image], max_new_tokens=512, temperature=0.2) -> Optional[str]:
    """
    llama.cpp 서버(OpenAI 호환 /v1/chat/completions)로 MiniCPM-V 2.6 멀티모달 호출.
    * LLM_HTTP_URL=http://llm:8080  (docker-compose로 제공)
    * 서버는 MiniCPM-V 2.6 GGUF(비전) 로드 필요.
    """
    base = os.getenv("LLM_HTTP_URL", "").rstrip("/")
    if not base:
        return None

    b64s = _encode_images_to_base64(pil_images)
    # OpenAI 호환 형식: text + image 함께 전달
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            for b64 in b64s
        ]
    }]

    payload = {
        "model": "minicpm-v-2.6",  # 임의 식별자(서버측은 단일모델이므로 무시 가능)
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens
    }

    url = f"{base}/v1/chat/completions"
    timeout = httpx.Timeout(120.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        if r.status_code != 200:
            return None
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

@app.post("/ask_mm")
async def ask_mm(
    question: str = Form(...),
    top_k: int = Form(4),
    images: List[UploadFile] = File(default=[])
):
    # 1) 인덱스 컨텍스트
    ctx = search_context(question, k=top_k) or []

    # 2) 즉석 이미지 첨부 시, OCR 텍스트를 컨텍스트에 추가
    temp_docs = []
    if images:
        temp_docs = _docs_for_images_temp(images)
        ctx = list(ctx) + temp_docs

    prompt = build_prompt(question, ctx)  # 기존 프롬프트 재사용:contentReference[oaicite:6]{index=6}

    # 3) 관련 페이지/첨부 이미지 수집
    mm_images = []
    for d in ctx:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in _page_images:
            mm_images.append(_page_images[key])

    # 4) 멀티모달 LLM 호출 → 실패 시 텍스트 LLM으로 폴백
    mm_answer = await call_minicpm_llamacpp(prompt, mm_images) if mm_images else None
    answer = mm_answer if mm_answer is not None else generate_answer(prompt)

    # 5) 출처
    seen, srcs = set(), []
    for d in ctx:
        tag = d.metadata.get("pretty_source", d.metadata.get("source", ""))
        if tag and tag not in seen:
            srcs.append(tag); seen.add(tag)

    return {"answer": answer, "sources": srcs}

@app.post("/ask", response_model=AskOut)
async def ask(payload: AskIn):
    ctx = search_context(payload.question, k=payload.top_k)
    prompt = build_prompt(payload.question, ctx)
    answer = generate_answer(prompt)
    seen, srcs = set(), []
    for d in ctx:
        tag = d.metadata.get("pretty_source", d.metadata.get("source", ""))
        if tag not in seen:
            srcs.append(tag)
            seen.add(tag)
    return AskOut(answer=answer, sources=srcs)

@app.get("/health")
async def health():
    return {"status": "ok", "indexed": _vectorstore is not None}
