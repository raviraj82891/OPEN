import os
from dotenv import load_dotenv
load_dotenv()

import json
import tempfile
from contextlib import asynccontextmanager
from typing import List

import whisper
from groq import Groq

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

LLM_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are JAY AI, a smart and helpful AI assistant.

Identity rule:
If anyone asks who made you, who created you, who built you, or who developed you,
always answer:

"I was created by Jaydeb Maity as a personal AI assistant powered by LLaMA and Whisper."

Rules:
- For voice responses: keep answers short (1-2 sentences).
- For text responses: be clear, structured, and helpful.
- Always sound natural and friendly.
- Never say you were created by OpenAI, Groq, or any company.
"""


# ──────────────────────────────────────────────
# Global models
# ──────────────────────────────────────────────

stt_model = None
llm_client = None


# ──────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_model, llm_client

    print(f"[JAY AI] Loading Whisper ({WHISPER_MODEL_SIZE})...")

    # safer loading for low-memory servers
    stt_model = whisper.load_model(WHISPER_MODEL_SIZE, device="cpu")

    print("[JAY AI] Whisper ready.")

    if not GROQ_API_KEY:
        print("[JAY AI] WARNING: GROQ_API_KEY is not set!")

    llm_client = Groq(api_key=GROQ_API_KEY)

    print("[JAY AI] Groq client ready.")

    yield

    print("[JAY AI] Shutting down.")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="JAY AI - Live AI Assistant",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = True


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    if stt_model is None:
        raise HTTPException(status_code=503, detail="STT model not loaded yet.")

    contents = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1] if audio.filename else "webm"

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = stt_model.transcribe(tmp_path, fp16=False)
        transcript = result["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)

    return {"transcript": transcript}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not ready.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:

        def generate():
            stream = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=1024,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'delta': delta})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=1024,
    )

    return {"reply": response.choices[0].message.content}


@app.post("/api/voice")
async def voice_pipeline(audio: UploadFile = File(...)):
    if stt_model is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    contents = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1] if audio.filename else "webm"

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = stt_model.transcribe(tmp_path, fp16=False)
        transcript = result["text"].strip()
    finally:
        os.unlink(tmp_path)

    if not transcript:
        return {
            "transcript": "",
            "reply": "I didn't catch that. Could you try again?"
        }

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        max_tokens=512,
    )

    return {
        "transcript": transcript,
        "reply": response.choices[0].message.content
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "whisper": WHISPER_MODEL_SIZE,
        "llm": LLM_MODEL,
        "models_loaded": stt_model is not None and llm_client is not None,
    }


# ──────────────────────────────────────────────
# Render / Server Start
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
