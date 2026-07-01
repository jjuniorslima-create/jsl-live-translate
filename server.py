"""
JSL Live Translate — Servidor Web para Telemóvel
FastAPI + WebSocket → Gemini Live Translate → Browser
"""
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

load_dotenv()

INPUT_RATE = 16000
MODEL = "gemini-3.5-live-translate-preview"

LANGUAGES = [
    ("Português (PT)", "pt"),
    ("Inglês (EN)",    "en"),
    ("Espanhol (ES)",  "es"),
    ("Francês (FR)",   "fr"),
    ("Alemão (DE)",    "de"),
    ("Italiano (IT)",  "it"),
    ("Japonês (JA)",   "ja"),
    ("Coreano (KO)",   "ko"),
    ("Chinês (ZH)",    "zh"),
    ("Árabe (AR)",     "ar"),
    ("Hindi (HI)",     "hi"),
    ("Russo (RU)",     "ru"),
    ("Holandês (NL)",  "nl"),
    ("Polonês (PL)",   "pl"),
    ("Turco (TR)",     "tr"),
]

app = FastAPI(title="JSL Live Translate")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/languages")
async def languages():
    return [{"label": l, "code": c} for l, c in LANGUAGES]


@app.websocket("/ws")
async def ws_translate(websocket: WebSocket):
    await websocket.accept()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        await websocket.send_text(json.dumps({
            "type": "error",
            "msg": "GEMINI_API_KEY não configurada no servidor."
        }))
        await websocket.close()
        return

    # Aguarda mensagem inicial com o idioma alvo
    try:
        init_raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        init = json.loads(init_raw)
        target_lang = init.get("lang", "en")
    except Exception:
        await websocket.send_text(json.dumps({"type": "error", "msg": "Handshake inválido."}))
        await websocket.close()
        return

    await websocket.send_text(json.dumps({"type": "status", "msg": f"A conectar ao Gemini ({target_lang.upper()})..."}))

    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    stop_event = asyncio.Event()

    async def browser_receiver():
        """Recebe áudio (bytes) e controlo (JSON) do browser."""
        try:
            while not stop_event.is_set():
                msg = await websocket.receive()
                if msg.get("bytes"):
                    await audio_queue.put(msg["bytes"])
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    if data.get("type") == "stop":
                        break
        except (WebSocketDisconnect, Exception):
            pass
        stop_event.set()
        await audio_queue.put(None)

    async def gemini_send(session):
        """Envia áudio do browser para o Gemini."""
        while not stop_event.is_set():
            chunk = await audio_queue.get()
            if chunk is None:
                break
            try:
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                )
            except Exception:
                break

    async def gemini_recv(session):
        """Recebe áudio traduzido e transcrições do Gemini e envia ao browser."""
        try:
            async for resp in session.receive():
                if stop_event.is_set():
                    break
                sc = resp.server_content
                if not sc:
                    continue
                if sc.input_transcription and sc.input_transcription.text:
                    await websocket.send_text(json.dumps({
                        "type": "transcript_in",
                        "text": sc.input_transcription.text,
                        "final": bool(getattr(sc.input_transcription, "finished", False))
                    }))
                if sc.output_transcription and sc.output_transcription.text:
                    await websocket.send_text(json.dumps({
                        "type": "transcript_out",
                        "text": sc.output_transcription.text,
                        "final": bool(getattr(sc.output_transcription, "finished", False))
                    }))
                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            await websocket.send_bytes(part.inline_data.data)
        except Exception:
            pass

    client = genai.Client(api_key=api_key)
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        translation_config=types.TranslationConfig(
            target_language_code=target_lang,
            echo_target_language=False,
        ),
    )

    try:
        async with client.aio.live.connect(model=MODEL, config=cfg) as session:
            await websocket.send_text(json.dumps({"type": "status", "msg": "Conectado! A traduzir..."}))
            recv_task = asyncio.create_task(browser_receiver())
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(gemini_send(session)),
                    asyncio.create_task(gemini_recv(session)),
                    recv_task,
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "msg": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.send_text(json.dumps({"type": "status", "msg": "Sessão terminada."}))
        except Exception:
            pass
