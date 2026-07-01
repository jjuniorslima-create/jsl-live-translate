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
                    await audio_queue.put(("audio", msg["bytes"]))
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    if data.get("type") == "stop":
                        break
                    elif data.get("type") == "turn_start":
                        await audio_queue.put(("turn_start", None))
                    elif data.get("type") == "turn_end":
                        await audio_queue.put(("turn_end", None))
        except (WebSocketDisconnect, Exception):
            pass
        stop_event.set()
        await audio_queue.put(None)

    async def gemini_send(session):
        """Envia áudio do browser para o Gemini e marca início/fim de turno explicitamente."""
        while not stop_event.is_set():
            item = await audio_queue.get()
            if item is None:
                break
            kind, payload = item
            try:
                if kind == "audio":
                    await session.send_realtime_input(
                        audio=types.Blob(data=payload, mime_type="audio/pcm;rate=16000")
                    )
                elif kind == "turn_start":
                    print("[gemini_send] turn_start", flush=True)
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                elif kind == "turn_end":
                    print("[gemini_send] turn_end", flush=True)
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
            except Exception as e:
                print(f"[gemini_send] erro: {type(e).__name__}: {e}", flush=True)
                break

    async def gemini_recv(session):
        """Recebe áudio traduzido e transcrições do Gemini e envia ao browser.

        session.receive() é um generator contínuo válido para toda a sessão —
        entrega respostas de vários turnos ao longo do tempo, não deve ser
        recriado a cada turno (fazê-lo interrompe o consumo da ligação).
        Usamos wait_for por mensagem só para detectar silêncio prolongado
        sem travar a app.
        """
        RECV_TIMEOUT = 30.0
        try:
            gen = session.receive()
            while not stop_event.is_set():
                try:
                    resp = await asyncio.wait_for(gen.__anext__(), timeout=RECV_TIMEOUT)
                except StopAsyncIteration:
                    print("[gemini_recv] sessão Gemini terminou o stream", flush=True)
                    break
                except asyncio.TimeoutError:
                    print("[gemini_recv] timeout à espera de resposta do Gemini", flush=True)
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "msg": "Sem resposta do Gemini — tente falar novamente."
                        }))
                    except Exception:
                        pass
                    continue

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
        except Exception as e:
            print(f"[gemini_recv] erro: {type(e).__name__}: {e}", flush=True)
            if stop_event.is_set():
                return
            try:
                await websocket.send_text(json.dumps({"type": "error", "msg": f"Gemini: {e}"}))
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
        # VAD automático desligado: o turno é controlado manualmente pelo
        # botão push-to-talk (activity_start / activity_end), senão o
        # audio_stream_end/VAD do Gemini corta a fala de forma imprevisível.
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
        ),
    )

    try:
        async with client.aio.live.connect(model=MODEL, config=cfg) as session:
            await websocket.send_text(json.dumps({"type": "status", "msg": "Conectado! A traduzir..."}))

            browser_task = asyncio.create_task(browser_receiver())
            send_task    = asyncio.create_task(gemini_send(session))
            recv_task    = asyncio.create_task(gemini_recv(session))

            # Aguarda apenas o browser desligar (stop ou disconnect)
            # gemini_send/recv podem terminar por si e não derrubam a sessão
            await browser_task

            for t in [send_task, recv_task]:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
    except Exception as e:
        print(f"[ws_translate] erro sessão: {type(e).__name__}: {e}", flush=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "msg": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.send_text(json.dumps({"type": "status", "msg": "Sessão terminada."}))
        except Exception:
            pass
