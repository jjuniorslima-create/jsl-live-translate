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

from comum import LANGUAGES, MODEL, criar_config_live

load_dotenv()

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

    await websocket.send_text(json.dumps({
        "type": "status",
        "msg": f"A conectar ao Gemini ({target_lang.upper()})...",
        "ready": False,
    }))

    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    stop_event = asyncio.Event()
    vistos_mime: set[str] = set()   # diagnóstico: formato real do áudio do Gemini

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
        """Envia áudio do browser para o Gemini (VAD automático do Gemini decide os turnos).

        NÃO enviar silêncio nas pausas. Testado a 26/07/2026: alimentar
        silêncio ao ritmo real faz o Gemini devolver áudio continuamente
        (~250 ms de áudio a cada 250 ms), o que enche a fila de reprodução
        do browser e é a parte mais cara da factura. As quedas de ligação
        que o silêncio tentava evitar tinham outra causa — o ping do
        uvicorn a expirar aos 20 s — já resolvida no Procfile/railway.toml.
        """
        while not stop_event.is_set():
            chunk = await audio_queue.get()
            if chunk is None:
                break
            try:
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                )
            except Exception as e:
                print(f"[gemini_send] erro: {type(e).__name__}: {e}", flush=True)
                break

    async def gemini_recv(session):
        """Recebe áudio traduzido e transcrições do Gemini e envia ao browser.

        gemini-3.5-live-translate-preview é "designed for continuous stream
        processing without turn-based interactions" (doc oficial:
        ai.google.dev/gemini-api/docs/live-api/live-translate) — ao contrário
        da Live API genérica (por turnos), aqui session.receive() é chamado
        UMA ÚNICA VEZ e iterado continuamente durante toda a sessão. Recriar
        o generator a cada "turno" (padrão correto só para a Live API
        genérica) interrompe este stream contínuo.
        """
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
                            nonlocal_mime = getattr(part.inline_data, "mime_type", None)
                            if nonlocal_mime and nonlocal_mime not in vistos_mime:
                                vistos_mime.add(nonlocal_mime)
                                print(f"[audio-saida] mime_type do Gemini: {nonlocal_mime} "
                                      f"(bytes por bocado: {len(part.inline_data.data)})", flush=True)
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
    cfg = criar_config_live(target_lang)

    try:
        async with client.aio.live.connect(model=MODEL, config=cfg) as session:
            await websocket.send_text(json.dumps({
                "type": "status",
                "msg": "Conectado! A traduzir...",
                "ready": True,
            }))

            browser_task = asyncio.create_task(browser_receiver())
            send_task    = asyncio.create_task(gemini_send(session))
            recv_task    = asyncio.create_task(gemini_recv(session))

            # Termina à primeira das três que acabar. Se o Gemini fechar a
            # sessão (visto em produção: "received 1001 going away"), esperar
            # só pelo browser deixava a ligação viva mas sem ninguém do outro
            # lado — os botões pareciam bons e nunca chegava tradução.
            # Fechando aqui, o browser detecta a queda e reabre sozinho.
            done, pending = await asyncio.wait(
                [browser_task, send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in pending:
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
