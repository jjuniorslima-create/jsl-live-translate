"""
JSL Live Translate — Bidirecional com Gemini 3.5 Live Translate
Modo Saída  : seu mic → traduz → cabo virtual (Zoom ouve no idioma escolhido)
Modo Entrada: cabo virtual (áudio da reunião) → traduz → seus fones
Modo Completo: ambos ao mesmo tempo
"""

import asyncio
import os
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

INPUT_RATE     = 16000
INPUT_CHUNK_MS = 100
INPUT_FRAMES   = INPUT_RATE * INPUT_CHUNK_MS // 1000

OUTPUT_RATE    = 24000

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

RECORDINGS_DIR = Path("gravacoes")
RECORDINGS_DIR.mkdir(exist_ok=True)


# ── Enumeração de dispositivos ─────────────────────────────────────────────────

def get_devices():
    """Retorna (inputs, outputs) como lista de (nome_display, índice)."""
    seen_in, seen_out = set(), set()
    inputs, outputs = [], []
    for i, d in enumerate(sd.query_devices()):
        name = d["name"]
        if d["max_input_channels"] > 0 and d["hostapi"] == 0:  # MME apenas
            key = name.lower()
            if key not in seen_in:
                seen_in.add(key)
                inputs.append((name, i))
        if d["max_output_channels"] > 0 and d["hostapi"] == 0:
            key = name.lower()
            if key not in seen_out:
                seen_out.add(key)
                outputs.append((name, i))
    return inputs, outputs


# ── Núcleo de tradução ─────────────────────────────────────────────────────────

class Translator:
    def __init__(self, label, on_transcript, on_output_audio, on_error):
        self.label = label
        self.on_transcript   = on_transcript
        self.on_output_audio = on_output_audio
        self.on_error        = on_error

        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop   = threading.Event()
        self._restart = False
        self._lang   = "en"

    def start(self, lang: str):
        self._lang = lang
        self._stop.clear()
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def change_language(self, lang: str):
        self._lang = lang
        self._restart = True
        self._q.put(None)

    def push(self, chunk: bytes):
        if not self._stop.is_set():
            self._q.put(chunk)

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._loop())
        except Exception as e:
            self.on_error(self.label, str(e))
        finally:
            loop.close()

    async def _loop(self):
        while not self._stop.is_set():
            self._restart = False
            try:
                await self._session(self._lang)
            except Exception as e:
                if not self._stop.is_set():
                    self.on_error(self.label, str(e))
                    break
            if self._restart and not self._stop.is_set():
                continue
            break

    async def _session(self, lang: str):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY não encontrada no .env")

        client = genai.Client(api_key=api_key)
        cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(
                target_language_code=lang,
                echo_target_language=False,
            ),
        )
        async with client.aio.live.connect(model=MODEL, config=cfg) as session:
            send = asyncio.create_task(self._send(session))
            recv = asyncio.create_task(self._recv(session))
            done, pending = await asyncio.wait(
                [send, recv], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def _send(self, session):
        loop = asyncio.get_event_loop()
        while not self._stop.is_set() and not self._restart:
            chunk = await loop.run_in_executor(None, self._q.get)
            if chunk is None:
                break
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
            )

    async def _recv(self, session):
        async for resp in session.receive():
            if self._stop.is_set() or self._restart:
                break
            sc = resp.server_content
            if not sc:
                continue
            if sc.input_transcription and sc.input_transcription.text:
                self.on_transcript(self.label, "in", sc.input_transcription.text)
            if sc.output_transcription and sc.output_transcription.text:
                self.on_transcript(self.label, "out", sc.output_transcription.text)
            if sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        self.on_output_audio(self.label, part.inline_data.data)


# ── Canal de áudio (entrada ou saída) ─────────────────────────────────────────

class AudioOut:
    """Reproduz e/ou grava PCM 24kHz em um dispositivo de saída."""
    def __init__(self, device_index: int | None):
        self._dev   = device_index
        self._buf   = queue.Queue()
        self._wav   = None
        self._stream= None

    def open(self, wav_path: Path | None = None):
        self._buf = queue.Queue()
        if wav_path:
            self._wav = wave.open(str(wav_path), "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)
            self._wav.setframerate(OUTPUT_RATE)

        def cb(outdata, frames, time, status):
            need = frames * 2
            buf  = b""
            while len(buf) < need:
                try:
                    buf += self._buf.get_nowait()
                except queue.Empty:
                    break
            arr = np.zeros(frames, dtype="int16")
            if buf:
                n = min(len(buf) // 2, frames)
                arr[:n] = np.frombuffer(buf[:n*2], dtype="int16")
                leftover = buf[n*2:]
                if leftover:
                    self._buf.put(leftover)
            outdata[:] = arr.reshape(-1, 1)

        self._stream = sd.OutputStream(
            device=self._dev,
            samplerate=OUTPUT_RATE,
            channels=1,
            dtype="int16",
            blocksize=2048,
            callback=cb,
        )
        self._stream.start()

    def write(self, data: bytes):
        self._buf.put(data)
        if self._wav:
            self._wav.writeframes(data)

    def close(self) -> str | None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._wav:
            self._wav.close()
            path = self._wav.fp.name if hasattr(self._wav, "fp") else None
            self._wav = None
            return path
        return None


# ── App principal ──────────────────────────────────────────────────────────────

BG    = "#1a1a2e"
CARD  = "#16213e"
PANEL = "#0f2040"
BLUE  = "#4cc9f0"
GREEN = "#06d6a0"
RED   = "#ef233c"
GOLD  = "#ffd166"
TEXT  = "#edf2f4"
GRAY  = "#adb5bd"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JSL Live Translate")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self._inputs, self._outputs = get_devices()

        self._tr_out = Translator("out", self._on_transcript, self._on_audio, self._on_error)
        self._tr_in  = Translator("in",  self._on_transcript, self._on_audio, self._on_error)

        self._audio_out_out: AudioOut | None = None  # saída da tradução de saída
        self._audio_out_in:  AudioOut | None = None  # saída da tradução de entrada

        self._stream_mic: sd.InputStream | None = None
        self._stream_meet: sd.InputStream | None = None

        self._active_out  = False
        self._active_in   = False

        self._wav_out_path: Path | None = None
        self._wav_in_path:  Path | None = None

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        f_title = tkfont.Font(family="Segoe UI", size=15, weight="bold")
        f_label = tkfont.Font(family="Segoe UI", size=9)
        f_sec   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        f_btn   = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        f_mono  = tkfont.Font(family="Consolas",  size=9)

        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=16, pady=(18, 2))

        tk.Label(header, text="🎙  JSL Live Translate — Bidirecional",
                 font=f_title, bg=BG, fg=BLUE).pack(side="left")

        tk.Button(header, text="? Ajuda", font=f_label,
                  bg=CARD, fg=BLUE, activebackground=PANEL,
                  activeforeground=BLUE, relief="flat", padx=10, pady=4,
                  cursor="hand2", command=self._abrir_ajuda).pack(side="right")

        tk.Label(self.root, text="Gemini 3.5 Live Translate Preview",
                 font=f_label, bg=BG, fg="#666").pack(pady=(0, 12))

        outer = tk.Frame(self.root, bg=BG)
        outer.pack(padx=16, fill="x")

        # ── Coluna SAÍDA ───────────────────────────────────────────────────────
        col_out = tk.Frame(outer, bg=CARD, padx=14, pady=12)
        col_out.grid(row=0, column=0, padx=(0, 6), sticky="nsew")

        tk.Label(col_out, text="▶  SAÍDA  (você → reunião)",
                 font=f_sec, bg=CARD, fg=GOLD).grid(row=0, column=0, columnspan=2,
                                                      sticky="w", pady=(0, 8))

        tk.Label(col_out, text="Seu microfone:", font=f_label,
                 bg=CARD, fg=TEXT).grid(row=1, column=0, sticky="w")
        self._mic_var = tk.StringVar()
        self._mic_menu = self._make_menu(col_out, self._mic_var,
                                         [n for n, _ in self._inputs], row=1, col=1)

        tk.Label(col_out, text="Traduzir para:", font=f_label,
                 bg=CARD, fg=TEXT).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._lang_out_var = tk.StringVar(value=LANGUAGES[1][0])
        self._make_lang_menu(col_out, self._lang_out_var, row=2, col=1,
                             cmd=self._on_lang_out_change)

        tk.Label(col_out, text="Saída → cabo virtual:", font=f_label,
                 bg=CARD, fg=TEXT).grid(row=3, column=0, sticky="w", pady=(6, 0))
        self._vout_var = tk.StringVar()
        self._vout_menu = self._make_menu(col_out, self._vout_var,
                                          [n for n, _ in self._outputs], row=3, col=1)

        self._btn_out = tk.Button(col_out, text="▶ Iniciar Saída", font=f_btn,
                                  bg=GOLD, fg="#000", activebackground="#e6bc00",
                                  relief="flat", padx=10, pady=8, cursor="hand2",
                                  command=self._toggle_out)
        self._btn_out.grid(row=4, column=0, columnspan=2, pady=(14, 0), sticky="ew")

        self._status_out = tk.Label(col_out, text="Parado", font=f_label,
                                    bg=CARD, fg=GRAY)
        self._status_out.grid(row=5, column=0, columnspan=2, pady=(4, 0))

        # ── Coluna ENTRADA ─────────────────────────────────────────────────────
        col_in = tk.Frame(outer, bg=CARD, padx=14, pady=12)
        col_in.grid(row=0, column=1, padx=(6, 0), sticky="nsew")

        tk.Label(col_in, text="◀  ENTRADA  (reunião → você)",
                 font=f_sec, bg=CARD, fg=GREEN).grid(row=0, column=0, columnspan=2,
                                                       sticky="w", pady=(0, 8))

        tk.Label(col_in, text="Áudio da reunião:", font=f_label,
                 bg=CARD, fg=TEXT).grid(row=1, column=0, sticky="w")
        self._meet_var = tk.StringVar()
        self._meet_menu = self._make_menu(col_in, self._meet_var,
                                          [n for n, _ in self._inputs], row=1, col=1)

        tk.Label(col_in, text="Traduzir para:", font=f_label,
                 bg=CARD, fg=TEXT).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._lang_in_var = tk.StringVar(value=LANGUAGES[0][0])
        self._make_lang_menu(col_in, self._lang_in_var, row=2, col=1,
                             cmd=self._on_lang_in_change)

        tk.Label(col_in, text="Saída → seus fones:", font=f_label,
                 bg=CARD, fg=TEXT).grid(row=3, column=0, sticky="w", pady=(6, 0))
        self._ear_var = tk.StringVar()
        self._ear_menu = self._make_menu(col_in, self._ear_var,
                                         [n for n, _ in self._outputs], row=3, col=1)

        self._btn_in = tk.Button(col_in, text="▶ Iniciar Entrada", font=f_btn,
                                 bg=GREEN, fg="#000", activebackground="#05c091",
                                 relief="flat", padx=10, pady=8, cursor="hand2",
                                 command=self._toggle_in)
        self._btn_in.grid(row=4, column=0, columnspan=2, pady=(14, 0), sticky="ew")

        self._status_in = tk.Label(col_in, text="Parado", font=f_label,
                                   bg=CARD, fg=GRAY)
        self._status_in.grid(row=5, column=0, columnspan=2, pady=(4, 0))

        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        # ── Transcrições ───────────────────────────────────────────────────────
        t_frame = tk.Frame(self.root, bg=CARD, padx=12, pady=10)
        t_frame.pack(fill="both", expand=True, padx=16, pady=(12, 6))

        tk.Label(t_frame,
                 text="Transcrições   🟡 você fala → tradução   🟢 reunião → tradução",
                 font=f_label, bg=CARD, fg="#666").pack(anchor="w")

        self._log = tk.Text(t_frame, height=9, width=72, bg="#0d1117", fg=TEXT,
                            font=f_mono, relief="flat", state="disabled",
                            wrap="word")
        self._log.pack(fill="both", expand=True, pady=(4, 0))
        self._log.tag_config("out_in",  foreground=GOLD)
        self._log.tag_config("out_out", foreground="#ffeb99")
        self._log.tag_config("in_in",   foreground=GREEN)
        self._log.tag_config("in_out",  foreground="#99f5d5")
        self._log.tag_config("sys",     foreground=BLUE)
        self._log.tag_config("err",     foreground=RED)

        # ── Arquivos salvos ────────────────────────────────────────────────────
        self._saved_label = tk.Label(self.root, text="", font=f_label,
                                     bg=BG, fg=GREEN, wraplength=500)
        self._saved_label.pack(pady=(0, 14))

        # seleção padrão
        self._auto_select_defaults()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _make_menu(self, parent, var, options, row, col):
        if not options:
            options = ["(nenhum dispositivo)"]
        var.set(options[0])
        m = tk.OptionMenu(parent, var, *options)
        m.config(bg=PANEL, fg=TEXT, activebackground=BLUE, activeforeground="#000",
                 relief="flat", font=tkfont.Font(family="Segoe UI", size=9),
                 width=22, highlightthickness=0)
        m["menu"].config(bg=PANEL, fg=TEXT, activebackground=BLUE, activeforeground="#000")
        m.grid(row=row, column=col, sticky="w", padx=(8, 0), pady=(2, 0))
        return m

    def _make_lang_menu(self, parent, var, row, col, cmd):
        names = [l[0] for l in LANGUAGES]
        m = tk.OptionMenu(parent, var, *names, command=cmd)
        m.config(bg=PANEL, fg=TEXT, activebackground=BLUE, activeforeground="#000",
                 relief="flat", font=tkfont.Font(family="Segoe UI", size=9),
                 width=22, highlightthickness=0)
        m["menu"].config(bg=PANEL, fg=TEXT, activebackground=BLUE, activeforeground="#000")
        m.grid(row=row, column=col, sticky="w", padx=(8, 0), pady=(2, 0))
        return m

    def _auto_select_defaults(self):
        """Tenta pré-selecionar dispositivos óbvios."""
        in_names  = [n for n, _ in self._inputs]
        out_names = [n for n, _ in self._outputs]

        # mic padrão
        for keyword in ["external", "microfone", "microphone", "mic"]:
            for n in in_names:
                if keyword in n.lower():
                    self._mic_var.set(n)
                    break

        # cabo virtual saída (CABLE Input)
        for n in out_names:
            if "cable" in n.lower() and "input" in n.lower():
                self._vout_var.set(n)
                break

        # cabo virtual entrada (Hi-Fi CABLE Output ou CABLE Output)
        for n in in_names:
            if "cable" in n.lower() and "output" in n.lower():
                self._meet_var.set(n)
                break

        # fones de ouvido
        for keyword in ["headphone", "fone", "auscultadores"]:
            for n in out_names:
                if keyword in n.lower():
                    self._ear_var.set(n)
                    break

    # ── Controles ──────────────────────────────────────────────────────────────

    def _toggle_out(self):
        if not self._active_out:
            self._start_out()
        else:
            self._stop_out()

    def _toggle_in(self):
        if not self._active_in:
            self._start_in()
        else:
            self._stop_in()

    def _start_out(self):
        self._active_out = True
        self._btn_out.config(text="■ Parar & Salvar", bg=RED,
                             activebackground="#c81d2d", fg="white")
        self._status_out.config(text="● Traduzindo...", fg=RED)

        lang = self._get_lang_code(self._lang_out_var)
        out_dev = self._get_out_index(self._vout_var)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._wav_out_path = RECORDINGS_DIR / f"saida_{lang}_{ts}.wav"

        self._audio_out_out = AudioOut(out_dev)
        self._audio_out_out.open(wav_path=self._wav_out_path)

        self._tr_out.start(lang)
        self._stream_mic = self._open_input(self._mic_var, self._tr_out.push)
        self._write_log("sys", f"Saída iniciada → {self._lang_out_var.get()}")

    def _stop_out(self):
        self._active_out = False
        self._btn_out.config(text="▶ Iniciar Saída", bg=GOLD,
                             activebackground="#e6bc00", fg="#000")
        self._status_out.config(text="Parado", fg=GRAY)

        self._close_stream(self._stream_mic)
        self._stream_mic = None
        self._tr_out.stop()

        if self._audio_out_out:
            self._audio_out_out.close()
            self._audio_out_out = None

        if self._wav_out_path and self._wav_out_path.exists():
            self._saved_label.config(
                text=f"Salvo: {self._wav_out_path}"
            )
            self._write_log("sys", f"Gravação saída → {self._wav_out_path.name}")
        self._wav_out_path = None

    def _start_in(self):
        self._active_in = True
        self._btn_in.config(text="■ Parar & Salvar", bg=RED,
                            activebackground="#c81d2d", fg="white")
        self._status_in.config(text="● Traduzindo...", fg=RED)

        lang = self._get_lang_code(self._lang_in_var)
        ear_dev = self._get_out_index(self._ear_var)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._wav_in_path = RECORDINGS_DIR / f"entrada_{lang}_{ts}.wav"

        self._audio_out_in = AudioOut(ear_dev)
        self._audio_out_in.open(wav_path=self._wav_in_path)

        self._tr_in.start(lang)
        self._stream_meet = self._open_input(self._meet_var, self._tr_in.push)
        self._write_log("sys", f"Entrada iniciada → {self._lang_in_var.get()}")

    def _stop_in(self):
        self._active_in = False
        self._btn_in.config(text="▶ Iniciar Entrada", bg=GREEN,
                            activebackground="#05c091", fg="#000")
        self._status_in.config(text="Parado", fg=GRAY)

        self._close_stream(self._stream_meet)
        self._stream_meet = None
        self._tr_in.stop()

        if self._audio_out_in:
            self._audio_out_in.close()
            self._audio_out_in = None

        if self._wav_in_path and self._wav_in_path.exists():
            self._saved_label.config(
                text=f"Salvo: {self._wav_in_path}"
            )
            self._write_log("sys", f"Gravação entrada → {self._wav_in_path.name}")
        self._wav_in_path = None

    def _on_lang_out_change(self, _=None):
        if self._active_out:
            self._tr_out.change_language(self._get_lang_code(self._lang_out_var))
            self._write_log("sys", f"Saída → idioma alterado para {self._lang_out_var.get()}")

    def _on_lang_in_change(self, _=None):
        if self._active_in:
            self._tr_in.change_language(self._get_lang_code(self._lang_in_var))
            self._write_log("sys", f"Entrada → idioma alterado para {self._lang_in_var.get()}")

    # ── Áudio helpers ──────────────────────────────────────────────────────────

    def _open_input(self, var: tk.StringVar, push_fn) -> sd.InputStream:
        dev = self._get_in_index(var)

        def cb(indata, frames, time, status):
            push_fn(indata.tobytes())

        stream = sd.InputStream(
            device=dev,
            samplerate=INPUT_RATE,
            channels=1,
            dtype="int16",
            blocksize=INPUT_FRAMES,
            callback=cb,
        )
        stream.start()
        return stream

    def _close_stream(self, stream):
        if stream:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _get_in_index(self, var: tk.StringVar) -> int | None:
        name = var.get()
        for n, i in self._inputs:
            if n == name:
                return i
        return None

    def _get_out_index(self, var: tk.StringVar) -> int | None:
        name = var.get()
        for n, i in self._outputs:
            if n == name:
                return i
        return None

    def _get_lang_code(self, var: tk.StringVar) -> str:
        name = var.get()
        for label, code in LANGUAGES:
            if label == name:
                return code
        return "en"

    # ── Callbacks thread-safe ─────────────────────────────────────────────────

    def _on_transcript(self, label: str, direction: str, text: str):
        self.root.after(0, self._write_log, f"{label}_{direction}", text)

    def _on_audio(self, label: str, data: bytes):
        if label == "out" and self._audio_out_out:
            self._audio_out_out.write(data)
        elif label == "in" and self._audio_out_in:
            self._audio_out_in.write(data)

    def _on_error(self, label: str, msg: str):
        self.root.after(0, self._write_log, "err", f"[{label}] {msg}")

    def _write_log(self, kind: str, text: str):
        tag_map = {
            "out_in":  ("out_in",  "🟡 PT → "),
            "out_out": ("out_out", "   TR → "),
            "in_in":   ("in_in",  "🟢 MT → "),
            "in_out":  ("in_out", "   TR → "),
            "sys":     ("sys",    "• "),
            "err":     ("err",    "⚠ "),
        }
        tag, prefix = tag_map.get(kind, ("sys", ""))
        self._log.config(state="normal")
        self._log.insert("end", prefix + text + "\n", tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _abrir_ajuda(self):
        win = tk.Toplevel(self.root)
        win.title("Como usar — JSL Live Translate")
        win.configure(bg=BG)
        win.resizable(False, False)

        f_sec  = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        f_sub  = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        f_body = tkfont.Font(family="Segoe UI", size=9)
        f_mono = tkfont.Font(family="Consolas",  size=9)

        # scroll
        canvas = tk.Canvas(win, bg=BG, highlightthickness=0, width=600, height=520)
        scroll = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas, bg=BG)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(
            int(-1 * (e.delta / 120)), "units"))

        def sec(texto, cor=BLUE):
            tk.Label(frame, text=texto, font=f_sec, bg=BG, fg=cor,
                     anchor="w").pack(fill="x", padx=20, pady=(18, 4))
            tk.Frame(frame, bg=cor, height=1).pack(fill="x", padx=20, pady=(0, 8))

        def sub(texto, cor=GOLD):
            tk.Label(frame, text=texto, font=f_sub, bg=BG, fg=cor,
                     anchor="w").pack(fill="x", padx=28, pady=(10, 2))

        def body(texto):
            tk.Label(frame, text=texto, font=f_body, bg=BG, fg=TEXT,
                     anchor="w", justify="left", wraplength=540).pack(
                fill="x", padx=36, pady=1)

        def code(texto):
            tk.Label(frame, text=texto, font=f_mono, bg=CARD, fg=GREEN,
                     anchor="w", padx=8, pady=4).pack(
                fill="x", padx=36, pady=(2, 4))

        # ── Pré-requisito ──────────────────────────────────────────────────────
        sec("⚙️  Pré-requisito (fazer uma vez só)")
        body("O VB-Audio Virtual Cable precisa estar instalado (vb-audio.com/Cable).")
        body("Após instalar, reinicie o PC. Os dispositivos aparecem automaticamente.")

        # ── Como funciona ──────────────────────────────────────────────────────
        sec("🔁  Como funciona")
        sub("Modo Saída (você → participantes)", GOLD)
        body("Seu mic real → JSL traduz → CABLE Input → plataforma transmite traduzido")
        sub("Modo Entrada (participantes → você)", GREEN)
        body("Plataforma → CABLE Output → JSL traduz → seus fones em português")
        sub("Modo Bidirecional (os dois ao mesmo tempo)", BLUE)
        body("Inicie Saída e Entrada juntos. Use fones para evitar eco.")

        # ── ZOOM ──────────────────────────────────────────────────────────────
        sec("💼  Zoom")
        sub("Configuração no Zoom (Configurações → Áudio):", TEXT)
        code("Microfone   →  CABLE Output (VB-Audio Virtual Cable)")
        code("Alto-falante →  CABLE Input  (VB-Audio Virtual Cable)")
        sub("Configuração no app — Saída:", GOLD)
        body("• Seu microfone    → External Microphone")
        body("• Traduzir para    → idioma dos participantes")
        body("• Saída cabo virtual → CABLE Input")
        sub("Configuração no app — Entrada (bidirecional):", GREEN)
        body("• Áudio da reunião → CABLE Output")
        body("• Traduzir para    → Português (PT)")
        body("• Saída seus fones → Headphones")

        # ── GOOGLE MEET ───────────────────────────────────────────────────────
        sec("📹  Google Meet")
        sub("Configuração no Meet (⋮ → Configurações → Áudio):", TEXT)
        code("Microfone   →  CABLE Output (VB-Audio Virtual Cable)")
        code("Alto-falante →  CABLE Input  (VB-Audio Virtual Cable)")
        sub("Configuração no app:", GOLD)
        body("Mesma configuração do Zoom acima.")

        # ── MICROSOFT TEAMS ───────────────────────────────────────────────────
        sec("👥  Microsoft Teams")
        sub("Configuração no Teams (⚙️ → Dispositivos):", TEXT)
        code("Microfone   →  CABLE Output (VB-Audio Virtual Cable)")
        code("Alto-falante →  CABLE Input  (VB-Audio Virtual Cable)")
        sub("Configuração no app:", GOLD)
        body("Mesma configuração do Zoom acima.")

        # ── YOUTUBE LIVE ──────────────────────────────────────────────────────
        sec("▶️  YouTube Live (via OBS Studio)")
        sub("No OBS — Fontes de Áudio:", TEXT)
        body("1. Remova o microfone padrão das fontes")
        body("2. Adicione: Captura de Entrada de Áudio → CABLE Output")
        sub("No OBS — Configurações → Áudio:", TEXT)
        code("Dispositivo de monitoramento → CABLE Input")
        sub("Configuração no app — apenas Saída:", GOLD)
        body("• Seu microfone      → External Microphone")
        body("• Traduzir para      → idioma do público")
        body("• Saída cabo virtual → CABLE Input")
        body("(Entrada não se aplica em live solo — não há audio de volta)")

        # ── INSTAGRAM LIVE ────────────────────────────────────────────────────
        sec("📸  Instagram Live (via OBS + RTMP)")
        sub("Mesmo processo do YouTube Live:", TEXT)
        body("Configure o OBS igual ao YouTube acima.")
        body("No OBS use a chave RTMP do Instagram Live Producer.")
        body("O áudio traduzido vai direto para os seguidores.")

        # ── TWITCH ────────────────────────────────────────────────────────────
        sec("🎮  Twitch (via OBS Studio)")
        sub("Configuração idêntica ao YouTube Live.", TEXT)
        body("Use a chave de stream do Twitch no OBS → Configurações → Stream.")

        # ── DICAS ─────────────────────────────────────────────────────────────
        sec("💡  Dicas importantes")
        body("• Sempre use fones de ouvido no modo bidirecional para evitar eco.")
        body("• Inicie o JSL Live Translate ANTES de entrar na reunião/live.")
        body("• O áudio traduzido é gravado automaticamente na pasta 'gravacoes/'.")
        body("• Troque o idioma a qualquer momento — o trecho anterior é salvo.")
        body("• Se der erro, verifique se o GEMINI_API_KEY está correto no .env")

        # botão fechar
        tk.Button(frame, text="Fechar", font=f_body,
                  bg=RED, fg="white", activebackground="#c81d2d",
                  relief="flat", padx=16, pady=6, cursor="hand2",
                  command=win.destroy).pack(pady=(20, 24))

        # centraliza
        win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 620) // 2
        y = self.root.winfo_y() + 40
        win.geometry(f"620x560+{x}+{y}")

    def _on_close(self):
        if self._active_out:
            self._stop_out()
        if self._active_in:
            self._stop_in()
        self.root.destroy()


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
