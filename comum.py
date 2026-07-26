"""
Configuração compartilhada do JSL Live Translate.
Usada pelo main.py (aplicativo do computador) e pelo server.py (versão web).
Alterou aqui, valeu para os dois.
"""

from google.genai import types

MODEL = "gemini-3.5-live-translate-preview"

INPUT_RATE = 16000

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


def criar_config_live(target_lang: str) -> types.LiveConnectConfig:
    """Configuração da sessão de tradução ao vivo do Gemini."""
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        translation_config=types.TranslationConfig(
            target_language_code=target_lang,
            echo_target_language=False,
        ),
    )
