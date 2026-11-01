from __future__ import annotations

import io
import os

from openai import OpenAI


CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.6-luna")
TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "coral")

SYSTEM_PROMPT = """You are the friendly voice of a social robot speaking face-to-face.
Keep replies natural, warm, and quick: normally one short sentence and never more than two.
The conversation history belongs only to the currently identified person.
Use details they previously shared when relevant, but do not claim certainty about their identity.
If you learn their preferred name, use it naturally. Do not use markdown."""


def validate_api_key(api_key: str) -> None:
    client = OpenAI(api_key=api_key, timeout=15.0)
    # A lightweight authenticated request; the secret is never returned or logged by us.
    client.models.list()


def transcribe(api_key: str, audio: bytes, filename: str = "speech.webm") -> str:
    client = OpenAI(api_key=api_key, timeout=45.0)
    audio_file = io.BytesIO(audio)
    audio_file.name = filename
    result = client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=audio_file)
    return result.text.strip()


def respond(api_key: str, history: list[dict[str, str]], user_text: str) -> str:
    client = OpenAI(api_key=api_key, timeout=45.0)
    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=SYSTEM_PROMPT,
        input=[*history, {"role": "user", "content": user_text}],
        max_output_tokens=100,
        reasoning={"effort": "none"},
        text={"verbosity": "low"},
        store=False,
    )
    return response.output_text.strip()


def synthesize(api_key: str, text: str) -> bytes:
    client = OpenAI(api_key=api_key, timeout=45.0)
    options = {"model": TTS_MODEL, "voice": TTS_VOICE, "input": text, "response_format": "mp3"}
    if not TTS_MODEL.startswith("tts-1"):
        options["instructions"] = "Speak warmly and clearly, like a friendly social robot."
    speech = client.audio.speech.create(**options)
    if hasattr(speech, "read"):
        return speech.read()
    return bytes(speech.content)
