"""
Sarvam AI Service Wrapper
Provides ready-to-use helpers for all Sarvam AI capabilities:
- Chat & LLMs (sarvam-105b, sarvam-105b-conversations)
- Text-to-Speech (TTS - Bulbul models)
- Speech-to-Text (STT - Saaras models)
- Text Translation (Mayura / Sarvam Translate)
- Transliteration
- Language Identification
- Document Intelligence (Doc AI)
"""

import os
import base64
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from sarvamai import SarvamAI

def _load_env():
    """Load environment variables from .env if present."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))

class SarvamService:
    def __init__(self, api_key: Optional[str] = None):
        _load_env()
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY is not set. Please set it in .env or pass api_key to SarvamService."
            )
        self.client = SarvamAI(api_subscription_key=self.api_key)

    # -------------------------------------------------------------------------
    # 1. Chat Completion / LLM
    # -------------------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "sarvam-105b-conversations",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Chat completion using Sarvam models.
        Models:
            - 'sarvam-105b-conversations' (dialogue & conversational agent)
            - 'sarvam-105b' (general reasoning & coding)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions(**kwargs)
        return response.choices[0].message.content

    # -------------------------------------------------------------------------
    # 2. Text-to-Speech (TTS)
    # -------------------------------------------------------------------------
    def text_to_speech(
        self,
        text: str,
        language_code: str = "hi-IN",
        speaker: str = "aditya",
        model: str = "bulbul:v3",
        output_file: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """
        Synthesize speech from text.
        Languages: 'hi-IN', 'en-IN', 'bn-IN', 'gu-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'od-IN', 'pa-IN', 'ta-IN', 'te-IN'
        Speakers (bulbul:v3): 'aditya', 'ritu', 'priya', 'neha', 'rahul', 'pooja', 'rohan', 'simran', 'kavya', etc.
        Returns raw audio bytes (WAV) and optionally writes to output_file.
        """
        response = self.client.text_to_speech.convert(
            text=text,
            language_code=language_code,
            speaker=speaker,
            model=model,
        )
        audio_bytes = base64.b64decode(response.audios[0])

        if output_file:
            Path(output_file).write_bytes(audio_bytes)

        return audio_bytes

    # -------------------------------------------------------------------------
    # 3. Speech-to-Text (STT)
    # -------------------------------------------------------------------------
    def speech_to_text(
        self,
        audio_file_path: Union[str, Path],
        model: str = "saaras:v4",
        language_code: str = "unknown",
        with_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        Supports 23+ Indian languages and code-mixing.
        """
        path = Path(audio_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with open(path, "rb") as f:
            response = self.client.speech_to_text.transcribe(
                file=f,
                model=model,
                language_code=language_code,
                with_timestamps=with_timestamps,
            )

        return {
            "transcript": response.transcript,
            "language_code": response.language_code,
            "language_probability": response.language_probability,
            "timestamps": response.timestamps,
        }

    # -------------------------------------------------------------------------
    # 4. Text Translation
    # -------------------------------------------------------------------------
    def translate(
        self,
        text: str,
        source_language_code: str = "auto",
        target_language_code: str = "hi-IN",
        mode: str = "formal",
    ) -> str:
        """
        Translate text across English and Indian languages.
        Language codes: 'en-IN', 'hi-IN', 'bn-IN', 'gu-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'ta-IN', 'te-IN', etc.
        Modes: 'formal', 'modern-colloquial', 'classic-colloquial', 'code-mixed'
        """
        response = self.client.text.translate(
            input=text,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
            mode=mode,
        )
        return response.translated_text

    # -------------------------------------------------------------------------
    # 5. Script Transliteration
    # -------------------------------------------------------------------------
    def transliterate(
        self,
        text: str,
        source_language_code: str = "en-IN",
        target_language_code: str = "hi-IN",
    ) -> str:
        """
        Transliterate text between Romanized / native Indic scripts.
        E.g., 'namaste' -> 'नमस्ते'
        """
        response = self.client.text.transliterate(
            input=text,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )
        return response.transliterated_text

    # -------------------------------------------------------------------------
    # 6. Language Identification
    # -------------------------------------------------------------------------
    def identify_language(self, text: str) -> Dict[str, Optional[str]]:
        """
        Identify language and script code of input text.
        """
        response = self.client.text.identify_language(input=text)
        return {
            "language_code": response.language_code,
            "script_code": response.script_code,
        }

    # -------------------------------------------------------------------------
    # 7. Document Intelligence (Doc AI)
    # -------------------------------------------------------------------------
    def parse_document(
        self,
        file_path: Union[str, Path],
        content_type: str = "printed",
    ) -> Any:
        """
        Submit a document (PDF, image) for OCR / parsing job.
        Returns job initiation response containing job_id.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        with open(path, "rb") as f:
            response = self.client.doc_ai.parse(
                file=[f],
                content_type=content_type,
            )
        return response
