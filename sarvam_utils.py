"""
sarvam_utils.py - Production-ready Sarvam AI API wrappers for Heritage Storytelling Agent.

Supported Capabilities:
1. Speech-to-Text (STT) using Saaras v4 (`saaras:v4`)
2. Text Translation using Mayura/Sarvam-Translate (`mayura:v1` / `sarvam-translate:v1`)
3. Text-to-Speech (TTS) using Bulbul (`bulbul:v3`) with long-text chunking & stitching
"""

import os
import io
import re
import wave
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from sarvamai import SarvamAI
from sarvamai.errors import (
    BadRequestError,
    UnauthorizedError,
    TooManyRequestsError,
    InternalServerError,
)
from sarvamai.core.api_error import ApiError

logger = logging.getLogger(__name__)

# Supported Indian languages and BCP-47 codes for Heritage Storytelling
SUPPORTED_LANGUAGES = {
    "hi-IN": "Hindi (हिन्दी)",
    "en-IN": "English (Indian)",
    "bn-IN": "Bengali (বাংলা)",
    "gu-IN": "Gujarati (ગુજરાતી)",
    "kn-IN": "Kannada (ಕನ್ನಡ)",
    "ml-IN": "Malayalam (മലയാളം)",
    "mr-IN": "Marathi (मराठी)",
    "od-IN": "Odia (ଓଡ଼ିଆ)",
    "pa-IN": "Punjabi (ਪੰਜਾਬੀ)",
    "ta-IN": "Tamil (தமிழ்)",
    "te-IN": "Telugu (తెలుగు)",
    "as-IN": "Assamese (অসমীয়া)",
    "ur-IN": "Urdu (اردو)",
    "sa-IN": "Sanskrit (संस्कृतम्)",
}

# Curated recommended voice personas for historical narration (bulbul:v3)
VOICE_PERSONAS = {
    "aditya": "Aditya (Deep, authoritative male - great for epics & chronicles)",
    "ritu": "Ritu (Warm, articulate female - ideal for folklore & cultural tours)",
    "shubh": "Shubh (Expressive male - excellent for storytelling & podcasts)",
    "priya": "Priya (Gentle female - great for contemplative & historical archives)",
    "rohan": "Rohan (Engaging male narrator - great for guided museum tours)",
    "simran": "Simran (Clear, modern female voice - perfect for educational podcasts)",
    "kabir": "Kabir (Resonant, storytelling male voice)",
    "kavya": "Kavya (Poetic, expressive female voice)",
}


class SarvamAPIException(Exception):
    """Custom user-friendly exception for Sarvam AI errors."""
    pass


def load_sarvam_api_key(explicit_key: Optional[str] = None) -> str:
    """
    Resolve Sarvam API key in order of priority:
    1. Explicit key argument (from UI input)
    2. Streamlit Cloud Secrets (st.secrets["SARVAM_API_KEY"])
    3. os.environ['SARVAM_API_KEY']
    4. .env file in project directory
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # Check Streamlit Secrets (used in Streamlit Community Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "SARVAM_API_KEY" in st.secrets:
            secret_val = st.secrets["SARVAM_API_KEY"]
            if secret_val and str(secret_val).strip():
                return str(secret_val).strip()
    except Exception:
        pass

    env_key = os.environ.get("SARVAM_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # Search for .env in current and parent directories
    for search_dir in [Path.cwd(), Path(__file__).resolve().parent]:
        env_file = search_dir / ".env"
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "SARVAM_API_KEY":
                            val = v.strip().strip("'\"")
                            if val:
                                os.environ["SARVAM_API_KEY"] = val
                                return val

    raise SarvamAPIException(
        "Sarvam API Key is missing. Please enter your key in the sidebar or add SARVAM_API_KEY to your .env file."
    )


def get_sarvam_client(api_key: Optional[str] = None) -> SarvamAI:
    """Instantiate a SarvamAI client with proper validation."""
    key = load_sarvam_api_key(api_key)
    try:
        return SarvamAI(api_subscription_key=key)
    except Exception as e:
        raise SarvamAPIException(f"Failed to initialize Sarvam AI client: {str(e)}")


# =============================================================================
# 1. Speech-to-Text (STT)
# =============================================================================
def transcribe_audio(
    file_path: Union[str, Path],
    model: str = "saaras:v4",
    language_code: str = "unknown",
    with_timestamps: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribes an oral archive audio file using Sarvam's Saaras v4 model.

    Args:
        file_path: Path to the audio file (.wav, .mp3, .m4a, .aac, .flac, .ogg).
        model: STT model ('saaras:v4' or 'saaras:v3').
        language_code: Specific BCP-47 language code or 'unknown' for automatic detection.
        with_timestamps: Whether to request phrase-level timestamps.
        api_key: Optional explicit API key override.

    Returns:
        Dict containing:
            - 'transcript': Full transcribed text.
            - 'language_code': Detected or confirmed language code.
            - 'language_probability': Confidence score (0.0 to 1.0).
            - 'timestamps': List/dict of phrase timestamps if requested.
    """
    path = Path(file_path)
    if not path.exists():
        raise SarvamAPIException(f"Audio archive not found: {path.name}")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 25.0:
        raise SarvamAPIException(
            f"Audio file is too large ({file_size_mb:.1f} MB). Maximum supported file size is 25 MB. Please trim or compress the audio."
        )

    if path.stat().st_size == 0:
        raise SarvamAPIException(f"Audio file is empty: {path.name}")

    client = get_sarvam_client(api_key)

    try:
        with open(path, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model=model,
                language_code=language_code,
                with_timestamps=with_timestamps,
            )

        return {
            "transcript": response.transcript.strip(),
            "language_code": response.language_code or "unknown",
            "language_probability": response.language_probability,
            "timestamps": response.timestamps,
        }

    except UnauthorizedError:
        raise SarvamAPIException("Authentication failed: Invalid Sarvam API Key. Please verify your credentials.")
    except TooManyRequestsError:
        raise SarvamAPIException("Rate limit reached on Sarvam STT API. Please wait a moment before trying again.")
    except BadRequestError as e:
        raise SarvamAPIException(f"Invalid audio format or parameters: {str(e)}")
    except ApiError as e:
        raise SarvamAPIException(f"Sarvam STT API error: {str(e)}")
    except Exception as e:
        raise SarvamAPIException(f"Unexpected error during transcription: {str(e)}")


# =============================================================================
# 2. Text Translation
# =============================================================================
def _split_into_chunks(text: str, max_chunk_len: int = 800) -> List[str]:
    """
    Splits text on sentence delimiters into chunks not exceeding max_chunk_len.
    Preserves Indian punctuation (such as Purna Viram '।').
    """
    text = text.strip()
    if len(text) <= max_chunk_len:
        return [text]

    # Split on periods, exclamation marks, question marks, purna viram, or double newlines
    sentences = re.split(r"([।\.!\?\n\r]+)", text)
    chunks = []
    current_chunk = []
    current_len = 0

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_sentence = (sentence + delimiter).strip()
        if not full_sentence:
            continue

        if current_len + len(full_sentence) > max_chunk_len and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [full_sentence]
            current_len = len(full_sentence)
        else:
            current_chunk.append(full_sentence)
            current_len += len(full_sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def translate_text(
    text: str,
    target_lang: str = "en-IN",
    source_lang: str = "auto",
    mode: str = "formal",
    model: str = "mayura:v1",
    api_key: Optional[str] = None,
) -> str:
    """
    Translates manuscript or archive text between Indian languages and English.
    Automatically handles long-text chunking to stay within model limits.

    Args:
        text: Input text to translate.
        target_lang: Destination BCP-47 language code (e.g. 'en-IN', 'hi-IN').
        source_lang: Source language code or 'auto' for automatic detection.
        mode: Translation style: 'formal', 'modern-colloquial', 'classic-colloquial', 'code-mixed'.
        model: 'mayura:v1' (supports auto-detection) or 'sarvam-translate:v1'.
        api_key: Optional explicit API key override.

    Returns:
        Translated text string.
    """
    if not text or not text.strip():
        return ""

    # If source and target are the same, return as is
    if source_lang != "auto" and source_lang == target_lang:
        return text.strip()

    client = get_sarvam_client(api_key)
    max_len = 800 if model == "mayura:v1" else 1500
    chunks = _split_into_chunks(text, max_chunk_len=max_len)
    translated_parts = []

    try:
        for chunk in chunks:
            if not chunk.strip():
                continue
            response = client.text.translate(
                input=chunk,
                source_language_code=source_lang,
                target_language_code=target_lang,
                mode=mode,
                model=model,
            )
            translated_parts.append(response.translated_text.strip())

        return " ".join(translated_parts)

    except UnauthorizedError:
        raise SarvamAPIException("Authentication failed: Invalid Sarvam API Key.")
    except TooManyRequestsError:
        raise SarvamAPIException("Rate limit reached on Sarvam Translation API. Please pause and retry.")
    except BadRequestError as e:
        raise SarvamAPIException(f"Translation request error: {str(e)}")
    except ApiError as e:
        raise SarvamAPIException(f"Sarvam Translation API error: {str(e)}")
    except Exception as e:
        raise SarvamAPIException(f"Unexpected error during translation: {str(e)}")


# =============================================================================
# 3. Text-to-Speech (TTS) Narration
# =============================================================================
def _stitch_wav_buffers(audio_buffers: List[bytes]) -> bytes:
    """
    Concatenates multiple WAV audio byte segments into a single cohesive WAV file
    using Python's built-in wave module without external dependencies.
    """
    if not audio_buffers:
        return b""
    if len(audio_buffers) == 1:
        return audio_buffers[0]

    out_io = io.BytesIO()
    out_wav = None

    for buf in audio_buffers:
        in_io = io.BytesIO(buf)
        with wave.open(in_io, "rb") as in_wav:
            if out_wav is None:
                out_wav = wave.open(out_io, "wb")
                out_wav.setparams(in_wav.getparams())
            out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))

    if out_wav:
        out_wav.close()

    return out_io.getvalue()


def generate_speech(
    text: str,
    target_lang: str = "hi-IN",
    speaker: str = "aditya",
    model: str = "bulbul:v3",
    pace: float = 1.0,
    temperature: float = 0.6,
    output_path: Optional[Union[str, Path]] = None,
    api_key: Optional[str] = None,
) -> bytes:
    """
    Converts historical stories, podcast episodes, or museum tour scripts into voice narration.
    Handles long scripts by chunking text (max 2000 chars per API call) and stitching WAV segments.

    Args:
        text: Script text to be voiced.
        target_lang: Language code ('hi-IN', 'en-IN', 'bn-IN', 'ta-IN', etc.).
        speaker: Voice persona name ('aditya', 'ritu', 'shubh', 'priya', etc.).
        model: TTS model (default 'bulbul:v3').
        pace: Narration speed (0.5 to 2.0, default 1.0).
        temperature: Expressiveness (0.01 to 1.0, default 0.6).
        output_path: Optional file path to write the output WAV.
        api_key: Optional explicit API key override.

    Returns:
        Raw audio bytes (WAV format).
    """
    if not text or not text.strip():
        raise SarvamAPIException("Cannot generate speech from empty text.")

    client = get_sarvam_client(api_key)
    # Bulbul v3 supports up to 2500 chars, chunk at ~1800 for safety and natural phrasing
    chunks = _split_into_chunks(text, max_chunk_len=1800)
    audio_segments = []

    try:
        for chunk in chunks:
            if not chunk.strip():
                continue
            response = client.text_to_speech.convert(
                text=chunk,
                language_code=target_lang,
                speaker=speaker,
                model=model,
                pace=pace,
                temperature=temperature,
            )
            raw_chunk = base64.b64decode(response.audios[0])
            audio_segments.append(raw_chunk)

        combined_audio = _stitch_wav_buffers(audio_segments)

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(combined_audio)

        return combined_audio

    except UnauthorizedError:
        raise SarvamAPIException("Authentication failed: Invalid Sarvam API Key.")
    except TooManyRequestsError:
        raise SarvamAPIException("Rate limit reached on Sarvam TTS API. Please pause and retry.")
    except BadRequestError as e:
        raise SarvamAPIException(f"TTS conversion error: {str(e)}")
    except ApiError as e:
        raise SarvamAPIException(f"Sarvam TTS API error: {str(e)}")
    except Exception as e:
        raise SarvamAPIException(f"Unexpected error generating voice narration: {str(e)}")
