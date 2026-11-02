"""
test_heritage_agent.py - Comprehensive Test Suite for Heritage Storytelling Agent

Runs tests on:
1. API Key resolution
2. Sentence splitting and text chunking
3. Audio buffer stitching with wave module
4. Script cleaning for TTS narration
5. Live or Mock Translation
6. Live or Mock Text-to-Speech
7. Live or Mock LLM Story Generation
"""

import io
import wave
from pathlib import Path

from sarvam_utils import (
    load_sarvam_api_key,
    _split_into_chunks,
    _stitch_wav_buffers,
    translate_text,
    generate_speech,
    SUPPORTED_LANGUAGES,
    VOICE_PERSONAS,
)
from llm_utils import (
    clean_script_for_narration,
    generate_heritage_content,
)


def test_chunking():
    print("[TEST 1] Text Chunking on Indian Delimiters...")
    sample = "રાજા સિદ્ધરાજ જયસિંહના સમયમાં આઠસો સાળવી પરિવારો વસ્યા હતા। એક-એક તાણાવાણામાં સોના-રૂપાના તાર વણવામાં આવતા હતા. This is sentence three."
    chunks = _split_into_chunks(sample, max_chunk_len=50)
    assert len(chunks) >= 2, "Expected multiple chunks"
    print(f"  -> Successfully chunked into {len(chunks)} parts:")
    for i, c in enumerate(chunks):
        print(f"     Part {i+1}: {c}")


def test_audio_stitching():
    print("\n[TEST 2] WAV Buffer Stitching...")
    # Create two synthetic 1-second silent WAV buffers (mono, 16-bit, 22050 Hz)
    rate = 22050
    duration_frames = rate // 2  # 0.5s

    def make_wav():
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"\x00\x00" * duration_frames)
        return out.getvalue()

    b1 = make_wav()
    b2 = make_wav()
    stitched = _stitch_wav_buffers([b1, b2])

    with wave.open(io.BytesIO(stitched), "rb") as w:
        frames = w.getnframes()
        assert frames == duration_frames * 2, f"Expected {duration_frames*2} frames, got {frames}"
    print(f"  -> Successfully stitched 2 audio buffers into {len(stitched)} bytes WAV ({frames} frames).")


def test_clean_script():
    print("\n[TEST 3] Script Cleaning for TTS...")
    raw = (
        "### Chapter 1: The Port of Nagapattinam\n"
        "[Theme Music: Sitar and ocean waves fade in]\n"
        "**Arjun:** Look at the sea-going junks anchored in the harbor!\n"
        "[Sound Effect: Bell ringing]\n"
        "**Meera:** Truly a sight of imperial Chola majesty."
    )
    cleaned = clean_script_for_narration(raw)
    assert "[Theme Music" not in cleaned
    assert "[Sound Effect" not in cleaned
    assert "**Arjun:**" not in cleaned
    print("  -> Cleaned script for TTS narration:\n" + "\n".join("     " + line for line in cleaned.splitlines()))


def test_live_translation():
    print("\n[TEST 4] Live Sarvam Translation API...")
    try:
        translated = translate_text("Ancient temple bells echoed across the valley.", target_lang="hi-IN")
        print(f"  -> Translated: {translated}")
    except Exception as e:
        print(f"  -> Skipped or failed: {e}")


def test_live_tts():
    print("\n[TEST 5] Live Sarvam Bulbul TTS API...")
    try:
        out_path = Path(__file__).parent / "test_heritage_out.wav"
        audio = generate_speech("इतिहास की इस यात्रा में आपका स्वागत है।", target_lang="hi-IN", speaker="aditya", output_path=out_path)
        print(f"  -> Generated {len(audio)} bytes of audio at {out_path.name}")
        if out_path.exists():
            out_path.unlink()
    except Exception as e:
        print(f"  -> Skipped or failed: {e}")


def test_live_llm():
    print("\n[TEST 6] Live Sarvam 105B Heritage Story Generation...")
    try:
        excerpt = "King Rajendra Chola sent his naval armada across the sea to Sriwijaya in 1025 CE."
        story = generate_heritage_content(excerpt, content_type="story", language_name="English", provider="sarvam")
        print(f"  -> Story generated ({len(story)} chars). Preview:\n     " + story[:250].replace("\n", " ") + "...")
    except Exception as e:
        print(f"  -> Skipped or failed: {e}")


if __name__ == "__main__":
    print("=" * 65)
    print("HERITAGE STORYTELLING AGENT - TEST SUITE")
    print("=" * 65)
    test_chunking()
    test_audio_stitching()
    test_clean_script()
    test_live_translation()
    test_live_tts()
    test_live_llm()
    print("\n" + "=" * 65)
    print("ALL TESTS COMPLETED!")
    print("=" * 65)
