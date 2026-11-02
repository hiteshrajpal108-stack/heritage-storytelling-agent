"""
Example script demonstrating all Sarvam AI capabilities:
1. Chat Completion (sarvam-105b-conversations)
2. Translation (English to Hindi)
3. Transliteration (Roman script to Devanagari)
4. Language Identification
5. Text-to-Speech (Bulbul:v3)
6. Speech-to-Text (Saaras:v4)
"""

from pathlib import Path
from sarvam_service import SarvamService

def main():
    service = SarvamService()
    print("=" * 60)
    print("Sarvam AI - Multi-Capability Demo")
    print("=" * 60)

    # 1. Chat Completion
    print("\n[1] Chat Completion (sarvam-105b-conversations):")
    reply = service.chat("What are the advantages of using multilingual AI models?")
    print(reply[:200] + "..." if len(reply) > 200 else reply)

    # 2. Translation
    print("\n[2] Translation (English -> Hindi):")
    text_to_translate = "Good morning! Welcome to our developer platform."
    translated = service.translate(text_to_translate, source_language_code="en-IN", target_language_code="hi-IN")
    print(f"Original:   {text_to_translate}")
    print(f"Translated: {translated}")

    # 3. Transliteration
    print("\n[3] Transliteration (Roman -> Devanagari):")
    roman_text = "aap kaise ho"
    transliterated = service.transliterate(roman_text, source_language_code="en-IN", target_language_code="hi-IN")
    print(f"Romanized:      {roman_text}")
    print(f"Transliterated: {transliterated}")

    # 4. Language Identification
    print("\n[4] Language Identification:")
    sample = "നമസ്കാരം, സുഖമാണോ?"
    detected = service.identify_language(sample)
    print(f"Sample text:   {sample}")
    print(f"Language Code: {detected['language_code']}")
    print(f"Script Code:   {detected['script_code']}")

    # 5. Text-to-Speech (TTS)
    print("\n[5] Text-to-Speech (TTS):")
    audio_path = Path(__file__).parent / "demo_output.wav"
    service.text_to_speech(
        text="सर्वम एआई में आपका स्वागत है।",
        language_code="hi-IN",
        speaker="aditya",
        output_file=audio_path,
    )
    print(f"Generated audio saved to: {audio_path.name} ({audio_path.stat().st_size} bytes)")

    # 6. Speech-to-Text (STT)
    print("\n[6] Speech-to-Text (STT - Saaras:v4):")
    stt_result = service.speech_to_text(audio_path, model="saaras:v4")
    print(f"Transcribed Text: {stt_result['transcript']}")
    print(f"Detected Lang:    {stt_result['language_code']}")

    # Clean up generated demo file if desired
    if audio_path.exists():
        audio_path.unlink()
        print("(Cleaned up demo_output.wav)")

    print("\n" + "=" * 60)
    print("All capabilities executed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
