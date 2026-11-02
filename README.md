# 📜 Itihasa: Heritage Storytelling Agent

A museum-grade application that ingests ancient manuscripts, inscriptions, and oral archives, transcribes/translates them with Sarvam AI, and weaves them into interactive historical stories, podcasts, or guided museum audio tours with studio-grade Indian voice narration.

---

## 🏛️ Architecture & Capabilities

```
Oral Archive (Audio) / Manuscript (PDF/Text)
                    │
                    ▼
┌───────────────────────────────────────────────┐
│              INGESTION MODULE                 │
│  - Oral history audio (WAV, MP3, M4A, etc.)   │
│  - Manuscript / archival decrees (PDF, TXT)   │
│  - One-click curated archival presets         │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│        TRANSCRIPTION & TRANSLATION            │
│  - Sarvam Saaras v4 (STT across 23+ langs)    │
│  - Sarvam Mayura / Translate (Auto detection) │
│  - Intelligent Indian-delimiter chunking      │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│            STORYTELLING ENGINE                │
│  - Format 1: 📜 Interactive Historical Story  │
│  - Format 2: 🎙️ Chronicles Podcast (Dual Host)│
│  - Format 3: 🏛️ Guided Museum Audio Tour      │
│  - LLM: Sarvam 105B, GPT-4o, or Gemini 2.0    │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│           MULTILINGUAL VOICE NARRATION        │
│  - Sarvam Bulbul v3 Indian voice synthesis    │
│  - Seamless WAV PCM buffer concatenation      │
│  - 8 curated historical persona voices        │
│  - Audio player & instant .wav download       │
└───────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Configure Keys in `.env`
Ensure your `.env` contains your active Sarvam AI key:
```env
SARVAM_API_KEY=your_sarvam_api_key_here

# Optional: Add if using OpenAI or Gemini as LLM backend
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Run the Test Suite
Verify all components locally:
```bash
python test_heritage_agent.py
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

- [`app.py`](app.py): Museum-aesthetic Streamlit web interface with tabbed archival workflow.
- [`sarvam_utils.py`](sarvam_utils.py): Robust wrappers for Saaras v4 (STT), Mayura (Translation), and Bulbul (TTS with audio buffer stitching).
- [`llm_utils.py`](llm_utils.py): Prompt templates for Stories, Podcasts, and Guided Tours, plus multi-provider LLM dispatchers.
- [`test_heritage_agent.py`](test_heritage_agent.py): Complete test suite covering chunking, stitching, translation, TTS, and LLM generation.
- [`requirements.txt`](requirements.txt): Required Python libraries.
- [`.env`](.env): Local environment file (protected in `.gitignore`).
