"""
app.py - Heritage Storytelling Agent
A museum-grade web application transforming ancient manuscripts and oral archives
into immersive historical narratives, podcasts, and guided tours with multilingual Indian voice narration.

Powered by Sarvam AI (Saaras v4, Mayura, Bulbul v3) and Multilingual LLMs.
"""

import os
import io
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from pypdf import PdfReader

# Import our custom modules
from sarvam_utils import (
    transcribe_audio,
    translate_text,
    generate_speech,
    SUPPORTED_LANGUAGES,
    VOICE_PERSONAS,
    SarvamAPIException,
    load_sarvam_api_key,
)
from llm_utils import (
    generate_heritage_content,
    clean_script_for_narration,
)

# =============================================================================
# Page Configuration & Museum Aesthetic CSS
# =============================================================================

st.set_page_config(
    page_title="Itihasa | Heritage Storytelling Agent",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

MUSEUM_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

/* Main Background & Parchment Texture */
.stApp {
    background-color: #FAF6EE;
    color: #2B2118;
    font-family: 'Playfair Display', Georgia, serif;
}

/* Headings */
h1, h2, h3, .heritage-title {
    font-family: 'Cinzel', serif !important;
    color: #4A1E14 !important;
    letter-spacing: 0.05em;
    font-weight: 700;
}

/* Hero Banner */
.heritage-hero {
    background: linear-gradient(135deg, #3E160D 0%, #5E2618 50%, #8C3A27 100%);
    border: 2px solid #D4AF37;
    border-radius: 12px;
    padding: 30px 40px;
    margin-bottom: 30px;
    box-shadow: 0 10px 25px rgba(62, 22, 13, 0.15);
    color: #FAF4E6;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.heritage-hero h1 {
    color: #F7E7CE !important;
    font-size: 2.6rem;
    margin-bottom: 8px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.4);
}

.heritage-hero p {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #E6D2B5;
    font-size: 1.1rem;
    margin: 0 auto;
    max-width: 800px;
}

/* Parchment Card */
.parchment-card {
    background: #FFFDF9;
    border: 1px solid #E2D3B8;
    border-left: 5px solid #8C3A27;
    border-radius: 8px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(110, 80, 50, 0.06);
    color: #2C2016;
}

.parchment-card h4 {
    font-family: 'Cinzel', serif;
    color: #6B271A;
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 1.15rem;
}

/* Script Display Box */
.script-container {
    background-color: #FCF9F2;
    border: 1px solid #D9C8AA;
    border-radius: 8px;
    padding: 24px 28px;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #1F1610;
    white-space: pre-wrap;
    max-height: 480px;
    overflow-y: auto;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.03);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #F3ECE0 !important;
    border-right: 1px solid #D9C8AA;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Cinzel', serif;
    color: #5E2618 !important;
    border-bottom: 2px solid #D4AF37;
    padding-bottom: 6px;
    margin-top: 15px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7A2818 0%, #9E3720 100%) !important;
    color: #FFFDF9 !important;
    font-family: 'Cinzel', serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border: 1px solid #D4AF37 !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 10px rgba(122, 40, 24, 0.2) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #9E3720 0%, #BC4529 100%) !important;
    box-shadow: 0 6px 14px rgba(122, 40, 24, 0.35) !important;
    transform: translateY(-1px);
}

/* Badges / Tags */
.heritage-badge {
    display: inline-block;
    background: #EEDEC4;
    color: #5E2618;
    border: 1px solid #CDB591;
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}
</style>
"""

st.markdown(MUSEUM_THEME_CSS, unsafe_allow_html=True)

# =============================================================================
# Session State Initialization
# =============================================================================

if "source_text" not in st.session_state:
    st.session_state.source_text = ""
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "generated_script" not in st.session_state:
    st.session_state.generated_script = ""
if "generated_audio" not in st.session_state:
    st.session_state.generated_audio = None
if "selected_content_type" not in st.session_state:
    st.session_state.selected_content_type = "story"

# Sample historical archives for one-click testing
SAMPLE_ARCHIVES = {
    "Select a Sample Archival Piece...": "",
    "Mughal Royal Decrees (Akbar's Court Record)": (
        "In the thirty-fourth year of the Divine Era, His Imperial Majesty issued a firman "
        "concerning the sacred banyan grove near Fatehpur Sikri. It was recorded by the royal scribe "
        "Abul Fazl that no artisan shall fell the ancient cedar and neem trees where travelers congregate. "
        "The court treasury disbursed six hundred silver tankas for the construction of an adjoining sarai "
        "and freshwater cisterns to serve pilgrims traversing from Agra to Mewar."
    ),
    "Oral Chronicle: The Lost Weavers of Patan (Gujarati)": (
        "અમારા વડવાઓ કહેતા કે પાટણના પટોળાનું વણાટકામ એ ફક્ત કાપડ નથી પણ દેવી-દેવતાઓની આરાધના છે. "
        "એક-એક તાણાવાણામાં સોના-રૂપાના તાર અને રેશમનો એવો મેળ બેસાડવામાં આવતો કે સાડી સો વર્ષ સુધી ફાટે પણ રંગ ન જાય. "
        "રાજા સિદ્ધરાજ જયસિંહના સમયમાં આઠસો સાળવી પરિવારો દક્ષિણમાંથી અહીં પાટણ વસ્યા હતા."
    ),
    "Chola Maritime Inscription (Nagapattinam Port)": (
        "This copper-plate decree of Rajendra Chola I, inscribed in ancient grantha script, commemorates "
        "the fleet's departure across the Bay of Bengal towards Sriwijaya and Kedah. The harbor-master at "
        "Nagapattinam records the lading of sandalwood, black pepper, woven silk, and bronze bells onto fifteen "
        "multi-masted junks under the auspicious constellations of Karthigai."
    ),
}

# =============================================================================
# Sidebar: Settings & Configuration
# =============================================================================

with st.sidebar:
    st.markdown("### 🏛️ Archival Controls")

    # API Key Configuration
    try:
        loaded_key = load_sarvam_api_key()
        key_status = "✅ Sarvam Key Active"
        key_color = "green"
    except Exception:
        loaded_key = ""
        key_status = "⚠️ Sarvam Key Not Found"
        key_color = "red"

    st.markdown(
        f"<span style='color:{key_color}; font-weight:600; font-size:0.9rem;'>{key_status}</span>",
        unsafe_allow_html=True,
    )

    with st.expander("🔑 API Key Settings", expanded=(not bool(loaded_key))):
        sarvam_key_input = st.text_input(
            "Sarvam API Key",
            value=loaded_key,
            type="password",
            help="Configured from environment or .env by default.",
        )
        llm_provider = st.selectbox(
            "Storyteller LLM Engine",
            options=["sarvam", "openai", "gemini"],
            format_func=lambda x: {
                "sarvam": "Sarvam 105B (Indic Native)",
                "openai": "OpenAI (GPT-4o)",
                "gemini": "Google Gemini (2.0 Flash)",
            }[x],
            help="Select the model family used to weave historical stories.",
        )
        llm_key_input = ""
        if llm_provider == "openai":
            llm_key_input = st.text_input("OpenAI API Key", type="password")
        elif llm_provider == "gemini":
            llm_key_input = st.text_input("Gemini API Key", type="password")

    st.markdown("---")
    st.markdown("### 🗣️ Voice & Language")

    # Target Language for Story & TTS
    target_lang_code = st.selectbox(
        "Narration Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=0,  # 'hi-IN' by default
        format_func=lambda x: SUPPORTED_LANGUAGES[x],
    )

    # Voice Persona
    voice_speaker = st.selectbox(
        "Narrator Voice Persona",
        options=list(VOICE_PERSONAS.keys()),
        index=0,  # 'aditya'
        format_func=lambda x: VOICE_PERSONAS[x],
    )

    narration_pace = st.slider("Narration Pace", min_value=0.7, max_value=1.4, value=1.0, step=0.05)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem; color:#7A6B5D;'>"
        "<b>Itihasa Agent v1.0</b><br>"
        "Preserving oral archives & ancient scriptures with multilingual AI."
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Hero Header Banner
# =============================================================================

st.markdown(
    """
    <div class="heritage-hero">
        <h1>📜 ITIHASA : HERITAGE STORYTELLING AGENT</h1>
        <p>
            Breathe life into ancient manuscripts, stone inscriptions, and oral archives.
            Transcribe, translate, and transform historical memory into evocative chronicles, podcasts, and museum audio guides.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Main Tabbed Workflow
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "1. Ingest & Transcribe",
    "2. Generate Heritage Script",
    "3. Voice Narration & Audio",
])

# -----------------------------------------------------------------------------
# TAB 1: Ingest & Transcribe
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 🏺 Step 1: Ingest Historical Archives")
    st.markdown("Upload oral history recordings, ancient manuscripts (PDF / Text), or select a curated sample.")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown(
            """
            <div class="parchment-card">
                <h4>🎙️ Oral Archive Ingestion</h4>
                <p style="font-size:0.9rem; color:#6B5848;">
                    Upload an oral history recording, interview, or spoken folklore (.wav, .mp3, .m4a, .aac, .flac).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_audio = st.file_uploader(
            "Upload Spoken Archive (Audio)",
            type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
            label_visibility="collapsed",
        )

        if uploaded_audio:
            st.audio(uploaded_audio)
            if st.button("Transcribe Audio with Saaras v4"):
                with st.spinner("Transcribing oral archive with Sarvam Saaras v4..."):
                    try:
                        # Save uploaded file temporarily for the SDK
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_audio.name).suffix) as tmp:
                            tmp.write(uploaded_audio.getbuffer())
                            tmp_path = tmp.name

                        result = transcribe_audio(tmp_path, model="saaras:v4", api_key=sarvam_key_input)
                        os.unlink(tmp_path)

                        st.session_state.source_text = result["transcript"]
                        st.session_state.detected_lang = result["language_code"]
                        st.success(f"Transcription complete! Detected Language: {result['language_code']}")

                    except SarvamAPIException as e:
                        st.error(f"Transcription Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")

    with col2:
        st.markdown(
            """
            <div class="parchment-card">
                <h4>📜 Manuscript Ingestion</h4>
                <p style="font-size:0.9rem; color:#6B5848;">
                    Upload an archival manuscript, transcription notes, or historical decree (.pdf, .txt).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_doc = st.file_uploader(
            "Upload Document (PDF or TXT)",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
        )

        if uploaded_doc:
            if uploaded_doc.name.endswith(".pdf"):
                try:
                    pdf_reader = PdfReader(uploaded_doc)
                    extracted_text = ""
                    for page in pdf_reader.pages:
                        extracted_text += page.extract_text() or ""
                    st.session_state.source_text = extracted_text.strip()
                    st.success(f"Extracted {len(pdf_reader.pages)} page(s) from PDF.")
                except Exception as e:
                    st.error(f"Failed to extract PDF text: {str(e)}")
            else:
                st.session_state.source_text = uploaded_doc.read().decode("utf-8", errors="ignore")
                st.success("Manuscript text loaded successfully.")

        # Preset Sample Selection
        selected_sample = st.selectbox(
            "Or choose a curated historical archive:",
            options=list(SAMPLE_ARCHIVES.keys()),
        )
        if selected_sample and SAMPLE_ARCHIVES[selected_sample]:
            if st.button("Load Selected Sample"):
                st.session_state.source_text = SAMPLE_ARCHIVES[selected_sample]
                st.success(f"Loaded: {selected_sample}")

    # Display Ingested Text & Translation Option
    if st.session_state.source_text:
        st.markdown("---")
        st.markdown("#### 📖 Ingested Historical Record")

        col_text, col_action = st.columns([3, 1], gap="medium")

        with col_text:
            st.session_state.source_text = st.text_area(
                "Source Archival Text",
                value=st.session_state.source_text,
                height=180,
            )

        with col_action:
            st.markdown(
                f"<div class='heritage-badge'>Characters: {len(st.session_state.source_text)}</div>",
                unsafe_allow_html=True,
            )
            if st.session_state.detected_lang:
                st.markdown(
                    f"<div class='heritage-badge'>Lang: {st.session_state.detected_lang}</div>",
                    unsafe_allow_html=True,
                )

            st.write("")
            translate_btn = st.button("Translate Text")
            if translate_btn:
                with st.spinner(f"Translating archival text to {SUPPORTED_LANGUAGES[target_lang_code]}..."):
                    try:
                        translated = translate_text(
                            st.session_state.source_text,
                            target_lang=target_lang_code,
                            source_lang="auto",
                            api_key=sarvam_key_input,
                        )
                        st.session_state.translated_text = translated
                        st.success("Translation complete!")
                    except SarvamAPIException as e:
                        st.error(f"Translation Error: {str(e)}")

        if st.session_state.translated_text:
            st.markdown("#### 🌐 Translated Archival Text")
            st.markdown(
                f"<div class='parchment-card'>{st.session_state.translated_text}</div>",
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# TAB 2: Generate Heritage Script
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### ✍️ Step 2: Transform into Living Heritage")
    st.markdown("Select a creative format to transform historical material into engaging narratives.")

    # Story Type Selection Cards
    format_col1, format_col2, format_col3 = st.columns(3)

    with format_col1:
        st.markdown(
            """
            <div class="parchment-card" style="min-height: 180px;">
                <h4>📜 Interactive Story</h4>
                <p style="font-size:0.88rem; color:#6B5848;">
                    Evocative narrative with historical world-building, sensory details of ancient courts, and emotional character arcs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with format_col2:
        st.markdown(
            """
            <div class="parchment-card" style="min-height: 180px;">
                <h4>🎙️ Archival Podcast</h4>
                <p style="font-size:0.88rem; color:#6B5848;">
                    Conversational dialogue between hosts Arjun (historian) and Meera (storyteller), complete with ambient audio cues.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with format_col3:
        st.markdown(
            """
            <div class="parchment-card" style="min-height: 180px;">
                <h4>🏛️ Guided Museum Tour</h4>
                <p style="font-size:0.88rem; color:#6B5848;">
                    Immersive second-person audio docent guide directing visitors through exhibition galleries and artifact highlights.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    content_type = st.radio(
        "Choose Narrative Format:",
        options=["story", "podcast", "tour"],
        format_func=lambda x: {
            "story": "📜 Interactive Story (Narrative)",
            "podcast": "🎙️ Archival Podcast (Dual Hosts)",
            "tour": "🏛️ Guided Museum Tour (Docent)",
        }[x],
        horizontal=True,
    )
    st.session_state.selected_content_type = content_type

    # Choose Source Input for LLM
    text_source_option = "Original Source Text"
    if st.session_state.translated_text:
        text_source_option = st.radio(
            "Use context from:",
            options=["Translated Text", "Original Source Text"],
            horizontal=True,
        )

    active_context = (
        st.session_state.translated_text
        if text_source_option == "Translated Text" and st.session_state.translated_text
        else st.session_state.source_text
    )

    if not active_context:
        st.info("Please upload or load an archival document in Tab 1 first.")
    else:
        if st.button("Generate Heritage Script"):
            with st.spinner(f"Crafting {content_type} using {llm_provider.upper()}..."):
                try:
                    lang_name = SUPPORTED_LANGUAGES.get(target_lang_code, "English").split(" ")[0]
                    active_key = (
                        sarvam_key_input if llm_provider == "sarvam" else llm_key_input
                    )
                    generated = generate_heritage_content(
                        source_text=active_context,
                        content_type=content_type,
                        language_name=lang_name,
                        provider=llm_provider,
                        api_key=active_key,
                    )
                    st.session_state.generated_script = generated
                    st.success("Script generated successfully!")

                except Exception as e:
                    st.error(f"Script Generation Error: {str(e)}")

    if st.session_state.generated_script:
        st.markdown("---")
        st.markdown("#### 📜 Generated Script")
        st.session_state.generated_script = st.text_area(
            "Edit Script (Adjust wording before voice narration):",
            value=st.session_state.generated_script,
            height=320,
        )


# -----------------------------------------------------------------------------
# TAB 3: Voice Narration & Audio
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🎙️ Step 3: Multilingual Voice Narration")
    st.markdown("Convert your generated heritage story into studio-quality voice narration powered by Sarvam Bulbul.")

    if not st.session_state.generated_script:
        st.info("Generate a heritage script in Tab 2 before creating audio narration.")
    else:
        st.markdown(
            f"""
            <div class="parchment-card">
                <h4>🎙️ Narration Parameters</h4>
                <span class="heritage-badge">Language: {SUPPORTED_LANGUAGES[target_lang_code]}</span>
                <span class="heritage-badge">Voice: {VOICE_PERSONAS[voice_speaker]}</span>
                <span class="heritage-badge">Speed: {narration_pace}x</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        clean_audio_cues = st.checkbox(
            "Clean audio cues (strip bracketed cues like [Sound Effect: ...] and speaker names for seamless narration)",
            value=True,
        )

        script_for_tts = (
            clean_script_for_narration(st.session_state.generated_script)
            if clean_audio_cues
            else st.session_state.generated_script
        )

        col_synth, col_stats = st.columns([1, 1])
        with col_stats:
            st.caption(f"Script Length: {len(script_for_tts)} characters | Estimated chunks: {max(1, len(script_for_tts)//1800 + 1)}")

        with col_synth:
            if st.button("Synthesize Voice Narration"):
                with st.spinner("Synthesizing voice narration with Sarvam Bulbul..."):
                    try:
                        audio_bytes = generate_speech(
                            text=script_for_tts,
                            target_lang=target_lang_code,
                            speaker=voice_speaker,
                            pace=narration_pace,
                            api_key=sarvam_key_input,
                        )
                        st.session_state.generated_audio = audio_bytes
                        st.success("Voice narration generated successfully!")

                    except SarvamAPIException as e:
                        st.error(f"Narration Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")

        if st.session_state.generated_audio:
            st.markdown("---")
            st.markdown("#### 🎧 Listen to Voice Narration")
            st.audio(st.session_state.generated_audio, format="audio/wav")

            filename = f"heritage_{st.session_state.selected_content_type}_{voice_speaker}.wav"
            st.download_button(
                label="⬇️ Download Narration (.wav)",
                data=st.session_state.generated_audio,
                file_name=filename,
                mime="audio/wav",
            )
