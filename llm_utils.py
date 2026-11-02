"""
llm_utils.py - Heritage Storytelling Agent LLM Engine

Generates 3 distinctive formats from historical manuscripts & oral archives:
1. Interactive Historical Story (Narrative format)
2. Archival Podcast Script (Conversational dual-host format)
3. Guided Museum / Heritage Walk Tour (Immersive 2nd-person docent format)

Supports multi-provider execution:
- Sarvam AI LLM ('sarvam-105b-conversations' / 'sarvam-105b')
- OpenAI ('gpt-4o' / 'gpt-4o-mini')
- Google Gemini ('gemini-2.5-flash' / 'gemini-1.5-flash')
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# =============================================================================
# Curated Prompt Templates
# =============================================================================

STORY_PROMPT_TEMPLATE = """You are a master historical chronicler and cultural storyteller.
Your task is to transform the provided historical manuscript, archival excerpt, or oral testimony into an evocative, immersive, and historically authentic narrative.

Context / Source Material:
\"\"\"
{source_text}
\"\"\"

Guidelines for the Story:
1. Title: Give the story a majestic, evocative title reflecting its historical essence.
2. Atmospheric Setting: Establish the era, geography, smells, sounds (e.g., temple bells, desert caravans, courtroom murmurings, rustling parchment).
3. Human Experience: Center the narrative around real or representative historical figures, highlighting their dilemmas, devotion, valor, or everyday resilience.
4. Structure:
   - Prelude / The Awakening
   - The Unfolding Event / Conflict
   - Climax & Cultural Legacy
5. Language: Deliver the story in {language_name}. Use rich, poetic, and historically dignified vocabulary.
6. Fidelity: Respect the historical facts, folk traditions, and context in the source material without fabricating modern anachronisms.

Begin your story below:
"""

PODCAST_PROMPT_TEMPLATE = """You are the lead producer and scriptwriter for "Chronicles of Heritage" — a celebrated podcast that unearths lost stories from Indian manuscripts and oral history.

Context / Source Material:
\"\"\"
{source_text}
\"\"\"

Format Guidelines:
1. Hosts:
   - **Arjun**: A passionate historian and archival scholar who loves ancient scripts, dates, architecture, and deciphering clues.
   - **Meera**: An inquisitive cultural journalist who asks deep questions, connects the past to living traditions, and wonders about the human stories.
2. Dynamic Dialogue: Create an engaging, lively, and natural conversation between Arjun and Meera. Include genuine curiosity, mild debates, and moments of awe.
3. Audio Directions: Include brief atmospheric audio cues enclosed in brackets, e.g.:
   - [Theme Music: Traditional sitar & ambient rain fades in]
   - [Sound Effect: Turning brittle parchment pages]
   - [Ambient: Temple gongs in the distance]
4. Content Structure:
   - **Hook**: Intriguing question or discovery.
   - **The Archive**: Unpacking the source document / oral recording.
   - **The Story Within**: Reconstructing what happened and why it matters today.
   - **Takeaway & Sign-off**: Arjun and Meera reflecting on cultural memory.
5. Language: Write the podcast script in {language_name}.

Begin the Podcast Script below:
"""

TOUR_PROMPT_TEMPLATE = """You are the Chief Curator and Docent of a world-renowned National Heritage Museum.
You are designing an immersive audio-guided tour for visitors experiencing an exhibition centered around this historical artifact, manuscript, or oral testimony.

Context / Source Material:
\"\"\"
{source_text}
\"\"\"

Guidelines for Guided Tour:
1. Perspective: Speak directly to the listener in the second person ("You are now standing before...", "Look closely at the faded saffron ink...", "Notice the deliberate brushwork...").
2. Tone: Warm, authoritative, reverent, and deeply captivating.
3. Stops / Gallery Structure:
   - **Exhibition Introduction**: Setting the scene and era as visitors enter the gallery.
   - **Stop 1 - The Relic / Folio Revealed**: Detailed visual and physical breakdown of the manuscript or voice recording.
   - **Stop 2 - The Historical Crucible**: The world in which this artifact was born (kingdom, dynasty, village, or crisis).
   - **Stop 3 - The Living Memory**: Why this piece survives and what it teaches future generations.
   - **Curator's Reflection**: Parting words before directing visitors to the next chamber.
4. Language: Deliver the tour in {language_name}.

Begin the Guided Tour Script below:
"""


# =============================================================================
# Text Preparation for Voice Narration
# =============================================================================

def clean_script_for_narration(script_text: str) -> str:
    """
    Cleans script text for text-to-speech narration:
    - Removes stage/audio cues like [Sound Effect: ...] or [Music Fade In]
    - Cleans host prefixes like 'Arjun:' or 'Meera:' so dialogue flows naturally
    - Removes markdown headers and asterisks
    """
    if not script_text:
        return ""

    # Remove text inside square brackets [Sound Effect: ...]
    cleaned = re.sub(r"\[.*?\]", "", script_text)

    # Remove markdown headers (###, ##, #)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

    # Remove speaker prefixes at start of line (e.g., 'Arjun:', 'Meera:', '**Arjun:**')
    cleaned = re.sub(r"^\*{0,2}[A-Za-z0-9\s]+:\*{0,2}\s*", "", cleaned, flags=re.MULTILINE)

    # Remove bold/italic markdown markers
    cleaned = cleaned.replace("**", "").replace("*", "")

    # Clean redundant whitespace and empty lines
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n\n".join(lines)


# =============================================================================
# Multi-Provider Content Generation
# =============================================================================

def generate_heritage_content(
    source_text: str,
    content_type: str = "story",
    language_name: str = "English",
    provider: str = "sarvam",
    api_key: Optional[str] = None,
) -> str:
    """
    Generates historical narrative content using the chosen LLM provider.

    Args:
        source_text: Transcribed or translated archival text.
        content_type: One of 'story', 'podcast', 'tour'.
        language_name: Output language (e.g. 'English', 'Hindi', 'Bengali').
        provider: 'sarvam' (Sarvam-105B), 'openai' (GPT-4o), or 'gemini' (Gemini-2.0).
        api_key: Optional API key override for the selected provider.

    Returns:
        Generated story, podcast, or tour script text.
    """
    if not source_text or not source_text.strip():
        raise ValueError("Source archival text cannot be empty.")

    # Select appropriate prompt template
    if content_type == "podcast":
        prompt = PODCAST_PROMPT_TEMPLATE.format(source_text=source_text, language_name=language_name)
        system_instruction = "You are a master audio scriptwriter and historical researcher for museum podcasts."
    elif content_type == "tour":
        prompt = TOUR_PROMPT_TEMPLATE.format(source_text=source_text, language_name=language_name)
        system_instruction = "You are an eloquent museum curator creating immersive audio-guided tours."
    else:  # default "story"
        prompt = STORY_PROMPT_TEMPLATE.format(source_text=source_text, language_name=language_name)
        system_instruction = "You are a celebrated historical fiction author and cultural historian."

    # Dispatch to requested provider
    provider_clean = (provider or "sarvam").lower().strip()

    if provider_clean == "openai":
        return _generate_with_openai(prompt, system_instruction, api_key)
    elif provider_clean == "gemini":
        return _generate_with_gemini(prompt, system_instruction, api_key)
    else:  # 'sarvam' default
        return _generate_with_sarvam(prompt, system_instruction, api_key)


# =============================================================================
# Provider Implementations
# =============================================================================

def _generate_with_sarvam(prompt: str, system_instruction: str, api_key: Optional[str] = None) -> str:
    """Generate content using Sarvam 105B LLM."""
    from sarvam_utils import get_sarvam_client

    client = get_sarvam_client(api_key)
    response = client.chat.completions(
        model="sarvam-105b-conversations",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def _get_secret(key_name: str, fallback_arg: Optional[str] = None) -> Optional[str]:
    if fallback_arg and fallback_arg.strip():
        return fallback_arg.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            val = st.secrets[key_name]
            if val and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    return os.environ.get(key_name)

def _generate_with_openai(prompt: str, system_instruction: str, api_key: Optional[str] = None) -> str:
    """Generate content using OpenAI GPT-4o."""
    key = _get_secret("OPENAI_API_KEY", api_key)
    if not key:
        raise ValueError("OpenAI API Key is required. Please set OPENAI_API_KEY in .env, Streamlit Secrets, or the sidebar.")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI generation failed: {str(e)}")


def _generate_with_gemini(prompt: str, system_instruction: str, api_key: Optional[str] = None) -> str:
    """Generate content using Google Gemini."""
    key = _get_secret("GEMINI_API_KEY", api_key)
    if not key:
        raise ValueError("Gemini API Key is required. Please set GEMINI_API_KEY in .env, Streamlit Secrets, or the sidebar.")

    try:
        from google import genai
        client = genai.Client(api_key=key)
        full_content = f"{system_instruction}\n\n{prompt}"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_content,
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {str(e)}")
