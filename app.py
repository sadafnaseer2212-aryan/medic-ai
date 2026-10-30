"""
MediGuide AI — AI-Powered Medical Symptom Assessment & Patient Guidance Assistant
================================================================================
A LangChain + OpenAI + Streamlit application (educational prototype).

    Run:   pip install -r requirements.txt
           streamlit run app.py

IMPORTANT: Educational AI prototype only — NOT a doctor, NOT a diagnosis.
Always consult a qualified healthcare professional. In an emergency call your
local emergency number (1122 Pakistan · 911 US · 999 UK · 112 EU) immediately.

API KEY: never written in this file. Paste it on the secure "Connect" screen
(kept only in session memory) or put OPENAI_API_KEY=sk-... in a .env file.

How this file is organised (one section per responsibility)
-----------------------------------------------------------
  CONFIG ........ settings, languages, form options, urgency colours
  ICONS ......... inline SVG icons
  PROMPTS ....... PromptTemplate + ChatPromptTemplate + JSON schema + safety rules
  CACHE MANAGER . InMemoryCache vs SQLiteCache  (set_llm_cache)
  UTILS ......... safe JSON parsing, validation, safety floor
  CHAINS ........ ChatOpenAI, LLMChain, .stream(), System/Human/AI messages
  HISTORY ....... SQLite session history
  PDF REPORT .... downloadable multilingual PDF
  VOICE ......... voice input (speech-to-text)
  STYLES ........ CSS design system, light/dark themes, animations
  COMPONENTS .... HTML building blocks
  STREAMLIT UI .. Connect screen, chat assistant, history, analytics, how it works

Data flow: chat input -> build prompt inputs -> select cache -> build LLM +
LLMChain -> run chain -> parse JSON -> stream narrative -> results card.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sqlite3
import time
import urllib.request
import uuid
import warnings
from contextlib import closing
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# LLMChain moved to the `langchain-classic` package in LangChain 1.x.
# Try the new location first, then the old one, so both versions work.
try:
    from langchain_classic.chains import LLMChain
except ImportError:  # pragma: no cover - older LangChain versions
    from langchain.chains import LLMChain


# ============================================================================
# CONFIG — settings, languages, form options, urgency colours
# ============================================================================
# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent             # folder that contains app.py
DATA_DIR = BASE_DIR / "data"                            # SQLite files live here
FONTS_DIR = DATA_DIR / "fonts"                          # PDF fonts are downloaded here once
DATA_DIR.mkdir(exist_ok=True)
FONTS_DIR.mkdir(exist_ok=True)

LLM_CACHE_DB = DATA_DIR / "langchain_cache.db"          # SQLiteCache file
HISTORY_DB = DATA_DIR / "mediguide_history.db"          # patient session history

# --------------------------------------------------------------------------
# Environment (.env) — loaded once at import time
# --------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env")


def get_env_api_key() -> str | None:
    """Return the OpenAI key from .env / environment, or None if not set."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    # Ignore the placeholder that ships in .env.example
    if not key or key.startswith("sk-your") or key == "sk-...":
        return None
    return key


# --------------------------------------------------------------------------
# App identity
# --------------------------------------------------------------------------
APP_NAME = "MediGuide AI"
APP_TAGLINE = "AI-Powered Medical Symptom Assessment & Patient Guidance"
APP_DESCRIPTION = (
    "An educational AI assistant that turns your symptoms into clear, "
    "structured, safety-first guidance — and helps you prepare for a "
    "conversation with a real healthcare professional."
)
APP_VERSION = "1.0.0"

DISCLAIMER_SHORT = (
    "Educational prototype only — not a doctor, not a diagnosis. "
    "Always consult a qualified healthcare professional."
)
DISCLAIMER_FULL = (
    "MediGuide AI is an educational AI prototype. It is NOT a replacement for "
    "a licensed doctor, a professional diagnosis, emergency services, or "
    "medical treatment, and it never provides a confirmed diagnosis. Always "
    "consult a qualified healthcare professional. If you think you may be "
    "having a medical emergency, call your local emergency number "
    "(e.g. 1122 in Pakistan, 911 in the US, 999 in the UK) immediately."
)

# --------------------------------------------------------------------------
# Model configuration (shown in the sidebar)
# --------------------------------------------------------------------------
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2          # low = more consistent, safer answers
TRANSCRIBE_MODEL = "whisper-1"     # used for voice symptom input

CACHE_MODES = ["SQLite cache (disk)", "In-memory cache (RAM)", "Off"]
DEFAULT_CACHE_MODE = CACHE_MODES[0]

# --------------------------------------------------------------------------
# Languages — name shown in UI -> settings
#   rtl:   right-to-left script (affects layout + PDF)
#   script/lang: OpenType tags used by the PDF text-shaping engine
# --------------------------------------------------------------------------
LANGUAGES: dict[str, dict] = {
    "English":    {"native": "English",    "rtl": False, "script": "latn", "lang": "eng", "flag": "EN"},
    "Urdu":       {"native": "اردو",        "rtl": True,  "script": "arab", "lang": "urd", "flag": "UR"},
    "Arabic":     {"native": "العربية",     "rtl": True,  "script": "arab", "lang": "ara", "flag": "AR"},
    "Hindi":      {"native": "हिन्दी",        "rtl": False, "script": "deva", "lang": "hin", "flag": "HI"},
    "Spanish":    {"native": "Español",    "rtl": False, "script": "latn", "lang": "esp", "flag": "ES"},
    "French":     {"native": "Français",   "rtl": False, "script": "latn", "lang": "fra", "flag": "FR"},
    "German":     {"native": "Deutsch",    "rtl": False, "script": "latn", "lang": "deu", "flag": "DE"},
    "Turkish":    {"native": "Türkçe",     "rtl": False, "script": "latn", "lang": "trk", "flag": "TR"},
    "Portuguese": {"native": "Português",  "rtl": False, "script": "latn", "lang": "ptg", "flag": "PT"},
    "Indonesian": {"native": "Bahasa Indonesia", "rtl": False, "script": "latn", "lang": "ind", "flag": "ID"},
}
DEFAULT_LANGUAGE = "English"

# --------------------------------------------------------------------------
# Form options
# --------------------------------------------------------------------------
GENDERS = ["Female", "Male", "Non-binary / Other", "Prefer not to say"]

DURATIONS = [
    "Less than 24 hours",
    "1-3 days",
    "4-7 days",
    "1-2 weeks",
    "2-4 weeks",
    "More than a month",
]

COMMON_SYMPTOMS = sorted([
    "Fever", "Cough", "Sore throat", "Runny nose", "Headache", "Fatigue",
    "Shortness of breath", "Chest pain", "Nausea", "Vomiting", "Diarrhea",
    "Abdominal pain", "Dizziness", "Muscle aches", "Joint pain", "Rash",
    "Chills", "Loss of taste or smell", "Back pain", "Palpitations",
    "Blurred vision", "Confusion", "Sneezing", "Ear pain", "Insomnia",
    "Loss of appetite", "Swelling", "Numbness or tingling", "Wheezing",
    "Frequent urination", "Constipation", "Anxiety",
])

# Symptoms that are "red flags" — used by the safety net in utils.py
RED_FLAG_SYMPTOMS = {"Chest pain", "Shortness of breath", "Confusion",
                     "Numbness or tingling", "Blurred vision", "Palpitations"}

# --------------------------------------------------------------------------
# Urgency levels — colour, icon and plain-language meaning
# --------------------------------------------------------------------------
URGENCY_LEVELS: dict[str, dict] = {
    "LOW": {
        "color": "#10B981", "soft": "rgba(16,185,129,.12)", "score": 18,
        "icon": "circle-check", "label": "Low",
        "meaning": "Likely manageable with self-care and monitoring.",
    },
    "MEDIUM": {
        "color": "#F59E0B", "soft": "rgba(245,158,11,.14)", "score": 45,
        "icon": "clock", "label": "Medium",
        "meaning": "Book a visit with a healthcare professional soon.",
    },
    "HIGH": {
        "color": "#F97316", "soft": "rgba(249,115,22,.14)", "score": 72,
        "icon": "triangle-alert", "label": "High",
        "meaning": "Seek medical care promptly — today if possible.",
    },
    "EMERGENCY": {
        "color": "#EF4444", "soft": "rgba(239,68,68,.14)", "score": 95,
        "icon": "siren", "label": "Emergency",
        "meaning": "Seek emergency help immediately.",
    },
}

# Ready-made examples (match the testing scenarios in the brief)
EXAMPLE_SCENARIOS = {
    "Common cold": {
        "age": "25", "gender": "Female", "symptoms": ["Runny nose", "Sore throat"],
        "duration": "1-3 days", "severity": 2, "conditions": "", "medications": "",
        "notes": "Started after a rainy day. Otherwise feeling fine.", "other": "",
    },
    "Fever & cough": {
        "age": "40", "gender": "Male", "symptoms": ["Fever", "Cough", "Fatigue"],
        "duration": "4-7 days", "severity": 6, "conditions": "Mild asthma",
        "medications": "Salbutamol inhaler as needed",
        "notes": "Fever around 38.5°C in the evenings.", "other": "",
    },
    "Chest pain": {
        "age": "58", "gender": "Male", "symptoms": ["Chest pain", "Shortness of breath"],
        "duration": "Less than 24 hours", "severity": 9,
        "conditions": "High blood pressure", "medications": "Amlodipine 5 mg",
        "notes": "Pressure-like pain spreading to the left arm, sweating.", "other": "",
    },
}


# ============================================================================
# ICONS — inline SVG icons (Lucide, ISC licence)
# ============================================================================
ICONS = {
    "heart-pulse": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" /> <path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27" />',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2" />',
    "stethoscope": '<path d="M11 2v2" /> <path d="M5 2v2" /> <path d="M5 3H4a2 2 0 0 0-2 2v4a6 6 0 0 0 12 0V5a2 2 0 0 0-2-2h-1" /> <path d="M8 15a6 6 0 0 0 12 0v-3" /> <circle cx="20" cy="10" r="2" />',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" /> <path d="m9 12 2 2 4-4" />',
    "key-round": '<path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z" /> <circle cx="16.5" cy="7.5" r=".5" fill="currentColor" />',
    "zap": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z" />',
    "globe": '<circle cx="12" cy="12" r="10" /> <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" /> <path d="M2 12h20" />',
    "mic": '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /> <path d="M19 10v2a7 7 0 0 1-14 0v-2" /> <line x1="12" x2="12" y1="19" y2="22" />',
    "file-down": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" /> <path d="M14 2v4a2 2 0 0 0 2 2h4" /> <path d="M12 18v-6" /> <path d="m9 15 3 3 3-3" />',
    "chart-pie": '<path d="M21 12c.552 0 1.005-.449.95-.998a10 10 0 0 0-8.953-8.951c-.55-.055-.998.398-.998.95v8a1 1 0 0 0 1 1z" /> <path d="M21.21 15.89A10 10 0 1 1 8 2.83" />',
    "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /> <path d="M3 3v5h5" /> <path d="M12 7v5l4 2" />',
    "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" /> <path d="M12 9v4" /> <path d="M12 17h.01" />',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2" /> <path d="M7 11V7a5 5 0 0 1 10 0v4" />',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" /> <path d="M20 3v4" /> <path d="M22 5h-4" /> <path d="M4 17v2" /> <path d="M5 18H3" />',
    "brain-circuit": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" /> <path d="M9 13a4.5 4.5 0 0 0 3-4" /> <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" /> <path d="M3.477 10.896a4 4 0 0 1 .585-.396" /> <path d="M6 18a4 4 0 0 1-1.967-.516" /> <path d="M12 13h4" /> <path d="M12 18h6a2 2 0 0 1 2 2v1" /> <path d="M12 8h8" /> <path d="M16 8V5a2 2 0 0 1 2-2" /> <circle cx="16" cy="13" r=".5" /> <circle cx="18" cy="3" r=".5" /> <circle cx="20" cy="21" r=".5" /> <circle cx="20" cy="8" r=".5" />',
    "clock": '<circle cx="12" cy="12" r="10" /> <polyline points="12 6 12 12 16 14" />',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /> <circle cx="12" cy="7" r="4" />',
    "pill": '<path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z" /> <path d="m8.5 8.5 7 7" />',
    "clipboard-list": '<rect width="8" height="4" x="8" y="2" rx="1" ry="1" /> <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" /> <path d="M12 11h4" /> <path d="M12 16h4" /> <path d="M8 11h.01" /> <path d="M8 16h.01" />',
    "message-circle-question": '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z" /> <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /> <path d="M12 17h.01" />',
    "siren": '<path d="M7 18v-6a5 5 0 1 1 10 0v6" /> <path d="M5 21a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-1a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2z" /> <path d="M21 12h1" /> <path d="M18.5 4.5 18 5" /> <path d="M2 12h1" /> <path d="M12 2v1" /> <path d="m4.929 4.929.707.707" /> <path d="M12 12v6" />',
    "circle-check": '<circle cx="12" cy="12" r="10" /> <path d="m9 12 2 2 4-4" />',
    "arrow-right": '<path d="M5 12h14" /> <path d="m12 5 7 7-7 7" />',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3" /> <path d="M3 5V19A9 3 0 0 0 21 19V5" /> <path d="M3 12A9 3 0 0 0 21 12" />',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2" /> <rect width="6" height="6" x="9" y="9" rx="1" /> <path d="M15 2v2" /> <path d="M15 20v2" /> <path d="M2 15h2" /> <path d="M2 9h2" /> <path d="M20 15h2" /> <path d="M20 9h2" /> <path d="M9 2v2" /> <path d="M9 20v2" />',
    "languages": '<path d="m5 8 6 6" /> <path d="m4 14 6-6 2-3" /> <path d="M2 5h12" /> <path d="M7 2h1" /> <path d="m22 22-5-10-5 10" /> <path d="M14 18h6" />',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />',
    "sun": '<circle cx="12" cy="12" r="4" /> <path d="M12 2v2" /> <path d="M12 20v2" /> <path d="m4.93 4.93 1.41 1.41" /> <path d="m17.66 17.66 1.41 1.41" /> <path d="M2 12h2" /> <path d="M20 12h2" /> <path d="m6.34 17.66-1.41 1.41" /> <path d="m19.07 4.93-1.41 1.41" />',
    "info": '<circle cx="12" cy="12" r="10" /> <path d="M12 16v-4" /> <path d="M12 8h.01" />',
    "list-checks": '<path d="m3 17 2 2 4-4" /> <path d="m3 7 2 2 4-4" /> <path d="M13 6h8" /> <path d="M13 12h8" /> <path d="M13 18h8" />',
    "hospital": '<path d="M12 6v4" /> <path d="M14 14h-4" /> <path d="M14 18h-4" /> <path d="M14 8h-4" /> <path d="M18 12h2a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2h2" /> <path d="M18 22V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v18" />',
    "thermometer": '<path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z" />',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z" /> <path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65" /> <path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65" />',
    "rocket": '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z" /> <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" /> <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" /> <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" /> <path d="M14 2v4a2 2 0 0 0 2 2h4" /> <path d="M10 9H8" /> <path d="M16 13H8" /> <path d="M16 17H8" />',
    "refresh-cw": '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" /> <path d="M21 3v5h-5" /> <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" /> <path d="M8 16H3v5" />',
    "bot": '<path d="M12 8V4H8" /> <rect width="16" height="12" x="4" y="8" rx="2" /> <path d="M2 14h2" /> <path d="M20 14h2" /> <path d="M15 13v2" /> <path d="M9 13v2" />',
    "workflow": '<rect width="8" height="8" x="3" y="3" rx="2" /> <path d="M7 11v4a2 2 0 0 0 2 2h4" /> <rect width="8" height="8" x="13" y="13" rx="2" />',
    "gauge": '<path d="m12 14 4-4" /> <path d="M3.34 19a10 10 0 1 1 17.32 0" />',
    "eye-off": '<path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49" /> <path d="M14.084 14.158a3 3 0 0 1-4.242-4.242" /> <path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143" /> <path d="m2 2 20 20" />',
    "log-out": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /> <polyline points="16 17 21 12 16 7" /> <line x1="21" x2="9" y1="12" y2="12" />',
    "send": '<path d="M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z" /> <path d="m21.854 2.147-10.94 10.939" />',
}


def icon(name: str, size: int = 20, stroke: float = 2, cls: str = "") -> str:
    """Return an inline <svg> string for the given icon name."""
    body = ICONS.get(name, ICONS["info"])
    return (
        f'<svg class="ic {cls}" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )


# ============================================================================
# PROMPTS — PromptTemplate, ChatPromptTemplate, JSON schema, safety rules
# ============================================================================
# --------------------------------------------------------------------------
# 1) The exact JSON structure required by the brief (section 10)
# --------------------------------------------------------------------------
JSON_SCHEMA_DICT = {
    "summary": "",
    "possible_conditions": [{"name": "", "reason": ""}],
    "urgency_level": "",
    "recommended_next_steps": [],
    "questions_for_doctor": [],
    "warning_signs": [],
}
JSON_SCHEMA = json.dumps(JSON_SCHEMA_DICT, indent=2)

# --------------------------------------------------------------------------
# 2) Safety rules — shared by every system prompt
# --------------------------------------------------------------------------
SAFETY_RULES = """\
You are MediGuide AI, an EDUCATIONAL medical-information assistant. You are NOT a doctor.

NON-NEGOTIABLE SAFETY RULES
1. Never give a confirmed diagnosis. Use cautious language such as
   "may be related to", "could be consistent with", "is one possibility".
2. Possible conditions are listed for EDUCATION ONLY, ordered from most to least likely.
3. Always encourage the person to consult a qualified healthcare professional.
4. If ANY symptom could indicate a life-threatening problem (e.g. chest pain with
   shortness of breath, stroke signs, severe bleeding, fainting, suicidal thoughts,
   difficulty breathing, severe allergic reaction), set urgency to EMERGENCY and tell
   the person to call their local emergency number / go to the nearest emergency
   department IMMEDIATELY.
5. Never recommend prescription drugs or specific doses. General self-care
   (rest, fluids, monitoring) is fine.
6. Be calm, kind, clear and non-alarming, at a reading level a teenager could follow.
7. Take age, existing conditions and medications into account (e.g. elderly,
   pregnancy, chronic illness raise the level of caution).

URGENCY SCALE (use exactly one of these English words)
- LOW:       mild, self-limiting; home care and monitoring are reasonable.
- MEDIUM:    should see a healthcare professional within a few days.
- HIGH:      should get medical care today / within 24 hours.
- EMERGENCY: needs emergency services right now.
"""

# --------------------------------------------------------------------------
# 3) PromptTemplate — a single-string template with variables
#    Used to turn the raw form data into a tidy "patient profile" block.
# --------------------------------------------------------------------------
PATIENT_PROFILE_TEMPLATE = PromptTemplate(
    input_variables=[
        "age", "gender", "symptoms", "duration", "severity",
        "existing_conditions", "medications", "notes",
    ],
    template=(
        "PATIENT PROFILE\n"
        "- Age: {age}\n"
        "- Gender: {gender}\n"
        "- Symptoms: {symptoms}\n"
        "- Duration: {duration}\n"
        "- Severity (1-10): {severity}\n"
        "- Existing medical conditions: {existing_conditions}\n"
        "- Current medications: {medications}\n"
        "- Additional notes: {notes}"
    ),
)

# --------------------------------------------------------------------------
# 4) ChatPromptTemplate — structured JSON assessment (used by the LLMChain)
#    {json_schema} is passed in as a variable so we never have to escape the
#    curly braces of the JSON inside the template.
# --------------------------------------------------------------------------
ASSESSMENT_SYSTEM = SAFETY_RULES + """
OUTPUT FORMAT
Return ONLY one valid JSON object — no markdown, no code fences, no extra text.
It must match this exact structure:
{json_schema}

Field rules:
- "summary": 2-3 sentences restating the person's situation.
- "possible_conditions": 2-4 items, each with "name" and a one-sentence "reason".
- "urgency_level": exactly one of LOW, MEDIUM, HIGH, EMERGENCY (in English, upper case).
- "recommended_next_steps": 3-6 short, practical actions.
- "questions_for_doctor": 3-5 questions the person could ask a professional.
- "warning_signs": 3-6 signs that mean they must seek help immediately.
- Keep JSON KEYS in English. Write all VALUES (except urgency_level) in {language}.
"""

ASSESSMENT_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", ASSESSMENT_SYSTEM),
    ("human",
     "{patient_profile}\n\n"
     "Please analyse this information and respond with the JSON object only. "
     "Answer language: {language}."),
])

# --------------------------------------------------------------------------
# 5) ChatPromptTemplate — friendly narrative that is STREAMED to the UI
# --------------------------------------------------------------------------
NARRATIVE_SYSTEM = SAFETY_RULES + """
TASK
Write a warm, easy-to-read guidance note (about 150-220 words) in {language}.
Structure it as short paragraphs using light Markdown (bold for key phrases,
no headings larger than ###). Cover: what the symptoms may mean in general
terms, sensible self-care, when to see a professional, and a closing reminder
that this is educational information and not a diagnosis.
The assessed urgency level is {urgency_level} — keep your tone consistent with it.
If it is EMERGENCY, the VERY FIRST sentence must tell the person to call
emergency services or go to the nearest emergency department immediately.
"""

NARRATIVE_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", NARRATIVE_SYSTEM),
    ("human", "{patient_profile}\n\nPlease write my guidance note in {language}."),
])

# --------------------------------------------------------------------------
# 6) Follow-up chat — shows how AIMessage fits into a conversation
# --------------------------------------------------------------------------
FOLLOWUP_SYSTEM = SAFETY_RULES + """
You are now answering FOLLOW-UP questions about an assessment you already gave.
Keep answers short (under 120 words), in {language}, and stay within the safety
rules. If a question asks for a diagnosis or prescription, politely decline and
point the person to a healthcare professional.
"""


def build_patient_profile(inputs: dict) -> str:
    """Fill the PromptTemplate with the form values and return the text block."""
    return PATIENT_PROFILE_TEMPLATE.format(
        age=inputs["age"],
        gender=inputs["gender"],
        symptoms=inputs["symptoms_text"],
        duration=inputs["duration"],
        severity=inputs["severity"],
        existing_conditions=inputs["existing_conditions"] or "None reported",
        medications=inputs["medications"] or "None reported",
        notes=inputs["notes"] or "None",
    )


# ============================================================================
# CACHE MANAGER — InMemoryCache vs SQLiteCache
# ============================================================================
# The cache objects are created once and re-used (module-level singletons),
# so the in-memory cache keeps its contents across Streamlit reruns.
_IN_MEMORY_CACHE: InMemoryCache | None = None
_SQLITE_CACHE: SQLiteCache | None = None

# Table used by the UI and README to explain the difference
CACHE_COMPARISON = [
    {"Feature": "Stored in", "InMemoryCache": "RAM (memory)", "SQLiteCache": "A file on disk (.db)"},
    {"Feature": "Speed", "InMemoryCache": "Fastest", "SQLiteCache": "Fast, slightly slower"},
    {"Feature": "Survives restart?", "InMemoryCache": "No", "SQLiteCache": "Yes"},
    {"Feature": "Best for", "InMemoryCache": "One session", "SQLiteCache": "Reusing across sessions"},
]


def _in_memory() -> InMemoryCache:
    global _IN_MEMORY_CACHE
    if _IN_MEMORY_CACHE is None:
        _IN_MEMORY_CACHE = InMemoryCache()
    return _IN_MEMORY_CACHE


def _sqlite() -> SQLiteCache:
    global _SQLITE_CACHE
    if _SQLITE_CACHE is None:
        _SQLITE_CACHE = SQLiteCache(database_path=str(LLM_CACHE_DB))
    return _SQLITE_CACHE


def configure_cache(mode: str) -> str:
    """Register the cache that matches the sidebar choice. Returns a short label."""
    if mode == CACHE_MODES[0]:          # SQLite
        set_llm_cache(_sqlite())
        return "sqlite"
    if mode == CACHE_MODES[1]:          # In-memory
        set_llm_cache(_in_memory())
        return "memory"
    set_llm_cache(None)                 # Off
    return "off"


def clear_cache(mode: str) -> None:
    """Empty the selected cache (useful to demo the 'first run' again)."""
    if mode == CACHE_MODES[0]:
        _sqlite().clear()
    elif mode == CACHE_MODES[1]:
        _in_memory().clear()


# --------------------------------------------------------------------------
# Helpers that use the SAME key format LangChain uses internally for chat
# models (prompt = dumps(messages), llm_string = model settings). This lets us
# (a) tell the user whether an answer came from the cache, and
# (b) cache the *streamed* narrative too — LangChain's .stream() bypasses the
#     cache, so we look it up / store it ourselves.
# --------------------------------------------------------------------------
def _key(llm, messages) -> tuple[str, str]:
    return dumps(messages), llm._get_llm_string()


def cache_lookup(llm, messages) -> str | None:
    """Return the cached text for these messages, or None on a miss."""
    cache = get_llm_cache()
    if cache is None:
        return None
    prompt, llm_string = _key(llm, messages)
    hit = cache.lookup(prompt, llm_string)
    if hit:
        return hit[0].text if hasattr(hit[0], "text") else str(hit[0])
    return None


def cache_store(llm, messages, text: str) -> None:
    """Save a finished (streamed) answer into the active cache."""
    cache = get_llm_cache()
    if cache is None or not text:
        return
    prompt, llm_string = _key(llm, messages)
    cache.update(prompt, llm_string, [ChatGeneration(message=AIMessage(content=text))])


# ============================================================================
# UTILS — safe JSON parsing, validation, safety floor
# ============================================================================
REQUIRED_KEYS = [
    "summary", "possible_conditions", "urgency_level",
    "recommended_next_steps", "questions_for_doctor", "warning_signs",
]


# --------------------------------------------------------------------------
# JSON handling
# --------------------------------------------------------------------------
def clean_json_text(text: str) -> str:
    """
    Remove things models sometimes add around JSON:
      * ```json ... ``` code fences
      * a sentence before or after the object
    and return just the {...} part.
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)   # opening fence
    text = re.sub(r"\s*```$", "", text)                                   # closing fence
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        text = text[first:last + 1]
    return text.strip()


def safe_parse_json(raw: str) -> tuple[dict | None, str | None]:
    """
    Try to turn the model's reply into a Python dict.

    Returns (data, None) on success or (None, error_message) on failure.
    """
    cleaned = clean_json_text(raw)
    if not cleaned:
        return None, "The model returned an empty response."
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Second chance: remove trailing commas, a common small mistake
        try:
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", cleaned))
        except json.JSONDecodeError:
            return None, f"The response was not valid JSON ({exc.msg} at position {exc.pos})."
    if not isinstance(data, dict):
        return None, "The response was JSON, but not an object with the expected fields."
    return normalize_assessment(data), None


def _as_list(value: Any) -> list[str]:
    """Make sure a field is a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def normalize_urgency(value: Any) -> str:
    """Map anything the model returns ('high', 'Urgent', ...) onto our 4 levels."""
    text = str(value or "").upper()
    if "EMERG" in text or "CRITICAL" in text or "IMMEDIATE" in text:
        return "EMERGENCY"
    if "HIGH" in text or "URGENT" in text or "SEVERE" in text:
        return "HIGH"
    if "MED" in text or "MODERATE" in text:
        return "MEDIUM"
    if "LOW" in text or "MILD" in text or "MINOR" in text:
        return "LOW"
    return "MEDIUM"          # unknown -> cautious middle ground


def normalize_assessment(data: dict) -> dict:
    """Guarantee every required key exists with the right type."""
    conditions = []
    for item in data.get("possible_conditions") or []:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            reason = str(item.get("reason", "")).strip()
        else:
            name, reason = str(item).strip(), ""
        if name:
            conditions.append({"name": name, "reason": reason})

    return {
        "summary": str(data.get("summary", "")).strip(),
        "possible_conditions": conditions,
        "urgency_level": normalize_urgency(data.get("urgency_level")),
        "recommended_next_steps": _as_list(data.get("recommended_next_steps")),
        "questions_for_doctor": _as_list(data.get("questions_for_doctor")),
        "warning_signs": _as_list(data.get("warning_signs")),
    }


# --------------------------------------------------------------------------
# Safety net — never let the AI under-rate an obvious red flag
# --------------------------------------------------------------------------
ORDER = ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]


def apply_safety_floor(assessment: dict, form: dict) -> tuple[dict, str | None]:
    """
    Raise the urgency if clear red flags are present but the model rated it low.
    Returns (assessment, note) — note explains any change for the UI.
    """
    symptoms = set(form.get("symptoms", []))
    text = (form.get("other_symptoms", "") + " " + form.get("notes", "")).lower()
    floor, reason = "LOW", None

    if {"Chest pain", "Shortness of breath"} <= symptoms or (
        "chest pain" in text and "breath" in text
    ):
        floor, reason = "EMERGENCY", "chest pain together with shortness of breath"
    elif symptoms & RED_FLAG_SYMPTOMS and form.get("severity", 0) >= 7:
        floor, reason = "HIGH", "a red-flag symptom with high severity"
    elif form.get("severity", 0) >= 9:
        floor, reason = "HIGH", "a very high severity score"

    current = assessment["urgency_level"]
    if ORDER.index(floor) > ORDER.index(current):
        assessment["urgency_level"] = floor
        return assessment, (
            f"Safety check: urgency raised from {current} to {floor} because of {reason}."
        )
    return assessment, None


# --------------------------------------------------------------------------
# Form validation
# --------------------------------------------------------------------------
def validate_form(form: dict) -> list[str]:
    """Return a list of problems (empty list = all good)."""
    problems = []
    age = str(form.get("age", "")).strip()
    if not age:
        problems.append("Please enter the patient's age.")
    elif not age.isdigit() or not (0 < int(age) <= 120):
        problems.append("Age should be a whole number between 1 and 120.")
    if not form.get("symptoms") and not form.get("other_symptoms", "").strip():
        problems.append("Please select or describe at least one symptom — the AI is not called without symptoms.")
    return problems


def build_symptoms_text(form: dict) -> str:
    """Combine multiselect symptoms with the free-text box."""
    parts = list(form.get("symptoms", []))
    other = form.get("other_symptoms", "").strip()
    if other:
        parts.append(other)
    return ", ".join(parts)


def severity_label(value: int) -> tuple[str, str]:
    """Human label + colour for the 1-10 severity slider."""
    if value <= 3:
        return "Mild", URGENCY_LEVELS["LOW"]["color"]
    if value <= 6:
        return "Moderate", URGENCY_LEVELS["MEDIUM"]["color"]
    if value <= 8:
        return "Severe", URGENCY_LEVELS["HIGH"]["color"]
    return "Very severe", URGENCY_LEVELS["EMERGENCY"]["color"]


def mask_key(key: str) -> str:
    """sk-abc...wxyz — show a key safely."""
    if not key:
        return ""
    return f"{key[:5]}••••••{key[-4:]}" if len(key) > 12 else "••••"


# ============================================================================
# CHAINS — ChatOpenAI, LLMChain, streaming, System/Human/AI messages
# ============================================================================
# LLMChain was moved to the `langchain-classic` package in LangChain 1.x.
# We try the new location first and fall back to the old one, so the code
# works on both older and newer installs.


# LLMChain is required by the assignment brief. LangChain marks it as "legacy"
# (the modern style is `prompt | llm`), so we silence that notice to keep the
# terminal clean. The chain still works exactly the same.
warnings.filterwarnings("ignore", message=".*LLMChain.*")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")


# --------------------------------------------------------------------------
# 1) ChatOpenAI — the model
# --------------------------------------------------------------------------
def build_llm(api_key: str, model: str, temperature: float, json_mode: bool = False):
    """
    Create a ChatOpenAI model.

    json_mode=True asks OpenAI to guarantee a JSON object back
    (we STILL parse it defensively in utils.py — belt and braces).
    """
    from langchain_openai import ChatOpenAI

    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,          # passed in at runtime — never hard-coded
        timeout=60,
        max_retries=2,
        model_kwargs=extra,
    )


# --------------------------------------------------------------------------
# 2) Build the inputs every prompt needs (PromptTemplate is used here)
# --------------------------------------------------------------------------
def build_prompt_inputs(form: dict, language: str) -> dict:
    """Turn validated form data into the variables our templates expect."""
    profile = build_patient_profile(form)          # <- PromptTemplate.format()
    return {
        "patient_profile": profile,
        "language": language,
        "json_schema": JSON_SCHEMA,
    }


# --------------------------------------------------------------------------
# 3) LLMChain — reusable assessment pipeline (prompt -> model -> text)
# --------------------------------------------------------------------------
def build_assessment_chain(llm) -> LLMChain:
    """A reusable chain: ChatPromptTemplate + ChatOpenAI."""
    return LLMChain(llm=llm, prompt=ASSESSMENT_CHAT_TEMPLATE, verbose=False)


def run_assessment(chain: LLMChain, prompt_inputs: dict) -> dict:
    """
    Run the chain and time it.

    Returns {"raw": <model text>, "elapsed": seconds, "cached": bool}.
    `cached` is checked BEFORE the call by looking the prompt up in the
    active LangChain cache — so the UI can honestly show "served from cache".
    """
    messages = ASSESSMENT_CHAT_TEMPLATE.format_messages(**prompt_inputs)
    was_cached = cache_lookup(chain.llm, messages) is not None

    start = time.perf_counter()
    result = chain.invoke(prompt_inputs)            # LangChain checks the cache here
    elapsed = time.perf_counter() - start

    return {"raw": result["text"], "elapsed": elapsed, "cached": was_cached}


# --------------------------------------------------------------------------
# 4) Streaming — yields text chunks for st.write_stream()
# --------------------------------------------------------------------------
def stream_narrative(llm, prompt_inputs: dict, urgency_level: str, stats: dict | None = None) -> Iterator[str]:
    """
    Stream the human-friendly guidance note chunk by chunk.

    Note: LangChain's .stream() does not consult the LLM cache, so we check
    the SAME cache ourselves first. On a hit we replay the stored text quickly
    (still a typing effect); on a miss we stream live and then save it.
    `stats` (optional dict) receives {"cached": bool, "elapsed": float}.
    """
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**prompt_inputs, urgency_level=urgency_level)
    start = time.perf_counter()

    cached_text = cache_lookup(llm, messages)
    if cached_text:
        if stats is not None:
            stats["cached"] = True
        words = cached_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.004)
    else:
        if stats is not None:
            stats["cached"] = False
        pieces: list[str] = []
        for chunk in llm.stream(messages):          # <- the .stream() call
            if chunk.content:
                pieces.append(chunk.content)
                yield chunk.content
        cache_store(llm, messages, "".join(pieces))

    if stats is not None:
        stats["elapsed"] = time.perf_counter() - start


# --------------------------------------------------------------------------
# 5) Raw System / Human / AI messages
# --------------------------------------------------------------------------
def demo_messages(llm, language: str = "English") -> dict:
    """
    A tiny, transparent example of the three message types:

      SystemMessage -> sets the role and the safety rules
      HumanMessage  -> what the user says
      AIMessage     -> what the model said earlier (conversation memory)

    We send System + Human + AI + Human, so the model sees a real conversation
    and answers the LAST human message with that context.
    """
    conversation = [
        SystemMessage(content=FOLLOWUP_SYSTEM.format(language=language)),
        HumanMessage(content="I have had a mild headache since this morning."),
        AIMessage(content="I'm sorry to hear that. Mild headaches are common and often linked to "
                          "tiredness, dehydration or screen time. Have you been drinking enough water?"),
        HumanMessage(content="Not really. What simple things can I try at home?"),
    ]
    start = time.perf_counter()
    reply = llm.invoke(conversation)
    return {
        "messages": conversation,
        "reply": reply.content,
        "elapsed": time.perf_counter() - start,
    }


def build_followup_messages(assessment_summary: str, history: list[dict], question: str, language: str) -> list:
    """
    Build the conversation for a follow-up question.

    `history` is a list like [{"role": "user"|"assistant", "content": "..."}].
    It is converted into HumanMessage / AIMessage objects — this is exactly how
    AIMessage lets the model "remember" its own earlier answers.
    """
    messages = [
        SystemMessage(content=FOLLOWUP_SYSTEM.format(language=language)),
        AIMessage(content=f"Here is the assessment I gave earlier: {assessment_summary}"),
    ]
    for turn in history[-8:]:                       # keep the last few turns
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(content=turn["content"]))
    messages.append(HumanMessage(content=question))
    return messages


def stream_followup(llm, assessment_summary: str, history: list[dict], question: str, language: str) -> Iterator[str]:
    """Stream the answer to a follow-up question (typing effect in the chat)."""
    for chunk in llm.stream(build_followup_messages(assessment_summary, history, question, language)):
        if chunk.content:
            yield chunk.content


# --------------------------------------------------------------------------
# 6) API key check used by the Connect screen
# --------------------------------------------------------------------------
def verify_api_key(api_key: str) -> tuple[bool, str]:
    """Make one cheap, free request (list models) to confirm the key works."""
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Please paste your OpenAI API key."
    if not api_key.startswith("sk-") or len(api_key) < 20:
        return False, "That doesn't look like an OpenAI key — it should start with “sk-”."
    try:
        from openai import AuthenticationError, OpenAI

        client = OpenAI(api_key=api_key, timeout=20)
        client.models.list()
        return True, "Connected successfully."
    except AuthenticationError:
        return False, "OpenAI rejected this key. Please check it and try again."
    except Exception as exc:  # network issues, rate limits, etc.
        return False, f"Couldn't reach OpenAI right now ({type(exc).__name__}). Check your internet and retry."


# ============================================================================
# HISTORY — patient session history in SQLite
# ============================================================================
_SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    age         TEXT,
    gender      TEXT,
    symptoms    TEXT,          -- JSON list
    severity    INTEGER,
    duration    TEXT,
    language    TEXT,
    model       TEXT,
    urgency     TEXT,
    elapsed     REAL,
    cached      INTEGER,
    form_json   TEXT,          -- full form (JSON)
    result_json TEXT,          -- parsed assessment (JSON)
    narrative   TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def save_assessment(form: dict, result: dict, narrative: str, language: str,
                    model: str, elapsed: float, cached: bool) -> int:
    """Insert one assessment and return its id."""
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            """INSERT INTO assessments
               (created_at, age, gender, symptoms, severity, duration, language, model,
                urgency, elapsed, cached, form_json, result_json, narrative)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                form.get("age"), form.get("gender"),
                json.dumps(form.get("symptoms", []), ensure_ascii=False),
                int(form.get("severity", 0)), form.get("duration"),
                language, model, result.get("urgency_level"),
                round(float(elapsed), 3), int(bool(cached)),
                json.dumps(form, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                narrative,
            ),
        )
        return int(cur.lastrowid)


def update_narrative(row_id: int, narrative: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE assessments SET narrative=? WHERE id=?", (narrative, row_id))


def list_assessments(limit: int = 200) -> list[dict]:
    """Newest first."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM assessments ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["symptoms"] = json.loads(d["symptoms"] or "[]")
        d["form"] = json.loads(d.pop("form_json") or "{}")
        d["result"] = json.loads(d.pop("result_json") or "{}")
        out.append(d)
    return out


def clear_history() -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM assessments")


def history_dataframe():
    """Return history as a pandas DataFrame for the analytics page."""
    import pandas as pd

    rows = list_assessments(limit=5000)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


# ============================================================================
# PDF REPORT — branded multilingual report (fpdf2 + HarfBuzz)
# ============================================================================
# Colour palette (matches the app)
INK = (14, 19, 37)
INK2 = (43, 51, 80)
MUTED = (102, 112, 138)
LINE = (228, 231, 241)
SOFT = (247, 248, 252)
CHIP = (238, 240, 255)
BRAND = (79, 70, 229)
BRAND_2 = (124, 58, 237)
TEAL = (14, 165, 164)
GREEN = (16, 185, 129)
SKY = (14, 165, 233)
RED = (239, 68, 68)
AMBER = (217, 119, 6)
WHITE = (255, 255, 255)

PAGE_W, MARGIN = 210, 16
CONTENT_W = PAGE_W - 2 * MARGIN


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _tint(rgb, amount: float) -> tuple[int, int, int]:
    """Mix a colour with white (amount 0 = white, 1 = the colour)."""
    return tuple(round(255 - (255 - v) * amount) for v in rgb)


def _strip_md(text: str) -> str:
    """Remove Markdown symbols (**bold**, ### etc.) for the PDF."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text or "")
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text.strip()


# Free, open-licensed fonts (SIL OFL). They are downloaded ONCE into data/fonts
# the first time a PDF is created; if there is no internet, common system fonts
# are used instead, so the PDF always works.
_GF = "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/"
_NF = "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/"
PDF_FONT_SOURCES = {
    "poppins": (_GF + "Poppins-Regular.ttf", _GF + "Poppins-Bold.ttf"),
    "poppinsm": (_GF + "Poppins-Medium.ttf", _GF + "Poppins-SemiBold.ttf"),
    "naskh": (_NF + "NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Regular.ttf",
              _NF + "NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Bold.ttf"),
    "deva": (_NF + "NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf",
             _NF + "NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf"),
}
PDF_SYSTEM_FONTS = {
    "poppins": ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "poppinsm": [],
    "naskh": ["C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "deva": ["C:/Windows/Fonts/Nirmala.ttf", "C:/Windows/Fonts/mangal.ttf",
             "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf",
             "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"],
}


def _download_font(url: str) -> Path | None:
    target = FONTS_DIR / url.rsplit("/", 1)[-1]
    if target.exists() and target.stat().st_size > 10_000:
        return target
    try:
        with urllib.request.urlopen(url, timeout=12) as resp:
            data = resp.read()
        if len(data) > 10_000:
            target.write_bytes(data)
            return target
    except Exception:
        pass
    return None


def resolve_pdf_fonts() -> dict[str, tuple[Path, Path]]:
    """family -> (regular, bold) font files that exist on this computer."""
    found = {}
    for family, (reg_url, bold_url) in PDF_FONT_SOURCES.items():
        reg = _download_font(reg_url)
        bold = _download_font(bold_url) if reg else None
        if not reg:
            reg = next((Path(p) for p in PDF_SYSTEM_FONTS[family] if Path(p).exists()), None)
        if reg:
            found[family] = (reg, bold or reg)
    return found


class _Report(FPDF):
    """A4 report with a branded header on every page and a numbered footer."""

    def __init__(self, meta: dict):
        super().__init__(format="A4")
        self.meta = meta
        self.set_auto_page_break(auto=True, margin=24)
        self.set_margins(MARGIN, 30, MARGIN)
        self.alias_nb_pages()
        self.families = []
        for family, (reg, bold) in resolve_pdf_fonts().items():
            try:
                self.add_font(family, "", str(reg))
                self.add_font(family, "B", str(bold))
                self.families.append(family)
            except Exception:
                pass
        self.unicode_ok = "poppins" in self.families

    # ---- graceful fallbacks if a font could not be found ---------------
    def set_font(self, family=None, style="", size=0):
        if family and family.lower() not in self.fonts and family.lower() + style.upper() not in self.fonts:
            family = "poppins" if "poppins" in self.families else "helvetica"
        return super().set_font(family, style, size)

    def set_text_shaping(self, *args, **kwargs):
        if self.unicode_ok:
            try:
                return super().set_text_shaping(*args, **kwargs)
            except Exception:
                return None
        return None

    def normalize_text(self, text: str) -> str:
        if not self.is_ttf_font:                       # core font: keep only basic characters
            text = text.encode("latin-1", "replace").decode("latin-1")
        return super().normalize_text(text)

    # ---- drawing helpers ---------------------------------------------------
    def ltr(self):
        self.set_text_shaping(use_shaping_engine=True, direction="ltr", script="latn", language="eng")

    def gradient_rect(self, x, y, w, h, colors, radius=0):
        """Horizontal gradient rectangle (falls back to a solid colour)."""
        try:
            from fpdf.pattern import LinearGradient
            grad = LinearGradient(x, y, x + w, y, [self._rgb_hex(c) for c in colors])
            with self.use_pattern(grad):
                self.rect(x, y, w, h, style="F", round_corners=bool(radius), corner_radius=radius)
        except Exception:
            self.set_fill_color(*colors[0])
            self.rect(x, y, w, h, style="F", round_corners=bool(radius), corner_radius=radius)

    @staticmethod
    def _rgb_hex(c):
        return "#%02x%02x%02x" % c

    def logo(self, x, y, s):
        """Rounded gradient tile with a white heartbeat line."""
        self.gradient_rect(x, y, s, s, [BRAND, BRAND_2], radius=s * 0.28)
        self.set_draw_color(*WHITE)
        self.set_line_width(s * 0.07)
        k = s / 14
        pts = [(2.2, 7.4), (4.8, 7.4), (6.0, 4.6), (7.6, 10.2), (9.0, 6.0), (9.8, 7.4), (11.8, 7.4)]
        self.polyline([(x + px * k, y + py * k) for px, py in pts])
        self.set_line_width(0.2)

    def header(self):
        m = self.meta
        self.gradient_rect(0, 0, PAGE_W, 2.2, [BRAND, BRAND_2, TEAL])
        self.ltr()
        if self.page_no() == 1:
            self.logo(MARGIN, 9, 13)
            self.set_xy(MARGIN + 16, 9.2)
            self.set_text_color(*INK)
            self.set_font("poppins", "B", 16)
            self.cell(80, 7, APP_NAME)
            self.set_xy(MARGIN + 16, 16.2)
            self.set_text_color(*MUTED)
            self.set_font("poppinsm", "", 8.5)
            self.cell(80, 5, "Patient Guidance Report  ·  Educational use only")
            # meta block (right)
            rows = [("REPORT ID", m["report_id"]), ("DATE", m["date"]), ("LANGUAGE", m["language"])]
            x0, w = PAGE_W - MARGIN - 96, 32
            for i, (k, v) in enumerate(rows):
                x = x0 + i * w
                self.set_xy(x, 9.5)
                self.set_font("poppinsm", "", 6.3)
                self.set_text_color(*MUTED)
                self.cell(w, 3.5, k, align="R")
                self.set_xy(x, 13.6)
                self.set_font("poppins", "B", 8.4)
                self.set_text_color(*INK)
                self.cell(w, 4.5, v, align="R")
            self.set_draw_color(*LINE)
            self.set_line_width(0.3)
            self.line(MARGIN, 26, PAGE_W - MARGIN, 26)
            self.set_y(31)
        else:
            self.logo(MARGIN, 8, 7)
            self.set_xy(MARGIN + 9.5, 8.6)
            self.set_font("poppins", "B", 9)
            self.set_text_color(*INK)
            self.cell(60, 6, APP_NAME)
            self.set_xy(PAGE_W - MARGIN - 80, 8.6)
            self.set_font("poppinsm", "", 7.5)
            self.set_text_color(*MUTED)
            self.cell(80, 6, f"Patient Guidance Report  ·  {m['report_id']}", align="R")
            self.set_draw_color(*LINE)
            self.line(MARGIN, 17, PAGE_W - MARGIN, 17)
            self.set_y(23)

    def footer(self):
        self.ltr()
        self.set_y(-15)
        self.set_draw_color(*LINE)
        self.set_line_width(0.3)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.set_y(-12.5)
        self.set_font("poppinsm", "", 6.8)
        self.set_text_color(*MUTED)
        self.cell(CONTENT_W - 30, 5, "Educational AI prototype — not a medical diagnosis. "
                                     "In an emergency call 1122 (PK) · 911 (US) · 999 (UK) · 112 (EU).")
        self.set_font("poppins", "B", 7)
        self.set_text_color(*INK2)
        self.cell(30, 5, f"Page {self.page_no()} of {{nb}}", align="R")


def build_pdf_report(form: dict, result: dict, narrative: str, language: str,
                     model: str, symptoms_text: str) -> bytes:
    """Create the PDF and return its bytes (ready for st.download_button)."""
    lang = LANGUAGES.get(language, LANGUAGES["English"])
    rtl = lang["rtl"]
    now = datetime.now()
    meta = {"report_id": f"MG-{now:%y%m%d-%H%M}", "date": f"{now:%d %b %Y, %H:%M}",
            "language": language}
    pdf = _Report(meta)
    if pdf.families:
        pdf.set_fallback_fonts(pdf.families)
    body_font = {"arab": "naskh", "deva": "deva"}.get(lang["script"], "poppins")
    if body_font not in pdf.families:
        body_font = "poppins"
    lh = 6.4 if body_font != "poppins" else 5.2           # line height for answer text
    pdf.add_page()

    level = result.get("urgency_level", "MEDIUM")
    um = URGENCY_LEVELS.get(level, URGENCY_LEVELS["MEDIUM"])
    ucol = _hex(um["color"])

    def content_dir():
        pdf.set_text_shaping(use_shaping_engine=True, direction="rtl" if rtl else "ltr",
                             script=lang["script"], language=lang["lang"])

    def text_height(w, text, size, style="", font=None):
        content_dir()
        pdf.set_font(font or body_font, style, size)
        return pdf.multi_cell(w, lh, text, dry_run=True, output="HEIGHT")

    def ensure(h):
        """Start a new page if a block of height h would not fit."""
        if pdf.get_y() + h > pdf.h - pdf.b_margin:
            pdf.add_page()

    # ---- 1. urgency hero card -------------------------------------------------
    y = pdf.get_y()
    H = 40
    pdf.set_fill_color(*_tint(ucol, .09))
    pdf.set_draw_color(*_tint(ucol, .45))
    pdf.set_line_width(0.35)
    pdf.rect(MARGIN, y, CONTENT_W, H, style="DF", round_corners=True, corner_radius=4)
    pdf.set_fill_color(*ucol)
    pdf.rect(MARGIN, y, 2.2, H, style="F", round_corners=True, corner_radius=1.1)
    pdf.ltr()
    # pill
    pdf.set_font("poppins", "B", 7)
    pill = "URGENCY LEVEL"
    pw = pdf.get_string_width(pill) + 7
    pdf.set_fill_color(*ucol)
    pdf.rect(MARGIN + 8, y + 6, pw, 5.6, style="F", round_corners=True, corner_radius=2.8)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(MARGIN + 8, y + 6.1)
    pdf.cell(pw, 5.4, pill, align="C")
    pdf.set_xy(MARGIN + 8, y + 13)
    pdf.set_text_color(*ucol)
    pdf.set_font("poppins", "B", 24)
    pdf.cell(100, 11, um["label"].upper())
    pdf.set_xy(MARGIN + 8, y + 25)
    pdf.set_text_color(*INK)
    pdf.set_font("poppins", "B", 10.5)
    pdf.cell(110, 5.5, um["meaning"])
    advice = {"LOW": "Rest, fluids and monitoring are usually enough. Re-check if anything changes.",
              "MEDIUM": "Arrange a visit with a doctor or clinic within the next few days.",
              "HIGH": "Get medical care today — contact a doctor, urgent-care or clinic now.",
              "EMERGENCY": "Call emergency services or go to the nearest emergency department now."}[level]
    pdf.set_xy(MARGIN + 8, y + 31)
    pdf.set_text_color(*MUTED)
    pdf.set_font("poppinsm", "", 8.2)
    pdf.cell(118, 4.5, advice)

    # gauge (semicircle) on the right
    cx, cy, r = PAGE_W - MARGIN - 26, y + 28, 15
    # fpdf2 measures angles on a y-down page, so the TOP half of the dial is 180°..360°
    for col, a0, a1 in [(GREEN, 180, 225), ((245, 158, 11), 225, 270), ((249, 115, 22), 270, 315), (RED, 315, 360)]:
        pdf.set_fill_color(*col)
        pdf.solid_arc(cx - r, cy - r, 2 * r, a0 + 1, a1 - 1, style="F")
    pdf.set_fill_color(*_tint(ucol, .09))
    pdf.circle(cx, cy, r * 0.68, style="F")
    ang = math.radians(180 + um["score"] * 1.8)
    pdf.set_draw_color(*INK)
    pdf.set_line_width(1.1)
    pdf.line(cx, cy, cx + math.cos(ang) * r * 0.86, cy + math.sin(ang) * r * 0.86)
    pdf.set_fill_color(*INK)
    pdf.circle(cx, cy, 2.2, style="F")
    pdf.set_fill_color(*ucol)
    pdf.circle(cx, cy, 1.0, style="F")
    pdf.set_line_width(0.2)
    pdf.set_font("poppinsm", "", 6)
    pdf.set_text_color(*MUTED)
    pdf.set_xy(cx - r - 3, cy + 1.6)
    pdf.cell(10, 3, "LOW")
    pdf.set_xy(cx + r - 9, cy + 1.6)
    pdf.cell(12, 3, "EMERG.", align="R")
    pdf.set_y(y + H + 5)

    # ---- 2. patient tiles -------------------------------------------------------
    sev = int(form.get("severity", 0) or 0)
    sev_label, sev_hex = severity_label(sev) if sev else ("-", "#94A3B8")
    tiles = [("AGE", f"{form.get('age', '-')} years"), ("GENDER", str(form.get("gender", "-"))),
             ("DURATION", str(form.get("duration", "-"))), ("SEVERITY", f"{sev}/10 · {sev_label}")]
    y = pdf.get_y()
    gap = 3.5
    tw = (CONTENT_W - gap * 3) / 4
    for i, (k, v) in enumerate(tiles):
        x = MARGIN + i * (tw + gap)
        pdf.set_fill_color(*SOFT)
        pdf.set_draw_color(*LINE)
        pdf.rect(x, y, tw, 17, style="DF", round_corners=True, corner_radius=2.5)
        pdf.set_xy(x + 4, y + 3)
        pdf.set_font("poppinsm", "", 6.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(tw - 8, 3.5, k)
        pdf.set_xy(x + 4, y + 7.6)
        pdf.set_font("poppins", "B", 10)
        pdf.set_text_color(*INK)
        pdf.cell(tw - 8, 5.5, v)
        if k == "SEVERITY":
            pdf.set_fill_color(*LINE)
            pdf.rect(x + 4, y + 13.6, tw - 8, 1.3, style="F", round_corners=True, corner_radius=.65)
            pdf.set_fill_color(*_hex(sev_hex))
            pdf.rect(x + 4, y + 13.6, max(1.5, (tw - 8) * sev / 10), 1.3, style="F",
                     round_corners=True, corner_radius=.65)
    pdf.set_y(y + 17 + 6)

    # ---- section heading ----------------------------------------------------------
    def heading(num: int, title: str, color, note: str = "", need: float = 22):
        ensure(need)
        pdf.ltr()
        y = pdf.get_y()
        pdf.set_fill_color(*color)
        pdf.circle(MARGIN + 3.3, y + 3.4, 3.3, style="F")
        pdf.set_xy(MARGIN, y + 1.2)
        pdf.set_font("poppins", "B", 7.6)
        pdf.set_text_color(*WHITE)
        pdf.cell(6.6, 4.4, str(num), align="C")
        pdf.set_xy(MARGIN + 9, y)
        pdf.set_font("poppins", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(pdf.get_string_width(title) + 2, 7, title)
        if note:
            pdf.set_font("poppinsm", "", 7.5)
            pdf.set_text_color(*MUTED)
            pdf.cell(80, 7, note)
        pdf.set_y(y + 10)

    def label(text: str):
        pdf.ltr()
        pdf.set_font("poppinsm", "B", 6.8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 4, text.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ---- 3. reported symptoms (chips) + context ------------------------------------------
    heading(1, "What you told us", BRAND)
    label("Reported symptoms")
    pdf.ln(1)
    chips = list(form.get("symptoms") or [])
    x, y = MARGIN, pdf.get_y()
    pdf.set_font("poppins", "B", 8)
    for chip in chips:
        w = pdf.get_string_width(chip) + 8
        if x + w > PAGE_W - MARGIN:
            x, y = MARGIN, y + 8
        pdf.set_fill_color(*CHIP)
        pdf.rect(x, y, w, 6, style="F", round_corners=True, corner_radius=3)
        pdf.set_xy(x, y + 0.4)
        pdf.set_text_color(*BRAND)
        pdf.cell(w, 5.2, chip, align="C")
        x += w + 2.5
    pdf.set_y(y + (8.5 if chips else 0))
    other = (form.get("other_symptoms") or "").strip()
    if other:
        pdf.ltr()
        pdf.set_font(body_font, "", 9)
        pdf.set_text_color(*INK2)
        pdf.multi_cell(0, 5, f"“{other}”", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
    ctx = [("Existing conditions", form.get("existing_conditions")), ("Medications", form.get("medications")),
           ("Notes", form.get("notes"))]
    ctx = [(k, (v or "").strip() or "None reported") for k, v in ctx]
    pdf.ln(1.5)
    y = pdf.get_y()
    cw = (CONTENT_W - 2 * gap) / 3
    heights = []
    for k, v in ctx:
        pdf.ltr()
        pdf.set_font(body_font, "", 8.6)
        heights.append(pdf.multi_cell(cw - 8, 4.6, v, dry_run=True, output="HEIGHT"))
    box_h = max(heights) + 11
    ensure(box_h)
    y = pdf.get_y()
    for i, (k, v) in enumerate(ctx):
        x = MARGIN + i * (cw + gap)
        pdf.set_draw_color(*LINE)
        pdf.set_fill_color(*WHITE)
        pdf.rect(x, y, cw, box_h, style="DF", round_corners=True, corner_radius=2.5)
        pdf.set_xy(x + 4, y + 3)
        pdf.ltr()
        pdf.set_font("poppinsm", "B", 6.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(cw - 8, 3.5, k.upper())
        pdf.set_xy(x + 4, y + 7.5)
        pdf.set_font(body_font, "", 8.6)
        pdf.set_text_color(*(INK2 if v != "None reported" else MUTED))
        pdf.multi_cell(cw - 8, 4.6, v, align="L")
    pdf.set_y(y + box_h + 7)

    # ---- 4. summary + guidance -----------------------------------------------------------
    summary = result.get("summary", "") or "-"
    sh = text_height(CONTENT_W - 14, summary, 10)
    heading(2, "Summary & guidance", BRAND_2, need=sh + 26)
    y = pdf.get_y()
    pdf.set_fill_color(*_tint(BRAND, .07))
    pdf.rect(MARGIN, y, CONTENT_W, sh + 9, style="F", round_corners=True, corner_radius=3)
    pdf.set_fill_color(*BRAND)
    bar_x = PAGE_W - MARGIN - 1.6 if rtl else MARGIN
    pdf.rect(bar_x, y, 1.6, sh + 9, style="F")
    content_dir()
    pdf.set_xy(MARGIN + 7, y + 4.5)
    pdf.set_font(body_font, "B", 10)
    pdf.set_text_color(*INK)
    pdf.multi_cell(CONTENT_W - 14, lh, summary, align="R" if rtl else "L")
    pdf.set_y(y + sh + 9 + 5)
    for para in [p.strip() for p in _strip_md(narrative).split("\n") if p.strip()]:
        h = text_height(CONTENT_W, para, 9.4)
        ensure(h + 2)
        content_dir()
        pdf.set_font(body_font, "", 9.4)
        pdf.set_text_color(*INK2)
        pdf.multi_cell(CONTENT_W, lh, para, align="R" if rtl else "L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2.2)
    pdf.ln(3)

    # ---- list rows with a badge ------------------------------------------------------------------
    def badge(x, y, color, text, soft=False):
        pdf.ltr()
        pdf.set_fill_color(*(_tint(color, .16) if soft else color))
        pdf.circle(x + 3.2, y + 3.2, 3.2, style="F")
        pdf.set_xy(x, y + 1.1)
        pdf.set_font("poppins", "B", 7.4)
        pdf.set_text_color(*(color if soft else WHITE))
        pdf.cell(6.4, 4.2, text, align="C")

    def rows(items: list[str], color, mark=None, box=False, title_items=None):
        """Draw items as tidy rows: badge + text (mirrored for right-to-left)."""
        tw = CONTENT_W - (16 if box else 11)
        for i, item in enumerate(items, 1):
            name = title_items[i - 1] if title_items else None
            h = text_height(tw, item, 9.2) + (text_height(tw, name, 9.6, "B") + 0.8 if name else 0)
            row_h = max(h, 6.4) + (6 if box else 3)
            ensure(row_h)
            y = pdf.get_y()
            if box:
                pdf.set_fill_color(*WHITE)
                pdf.set_draw_color(*LINE)
                pdf.rect(MARGIN, y, CONTENT_W, row_h - 1.8, style="DF", round_corners=True, corner_radius=2.5)
            pad = 4 if box else 0
            bx = (PAGE_W - MARGIN - pad - 6.4) if rtl else (MARGIN + pad)
            badge(bx, y + (3 if box else 0.2), color, mark or str(i), soft=bool(mark))
            tx = (MARGIN + pad) if rtl else (MARGIN + pad + 10)
            ty = y + (3 if box else 0)
            if name:
                content_dir()
                pdf.set_xy(tx, ty)
                pdf.set_font(body_font, "B", 9.6)
                pdf.set_text_color(*INK)
                pdf.multi_cell(tw, lh, name, align="R" if rtl else "L")
                ty = pdf.get_y() + 0.8
            content_dir()
            pdf.set_xy(tx, ty)
            pdf.set_font(body_font, "", 9.2)
            pdf.set_text_color(*(MUTED if name else INK2))
            pdf.multi_cell(tw, lh, item, align="R" if rtl else "L")
            pdf.set_y(y + row_h)

    def first_rows_height(items, extra=6, names=None):
        h = 0
        for i, it in enumerate(items[:2]):
            h += max(text_height(CONTENT_W - 16, it, 9.2), 6.4) + extra
            if names:
                h += text_height(CONTENT_W - 16, names[i], 9.6, "B") + 1
        return h + 14

    conds = result.get("possible_conditions", []) or []
    if conds:
        heading(3, "Possible causes", SKY, "  — for education only, not a diagnosis",
                need=first_rows_height([c.get("reason") or "" for c in conds], 6, [c.get("name", "") for c in conds]))
        rows([c.get("reason") or "" for c in conds], SKY, box=True, title_items=[c.get("name", "") for c in conds])
        pdf.ln(4)
    if result.get("recommended_next_steps"):
        heading(4, "Recommended next steps", GREEN, need=first_rows_height(result["recommended_next_steps"], 3))
        rows(result["recommended_next_steps"], GREEN)
        pdf.ln(4)
    if result.get("questions_for_doctor"):
        heading(5, "Questions to ask your doctor", BRAND, need=first_rows_height(result["questions_for_doctor"], 3))
        rows(result["questions_for_doctor"], SKY)
        pdf.ln(4)

    # ---- warning signs (red panel) ---------------------------------------------------------------------
    warns = result.get("warning_signs", []) or []
    if warns:
        heading(6, "Warning signs — get help immediately", RED, need=40)
        hs = [max(text_height(CONTENT_W - 22, w, 9.2), 6.4) + 2.6 for w in warns]
        total = sum(hs) + 8
        ensure(total)
        y = pdf.get_y()
        pdf.set_fill_color(*_tint(RED, .07))
        pdf.set_draw_color(*_tint(RED, .35))
        pdf.rect(MARGIN, y, CONTENT_W, total, style="DF", round_corners=True, corner_radius=3)
        pdf.set_y(y + 4)
        for w, h in zip(warns, hs):
            yy = pdf.get_y()
            bx = (PAGE_W - MARGIN - 5 - 6.4) if rtl else (MARGIN + 5)
            badge(bx, yy, RED, "!")
            content_dir()
            pdf.set_xy((MARGIN + 5) if rtl else (MARGIN + 15), yy)
            pdf.set_font(body_font, "", 9.2)
            pdf.set_text_color(*INK)
            pdf.multi_cell(CONTENT_W - 22, lh, w, align="R" if rtl else "L")
            pdf.set_y(yy + h)
        pdf.set_y(y + total + 4)

    # ---- disclaimer card ----------------------------------------------------------------------------------
    pdf.ltr()
    pdf.set_font("poppinsm", "", 8.2)
    dh = pdf.multi_cell(CONTENT_W - 22, 4.4, DISCLAIMER_FULL, dry_run=True, output="HEIGHT", align="L") + 11
    ensure(dh)
    y = pdf.get_y()
    pdf.set_fill_color(*_tint(AMBER, .08))
    pdf.set_draw_color(*_tint(AMBER, .4))
    pdf.rect(MARGIN, y, CONTENT_W, dh, style="DF", round_corners=True, corner_radius=3)
    badge(MARGIN + 5, y + 3.6, AMBER, "i")
    pdf.ltr()
    pdf.set_xy(MARGIN + 15, y + 3.4)
    pdf.set_font("poppins", "B", 9)
    pdf.set_text_color(146, 64, 14)
    pdf.cell(0, 5, "Important medical disclaimer")
    pdf.set_xy(MARGIN + 15, y + 9)
    pdf.set_font("poppinsm", "", 8.2)
    pdf.set_text_color(120, 53, 15)
    pdf.multi_cell(CONTENT_W - 22, 4.4, DISCLAIMER_FULL, align="L")
    pdf.set_y(y + dh + 4)
    if pdf.get_y() > pdf.h - pdf.b_margin - 5:      # no orphan page just for the credit line
        return bytes(pdf.output())
    pdf.set_font("poppinsm", "", 7)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 4, f"Generated by {APP_NAME} with {model} · {meta['date']} · Not a medical device", align="C")

    return bytes(pdf.output())


# ============================================================================
# VOICE — voice symptom input (OpenAI speech-to-text)
# ============================================================================
def transcribe_audio(api_key: str, audio_bytes: bytes, filename: str = "symptoms.wav") -> str:
    """Return the text spoken in the recording ('' if nothing was heard)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=60)
    buffer = io.BytesIO(audio_bytes)
    buffer.name = filename                       # the API uses the extension to detect format
    result = client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=buffer)
    return (result.text or "").strip()


# ============================================================================
# STYLES — design system (CSS), light & dark themes, animations
# ============================================================================
LIGHT = {
    "bg": "#F5F6FB", "bg2": "#EEF0FA", "surface": "#FFFFFF", "surface2": "#F7F8FC",
    "ink": "#0E1325", "ink2": "#2B3350", "muted": "#66708A", "line": "#E4E7F1",
    "brand": "#4F46E5", "brand2": "#7C3AED", "accent": "#0EA5A4",
    "glow": "rgba(79,70,229,.18)", "shadow": "0 1px 2px rgba(16,24,40,.04), 0 8px 24px -8px rgba(16,24,40,.10)",
    "shadow_lg": "0 24px 60px -18px rgba(49,46,129,.28)", "input": "#FFFFFF", "glass": "rgba(255,255,255,.72)",
    "grid": "rgba(79,70,229,.06)", "chip": "#EEF0FF",
}
DARK = {
    "bg": "#080B16", "bg2": "#0D1222", "surface": "#11172A", "surface2": "#161D34",
    "ink": "#EEF1FA", "ink2": "#C9CFE2", "muted": "#8D96B2", "line": "#232C47",
    "brand": "#818CF8", "brand2": "#A78BFA", "accent": "#2DD4BF",
    "glow": "rgba(129,140,248,.22)", "shadow": "0 1px 2px rgba(0,0,0,.4), 0 10px 30px -10px rgba(0,0,0,.6)",
    "shadow_lg": "0 30px 70px -20px rgba(0,0,0,.75)", "input": "#0E1427", "glass": "rgba(17,23,42,.72)",
    "grid": "rgba(129,140,248,.07)", "chip": "#1B2240",
}


def theme_vars(dark: bool) -> str:
    t = DARK if dark else LIGHT
    return ":root{" + "".join(f"--{k}:{v};" for k, v in t.items()) + "}"


FONTS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
@property --n{syntax:'<integer>';initial-value:0;inherits:false}
"""

BASE_CSS = r"""
/* ---------- foundations ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], button, input, textarea, select,
[data-testid="stMarkdownContainer"], [data-baseweb] { font-family:'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif !important; }
code, pre, .mono { font-family:'JetBrains Mono', ui-monospace, monospace !important; }
.stApp { background: var(--bg); color: var(--ink); transition: background .5s ease, color .5s ease; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 5rem; position:relative; z-index:1; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] [data-testid="stToolbarActions"], .stAppDeployButton, [data-testid="stMainMenu"], footer, #MainMenu, [data-testid="stStatusWidget"] { visibility:hidden; }
[data-testid="stDecoration"] { display:none; }
[data-testid="stHeaderActionElements"], [data-testid="stMarkdownContainer"] h1 > a, [data-testid="stMarkdownContainer"] h2 > a, [data-testid="stMarkdownContainer"] h3 > a { display:none !important; }
h1,h2,h3,h4,h5,h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: var(--ink) !important; letter-spacing: -.02em; font-family:'Plus Jakarta Sans', sans-serif !important; }
p, li, label, span, div { color: inherit; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--ink2); line-height: 1.7; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
a { color: var(--brand) !important; }
hr { border-color: var(--line) !important; }
::selection { background: var(--brand); color: #fff; }


/* ---------- shared ambient background (Connect screen AND main app) ---------- */
.gate-bg { position: fixed; inset:0; z-index:0; overflow:hidden; pointer-events:none; background: var(--bg); }
.gate-bg .b { position:absolute; border-radius:50%; filter: blur(70px); opacity:.55; animation: blob 16s ease-in-out infinite; }
.gate-bg .b1 { width: 520px; height:520px; background: var(--brand); top:-160px; left:-120px; opacity:.32; }
.gate-bg .b2 { width: 460px; height:460px; background: var(--brand2); bottom:-180px; right:-80px; animation-delay: -5s; opacity:.30; }
.gate-bg .b3 { width: 340px; height:340px; background: #0EA5A4; top: 40%; left: 45%; animation-delay: -9s; opacity:.18; }
.gate-bg .grid-l { position:absolute; inset:0; background: linear-gradient(var(--grid) 1px, transparent 1px) 0 0/48px 48px, linear-gradient(90deg, var(--grid) 1px, transparent 1px) 0 0/48px 48px; mask-image: radial-gradient(ellipse at center, #000 30%, transparent 75%); }
[data-testid="stAppViewContainer"] > section, [data-testid="stMain"] { position:relative; z-index:1; }
[data-testid="stSidebar"] { z-index: 5 !important; }
[data-testid="stHeader"] { z-index: 4 !important; }

/* ---------- shared frosted-glass surfaces (same look as the Connect card) ---------- */
[data-testid="stSidebar"] { background: var(--glass) !important; backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%); border-right: 1px solid var(--line); }
[data-testid="stChatMessage"]:has(:is([data-testid="stChatMessageAvatarAssistant"],[data-testid="stChatMessageAvatarCustom"])) [data-testid="stChatMessageContent"],
[data-testid="stMetric"], .urg, [class*="st-key-stream"], .li-card, [data-testid="stTabs"] [role="tablist"], [data-testid="stChatInput"],
[data-testid="stExpander"] details, .w-chips span, .stepper .sp:not(.on), [class*="st-key-card"], .h-card, .c-card, .flow .fs, .empty,
.stButton > button:not([kind="primary"]):not([kind="tertiary"]), .stDownloadButton > button:not([kind="primary"]), [data-testid="stPopover"] button {
  background-color: var(--glass) !important; backdrop-filter: blur(18px) saturate(140%); -webkit-backdrop-filter: blur(18px) saturate(140%); }
[data-testid="stTabs"] [role="tablist"] [data-testid="stTab"][aria-selected="true"] { background: var(--surface) !important; }
/* the welcome logo + assistant avatar use the same key-card gradient & glow */
.welcome .wl, .gc-key { background: linear-gradient(135deg, var(--brand), var(--brand2)); box-shadow: 0 16px 34px -14px var(--brand); }

/* ---------- keyframes ---------- */
@keyframes fadeUp { from { opacity:0; transform: translateY(18px);} to {opacity:1; transform:none;} }
@keyframes fadeIn { from { opacity:0;} to {opacity:1;} }
@keyframes floaty { 0%,100% { transform: translateY(0);} 50% { transform: translateY(-10px);} }
@keyframes beat { 0%,100% {transform: scale(1);} 14% {transform: scale(1.12);} 28% {transform: scale(1);} 42% {transform: scale(1.08);} 70% {transform: scale(1);} }
@keyframes ripple { 0% { transform: scale(.75); opacity:.55;} 100% { transform: scale(1.75); opacity:0;} }
@keyframes ecg { from { stroke-dashoffset: 1000; } to { stroke-dashoffset: 0; } }
@keyframes shine { to { background-position: 200% center; } }
@keyframes blob { 0%,100% { transform: translate(0,0) scale(1);} 33% { transform: translate(40px,-30px) scale(1.08);} 66% { transform: translate(-30px,25px) scale(.95);} }
@keyframes count { from { --n: 0; } }
@keyframes spinSlow { to { transform: rotate(360deg);} }
@keyframes pulseRed { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,.55), var(--shadow);} 50% { box-shadow: 0 0 0 14px rgba(239,68,68,0), var(--shadow);} }
@keyframes needle { from { transform: rotate(-90deg);} to { transform: rotate(var(--angle));} }
@keyframes arcdraw { from { stroke-dashoffset: 300;} to { stroke-dashoffset: 0;} }
@keyframes shake { 0%,100% {transform: rotate(0);} 20% {transform: rotate(-10deg);} 40% {transform: rotate(10deg);} 60% {transform: rotate(-6deg);} 80% {transform: rotate(6deg);} }
@keyframes borderflow { to { background-position: 300% 0; } }
@keyframes dots { 0%,20% { opacity:.2;} 50% { opacity:1;} 100% {opacity:.2;} }
@keyframes drawCheck { to { stroke-dashoffset: 0; } }
@keyframes popIn { 0% { transform: scale(.6); opacity:0;} 60% { transform: scale(1.08); opacity:1;} 100% { transform: scale(1);} }
@keyframes slideIn { from { opacity:0; transform: translateX(-16px);} to { opacity:1; transform:none;} }

.fade-up { animation: fadeUp .7s cubic-bezier(.2,.7,.2,1) both; }
.d1 { animation-delay: .08s; } .d2 { animation-delay: .16s; } .d3 { animation-delay: .24s; } .d4 { animation-delay: .32s; } .d5 { animation-delay: .40s; } .d6 { animation-delay: .48s; }
.ic { display:inline-block; vertical-align: middle; flex-shrink:0; }

/* ---------- inputs ---------- */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, .stTextInput label, .stSelectbox label { color: var(--ink2) !important; font-weight: 600 !important; font-size: .86rem !important; }
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"], [data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] div:has(> input[role="combobox"]), [data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]) {
  background: var(--input) !important; border: 1.5px solid var(--line) !important; border-radius: 12px !important;
  transition: border-color .2s ease, box-shadow .2s ease; }
[data-testid="stTextInputRootElement"]:hover, [data-testid="stTextAreaRootElement"]:hover, [data-testid="stSelectbox"] div:has(> input[role="combobox"]):hover, [data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]):hover { border-color: color-mix(in srgb, var(--brand) 45%, var(--line)) !important; }
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stTextAreaRootElement"]:focus-within, [data-testid="stSelectbox"] div:has(> input[role="combobox"]):focus-within, [data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]):focus-within {
  border-color: var(--brand) !important; box-shadow: 0 0 0 4px var(--glow) !important; }
[data-testid="stTextInputRootElement"] *, [data-testid="stTextAreaRootElement"] *, [data-testid="stSelectbox"] input, [data-testid="stSelectbox"] div:has(> input[role="combobox"]) *, [data-testid="stMultiSelect"] input { background-color: transparent !important; }
input, textarea, [data-testid="stSelectbox"] div:has(> input[role="combobox"]) { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg { color: var(--muted) !important; }
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] [role="listbox"], [role="listbox"] { background: var(--surface) !important; border-radius: 12px !important; }
[role="listbox"] { border: 1px solid var(--line) !important; box-shadow: var(--shadow_lg) !important; }
[role="option"] { color: var(--ink) !important; border-radius: 8px; }
[role="option"]:hover, [role="option"][aria-selected="true"], [role="option"][data-focused="true"] { background: var(--chip) !important; }
[data-testid="stAudioInput"] * { color: var(--muted); }
input::placeholder, textarea::placeholder { color: var(--muted) !important; opacity: .7 !important; -webkit-text-fill-color: var(--muted) !important; }
[data-baseweb="select"] svg { color: var(--muted) !important; fill: var(--muted) !important; }
[data-baseweb="tag"] { background: linear-gradient(135deg, var(--brand), var(--brand2)) !important; border-radius: 999px !important; border: none !important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: #fff !important; fill:#fff !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"], [role="listbox"] { background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 12px !important; box-shadow: var(--shadow_lg) !important; }
[data-baseweb="popover"] li, [role="option"] { color: var(--ink) !important; background: transparent !important; }
[data-baseweb="popover"] li:hover, [role="option"]:hover, [role="option"][aria-selected="true"] { background: var(--chip) !important; }
[data-testid="stTextInputRootElement"] button { background: transparent !important; color: var(--muted) !important; }
[data-testid="InputInstructions"] { display:none; }

/* slider */
[data-testid="stSlider"] [role="slider"] { background: #fff !important; border: 3px solid var(--brand) !important; box-shadow: 0 4px 14px var(--glow) !important; width: 22px !important; height: 22px !important; }
[data-testid="stSliderThumbValue"] { color: var(--brand) !important; font-weight: 800 !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; }

/* buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 12px !important; font-weight: 700 !important; padding: .62rem 1.15rem !important; transition: all .25s cubic-bezier(.2,.7,.2,1) !important; border: 1.5px solid var(--line) !important; background: var(--surface) !important; color: var(--ink) !important; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); border-color: var(--brand) !important; color: var(--brand) !important; box-shadow: 0 10px 24px -10px var(--glow) !important; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(120deg, var(--brand), var(--brand2), var(--brand)) !important; background-size: 200% auto !important; border: none !important; color: #fff !important;
  box-shadow: 0 12px 30px -12px var(--brand) !important; }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background-position: right center !important; color:#fff !important; transform: translateY(-2px) scale(1.01); box-shadow: 0 18px 38px -12px var(--brand) !important; }
.stButton > button p, .stDownloadButton > button p { font-weight: 700 !important; color: inherit !important; }

/* popover buttons (e.g. "Clear history", "View raw JSON") */
[data-testid="stPopover"] button, [data-testid="stPopoverButton"] { border-radius: 12px !important; font-weight: 700 !important; border: 1.5px solid var(--line) !important; background: var(--surface) !important; color: var(--ink) !important; transition: all .25s ease !important; }
[data-testid="stPopover"] button:hover, [data-testid="stPopoverButton"]:hover { border-color: var(--brand) !important; color: var(--brand) !important; }
[data-testid="stPopover"] button p, [data-testid="stPopoverButton"] p { color: inherit !important; font-weight:700 !important; }
[data-testid="stPopoverBody"] { background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 14px !important; color: var(--ink); }
[data-testid="stJson"] { background: transparent !important; }

/* toggles, radios, checkboxes */
[data-testid="stToggle"] label p, [data-testid="stCheckbox"] label p { color: var(--ink2) !important; }

/* ---------- containers used as cards (st.container(key="card_*")) ---------- */
[class*="st-key-card"] { background: var(--surface); border: 1px solid var(--line); border-radius: 20px; padding: 1.35rem 1.35rem 1.1rem; box-shadow: var(--shadow);
  animation: fadeUp .7s cubic-bezier(.2,.7,.2,1) both; transition: box-shadow .3s ease, transform .3s ease, border-color .3s ease; }
[class*="st-key-card"]:hover { box-shadow: var(--shadow_lg); border-color: color-mix(in srgb, var(--brand) 25%, var(--line)); }
.card-head { display:flex; align-items:center; gap:.7rem; margin-bottom:.35rem; }
.card-head .ib { width:38px; height:38px; border-radius:11px; display:grid; place-items:center; color:#fff; background: linear-gradient(135deg, var(--brand), var(--brand2)); box-shadow: 0 8px 18px -8px var(--brand); }
.card-head .ib.teal { background: linear-gradient(135deg, #0EA5A4, #22C55E); box-shadow: 0 8px 18px -8px #0EA5A4; }
.card-head .ib.rose { background: linear-gradient(135deg, #F43F5E, #F97316); box-shadow: 0 8px 18px -8px #F43F5E; }
.card-head .ib.sky { background: linear-gradient(135deg, #0EA5E9, #6366F1); box-shadow: 0 8px 18px -8px #0EA5E9; }
.card-head h4 { margin:0 !important; padding:0 !important; font-size: 1.02rem !important; font-weight: 800 !important; }
.card-head small { display:block; color: var(--muted); font-size: .78rem; font-weight:500; margin-top:1px; }

/* ---------- metrics ---------- */
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.1rem; box-shadow: var(--shadow); animation: fadeUp .6s both; position:relative; overflow:hidden; }
[data-testid="stMetric"]::after { content:""; position:absolute; left:0; top:0; height:3px; width:100%; background: linear-gradient(90deg, var(--brand), var(--brand2), var(--accent)); }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 600 !important; font-size: .8rem !important; text-transform: uppercase; letter-spacing: .06em; }
[data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800 !important; letter-spacing: -.02em; }

/* ---------- tabs ---------- */
[data-testid="stTabs"] [role="tablist"] { gap: .35rem; background: var(--surface2); padding: .35rem; border-radius: 14px; border: 1px solid var(--line); overflow-x: auto; box-shadow: none !important; }
[data-testid="stTabs"] [role="tablist"]::after, [data-testid="stTabs"] [role="tablist"]::before { display:none !important; }
[data-testid="stTab"], [data-testid="stTabs"] [role="tab"] { border-radius: 10px !important; padding: .5rem .95rem !important; background: transparent !important; color: var(--muted) !important; font-weight: 700 !important; transition: all .2s ease; border: none !important; box-shadow:none; white-space: nowrap; }
[data-testid="stTab"] p, [data-testid="stTab"] span { font-weight: 700 !important; color: inherit !important; font-size: .88rem !important; }
[data-testid="stTab"]:hover { color: var(--ink) !important; }
[data-testid="stTab"][aria-selected="true"] { background: var(--surface) !important; color: var(--brand) !important; box-shadow: var(--shadow) !important; }
[data-testid="stTab"] > div:last-child:empty { display:none; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: 1.1rem !important; animation: fadeIn .45s ease both; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none !important; }

/* ---------- expanders ---------- */
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--line) !important; border-radius: 14px !important; overflow:hidden; transition: border-color .25s ease, box-shadow .25s ease; animation: fadeUp .6s both; }
[data-testid="stExpander"] details:hover { border-color: color-mix(in srgb, var(--brand) 35%, var(--line)) !important; box-shadow: var(--shadow); }
[data-testid="stExpander"] summary { color: var(--ink) !important; font-weight: 700; }
[data-testid="stExpander"] summary:hover { color: var(--brand) !important; }
[data-testid="stExpander"] summary svg { color: var(--brand) !important; }

/* ---------- alerts ---------- */
[data-testid="stAlert"] > div, [data-testid="stAlertContainer"] { border-radius: 14px !important; border: 1px solid transparent; animation: fadeUp .55s both; }
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentInfo"] p { color: inherit; }
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color: inherit; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { background: transparent; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: .88rem; }
[data-testid="stSidebarHeader"] { padding-bottom: 0 !important; }
.sb-brand { display:flex; align-items:center; gap:.75rem; margin: -.4rem 0 .9rem; }
.sb-logo { width:44px; height:44px; border-radius:14px; display:grid; place-items:center; color:#fff; background: linear-gradient(135deg, var(--brand), var(--brand2)); box-shadow: 0 10px 24px -10px var(--brand); position:relative; }
.sb-logo svg { animation: beat 2.4s ease-in-out infinite; }
.sb-name { font-weight: 800; font-size: 1.15rem; color: var(--ink); letter-spacing:-.02em; line-height:1.1; }
.sb-ver { font-size:.7rem; font-weight:700; color: var(--brand); background: var(--chip); padding: 2px 8px; border-radius: 999px; }
.sb-desc { color: var(--muted); font-size: .84rem; line-height: 1.55; margin-bottom: .9rem; }
.sb-label { font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color: var(--muted); font-weight:800; margin: 1.1rem 0 .45rem; display:flex; align-items:center; gap:.4rem; }
.sb-disc { border-radius: 14px; padding: .8rem .9rem; background: linear-gradient(135deg, rgba(245,158,11,.12), rgba(239,68,68,.08)); border: 1px solid rgba(245,158,11,.35); color: var(--ink2); font-size: .8rem; line-height:1.5; }
.sb-disc b { color: #D97706; display:flex; align-items:center; gap:.35rem; margin-bottom:.2rem; font-size:.78rem; letter-spacing:.04em; text-transform:uppercase; }
.sb-conn { display:flex; align-items:center; gap:.55rem; padding:.6rem .75rem; border-radius:12px; background: var(--surface2); border:1px solid var(--line); font-size:.8rem; color: var(--ink2); }
.sb-conn .dot { width:9px; height:9px; border-radius:50%; background:#10B981; box-shadow: 0 0 0 0 rgba(16,185,129,.6); animation: pulseGreen 2s infinite; }
@keyframes pulseGreen { 0% { box-shadow: 0 0 0 0 rgba(16,185,129,.55);} 70% { box-shadow: 0 0 0 8px rgba(16,185,129,0);} 100% { box-shadow:0 0 0 0 rgba(16,185,129,0);} }

/* nav radio as pills */
[class*="st-key-nav"] [role="radiogroup"] { gap: .25rem; }
[class*="st-key-nav"] [role="radiogroup"] label { width:100%; padding: .55rem .7rem; border-radius: 12px; transition: all .2s ease; margin:0 !important; border: 1px solid transparent; }
[class*="st-key-nav"] [role="radiogroup"] label:hover { background: var(--surface2); }
[class*="st-key-nav"] [role="radiogroup"] label:has(input:checked) { background: linear-gradient(120deg, color-mix(in srgb, var(--brand) 14%, transparent), color-mix(in srgb, var(--brand2) 10%, transparent)); border-color: color-mix(in srgb, var(--brand) 30%, transparent); }
[class*="st-key-nav"] [role="radiogroup"] label:has(input:checked) p { color: var(--brand) !important; }
[class*="st-key-nav"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display:none !important; }
[class*="st-key-nav"] [role="radiogroup"], [class*="st-key-nav"] [role="radiogroup"] > div { width:100%; }
[class*="st-key-nav"] [role="radiogroup"] label p { font-weight: 700 !important; font-size: .92rem !important; color: var(--ink2) !important; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: .85rem 1rem; animation: fadeUp .5s both; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarCustom"] { background: linear-gradient(135deg, var(--brand), var(--brand2)) !important; color:#fff !important; }

/* audio input */
[data-testid="stAudioInput"] > div { background: var(--surface2) !important; border: 1.5px dashed color-mix(in srgb, var(--brand) 40%, var(--line)) !important; border-radius: 14px !important; }

/* tables / dataframes */
[data-testid="stTable"] table { border-radius: 14px; overflow:hidden; border: 1px solid var(--line); }
[data-testid="stTable"] th { background: var(--surface2) !important; color: var(--ink) !important; font-weight: 800 !important; }
[data-testid="stTable"] td { color: var(--ink2) !important; background: var(--surface) !important; border-color: var(--line) !important; }

/* code blocks */
[data-testid="stCode"] pre, .stCodeBlock pre { background: var(--surface2) !important; border: 1px solid var(--line); border-radius: 14px !important; }
[data-testid="stCode"] code { color: var(--ink2) !important; }

/* spinner / status */
[data-testid="stStatusWidget"], [data-testid="stExpander"] [data-testid="stStatus"] { color: var(--ink) !important; }
[data-testid="stSpinner"] p { color: var(--ink2) !important; }

/* ---------- reusable HTML components ---------- */
.eyebrow { display:inline-flex; align-items:center; gap:.45rem; padding:.38rem .8rem .38rem .45rem; border-radius:999px; background: var(--glass); border:1px solid var(--line); color: var(--ink2); font-size:.78rem; font-weight:700; backdrop-filter: blur(10px); box-shadow: var(--shadow); }
.eyebrow .pill { background: linear-gradient(135deg, var(--brand), var(--brand2)); color:#fff; border-radius:999px; padding: .12rem .55rem; font-size:.68rem; letter-spacing:.06em; }
.grad-text { background: linear-gradient(100deg, var(--brand) 0%, var(--brand2) 35%, var(--accent) 60%, var(--brand) 100%); background-size: 200% auto; -webkit-background-clip: text; background-clip:text; color: transparent !important; -webkit-text-fill-color: transparent; animation: shine 6s linear infinite; }
.section-title { display:flex; align-items:end; justify-content:space-between; gap:1rem; margin: 2.1rem 0 1rem; flex-wrap: wrap; }
.section-title h2 { margin:0 !important; padding:0 !important; font-size: 1.55rem !important; font-weight: 800 !important; }
.section-title p { margin:.25rem 0 0; color: var(--muted); font-size:.92rem; }
.kicker { font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color: var(--brand); margin-bottom:.35rem; display:flex; align-items:center; gap:.4rem; }

/* hero */
.hero { position:relative; display:grid; grid-template-columns: 1.5fr .85fr; gap: 2rem; align-items:center; padding: 2.4rem 2.4rem; border-radius: 28px; overflow:hidden;
  background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow_lg); animation: fadeUp .8s both; }
.hero::before { content:""; position:absolute; width: 520px; height:520px; right:-140px; top:-200px; border-radius:50%; background: radial-gradient(circle, color-mix(in srgb, var(--brand2) 28%, transparent), transparent 65%); animation: blob 14s ease-in-out infinite; }
.hero::after { content:""; position:absolute; width: 420px; height:420px; left:-160px; bottom:-220px; border-radius:50%; background: radial-gradient(circle, rgba(14,165,164,.20), transparent 65%); animation: blob 18s ease-in-out infinite reverse; }
.hero > * { position:relative; z-index:1; }
.hero h1 { font-size: clamp(1.9rem, 3vw, 2.65rem) !important; line-height: 1.06 !important; font-weight: 800 !important; margin: .9rem 0 .8rem !important; padding:0 !important; letter-spacing: -.035em !important; }
.hero p.lead { color: var(--muted); font-size: 1.02rem; line-height: 1.7; max-width: 560px; margin:0; }
.hero-stats { display:flex; gap: 1.4rem; margin-top: 1.6rem; flex-wrap:wrap; }
.hero-stats .st { min-width: 0; }
.hero-stats .num { font-size: 1.9rem; font-weight: 800; color: var(--ink); letter-spacing:-.03em; line-height:1; }
.hero-stats .num.count { animation: count 2.2s cubic-bezier(.2,.7,.2,1) forwards; counter-reset: n var(--n); }
.hero-stats .num.count::before { content: counter(n); }
.hero-stats .lbl { font-size:.74rem; color: var(--muted); font-weight:600; margin-top:.35rem; }
.hero-stats .sep { width:1px; background: var(--line); }

.orb-wrap { position:relative; height: 300px; display:grid; place-items:center; }
.orb-ring { position:absolute; width: 190px; height:190px; border-radius:50%; border: 2px solid color-mix(in srgb, var(--brand) 40%, transparent); animation: ripple 3s ease-out infinite; }
.orb-ring.r2 { animation-delay: 1s; } .orb-ring.r3 { animation-delay: 2s; }
.orb-dash { position:absolute; width: 240px; height:240px; border-radius:50%; border: 1.5px dashed color-mix(in srgb, var(--brand) 30%, transparent); animation: spinSlow 40s linear infinite; }
.orb { width: 170px; height:170px; border-radius:50%; display:grid; place-items:center; color:#fff;
  background: radial-gradient(circle at 30% 25%, color-mix(in srgb, #fff 28%, var(--brand2)), var(--brand) 55%, color-mix(in srgb, var(--brand) 70%, #000) 100%);
  box-shadow: 0 30px 60px -18px var(--brand), inset 0 -10px 30px rgba(0,0,0,.18), inset 0 10px 30px rgba(255,255,255,.18); }
.orb svg { animation: beat 2.2s ease-in-out infinite; filter: drop-shadow(0 6px 12px rgba(0,0,0,.2)); }
.float-badge { position:absolute; display:flex; align-items:center; gap:.4rem; padding: .5rem .75rem; border-radius: 12px; background: var(--glass); backdrop-filter: blur(12px); border:1px solid var(--line); box-shadow: var(--shadow); font-weight:700; font-size:.78rem; color: var(--ink); animation: floaty 5s ease-in-out infinite; }
.float-badge .b-ic { width:24px; height:24px; border-radius:8px; display:grid; place-items:center; color:#fff; }
.fb1 { top: 10px; right: 0; } .fb2 { bottom: 14px; right: -6px; animation-delay: 1.4s; } .fb3 { top: 34px; left: -14px; animation-delay: .7s; }
.ecg-line { width:100%; height: 46px; margin-top: 1.2rem; }
.ecg-line path { stroke-dasharray: 1000; stroke-dashoffset: 1000; animation: ecg 3.2s linear infinite; }

/* disclaimer ribbon */
.disc-ribbon { display:flex; gap: .9rem; align-items:flex-start; margin: 1.2rem 0 0; padding: .95rem 1.1rem; border-radius: 16px;
  background: linear-gradient(120deg, rgba(245,158,11,.12), rgba(239,68,68,.07)); border: 1px solid rgba(245,158,11,.38); animation: fadeUp .8s .15s both; }
.disc-ribbon .di { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; background: #F59E0B; color:#fff; flex-shrink:0; box-shadow: 0 8px 18px -8px #F59E0B; }
.disc-ribbon b { color: var(--ink); font-size: .9rem; }
.disc-ribbon span { color: var(--ink2); font-size: .86rem; line-height:1.55; display:block; margin-top: 2px; }

/* severity meter */
.sev { display:flex; align-items:center; gap:.7rem; margin: -.3rem 0 .6rem; }
.sev-bar { flex:1; height: 8px; border-radius: 999px; background: var(--surface2); overflow:hidden; border:1px solid var(--line); }
.sev-bar i { display:block; height:100%; border-radius:999px; background: linear-gradient(90deg, #10B981, #F59E0B, #F97316, #EF4444); background-size: 1000% 100%; transition: width .5s cubic-bezier(.2,.7,.2,1); }
.sev-chip { font-size:.75rem; font-weight:800; padding: .22rem .6rem; border-radius:999px; color:#fff; white-space:nowrap; transition: background .4s ease; }

/* steps strip */
.steps { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; }
.step { display:flex; align-items:center; gap:.45rem; font-size:.78rem; font-weight:700; color: var(--muted); }
.step .n { width:24px; height:24px; border-radius:50%; display:grid; place-items:center; font-size:.72rem; background: var(--surface2); border:1px solid var(--line); color: var(--muted); }
.step.on { color: var(--ink); } .step.on .n { background: linear-gradient(135deg, var(--brand), var(--brand2)); color:#fff; border:none; box-shadow: 0 6px 14px -6px var(--brand); }
.step-line { width: 26px; height:2px; background: var(--line); border-radius: 2px; }

/* urgency banner */
.urg { position:relative; display:grid; grid-template-columns: 1fr 260px; gap:1.5rem; align-items:center; padding: 1.6rem 1.8rem; border-radius: 24px; overflow:hidden;
  background: var(--surface); border: 1.5px solid var(--uc); box-shadow: var(--shadow_lg); animation: fadeUp .7s both; }
.urg::before { content:""; position:absolute; inset:0; background: radial-gradient(600px 220px at 0% 0%, var(--us), transparent 70%); pointer-events:none; }
.urg.emergency { animation: fadeUp .7s both, pulseRed 1.8s ease-in-out infinite; }
.urg > * { position:relative; }
.urg-level { display:inline-flex; align-items:center; gap:.55rem; padding: .45rem .9rem .45rem .5rem; border-radius:999px; background: var(--uc); color:#fff; font-weight:800; letter-spacing:.08em; font-size:.8rem; box-shadow: 0 10px 24px -10px var(--uc); }
.urg-level .ui { width:26px; height:26px; border-radius:50%; background: rgba(255,255,255,.22); display:grid; place-items:center; }
.urg.emergency .urg-level .ui svg { animation: shake 1s ease-in-out infinite; }
.urg h3 { font-size: 1.6rem !important; font-weight: 800 !important; margin: .8rem 0 .35rem !important; padding:0 !important; }
.urg p { color: var(--ink2); margin:0; line-height:1.6; text-align: start; }
.urg p[dir] { unicode-bidi: plaintext; }
.urg .meta { display:flex; gap:.5rem; flex-wrap:wrap; margin-top: 1rem; }
.tag { display:inline-flex; align-items:center; gap:.35rem; font-size:.74rem; font-weight:700; padding: .3rem .65rem; border-radius:999px; background: var(--surface2); border:1px solid var(--line); color: var(--ink2); }
.tag.cache { background: rgba(16,185,129,.12); border-color: rgba(16,185,129,.35); color: #059669; }
.tag.live { background: var(--chip); border-color: color-mix(in srgb, var(--brand) 30%, transparent); color: var(--brand); }

/* gauge */
.gauge { width: 100%; max-width: 260px; }
.gauge .needle { transform-origin: 130px 130px; transform: rotate(var(--angle)); animation: needle 1.6s cubic-bezier(.3,1.4,.4,1) both .2s; }
.gauge .arc { stroke-dasharray: 300; stroke-dashoffset: 300; animation: arcdraw 1s ease-out forwards; }
.gauge .a2 { animation-delay: .15s; } .gauge .a3 { animation-delay: .3s; } .gauge .a4 { animation-delay: .45s; }
.gauge text { font-family: 'Plus Jakarta Sans', sans-serif; }

/* lists inside tabs */
.li-card { display:flex; gap:.85rem; align-items:flex-start; padding: .9rem 1rem; border-radius: 14px; background: var(--surface); border:1px solid var(--line); margin-bottom: .6rem; animation: slideIn .5s both; transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; }
.li-card:hover { transform: translateX(4px); border-color: color-mix(in srgb, var(--lc, var(--brand)) 45%, var(--line)); box-shadow: var(--shadow); }
.li-card .num { width:30px; height:30px; border-radius:10px; display:grid; place-items:center; flex-shrink:0; font-weight:800; font-size:.82rem; color:#fff; background: var(--lc, var(--brand)); }
.li-card .tx { color: var(--ink2); line-height: 1.55; font-size: .94rem; padding-top: 4px; }

/* streaming panel */
[class*="st-key-stream"] { position:relative; border-radius: 20px; padding: 1.3rem 1.5rem; background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow); }
[class*="st-key-stream"]::before { content:""; position:absolute; inset:-1px; border-radius: 21px; padding:1.5px; background: linear-gradient(90deg, var(--brand), var(--brand2), var(--accent), var(--brand)); background-size: 300% 100%; animation: borderflow 4s linear infinite;
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0); -webkit-mask-composite: xor; mask-composite: exclude; pointer-events:none; }
.stream-head { display:flex; align-items:center; gap:.6rem; margin-bottom:.6rem; font-weight:800; color: var(--ink); }
.stream-head .live-dot { display:inline-flex; gap:3px; }
.stream-head .live-dot i { width:6px; height:6px; border-radius:50%; background: var(--brand); animation: dots 1.2s infinite; }
.stream-head .live-dot i:nth-child(2) { animation-delay: .2s; } .stream-head .live-dot i:nth-child(3) { animation-delay: .4s; }

/* RTL zone for Urdu / Arabic answers */
[class*="st-key-rtl"] [role="tabpanel"] [data-testid="stMarkdownContainer"], [class*="st-key-rtl"] .li-card .tx, [class*="st-key-rtl"] [data-testid="stExpander"] summary { direction: rtl; text-align: right; }
[class*="st-key-rtl"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] { direction: ltr; text-align: left; }
[class*="st-key-rtl"] [role="tabpanel"] .stream-head { direction: ltr; }
[class*="st-key-rtl"] [role="tabpanel"] [data-testid="stMarkdownContainer"] p, [class*="st-key-rtl"] .li-card .tx { font-size: 1.04rem; line-height: 2; }

/* emergency call box */
.sos { display:flex; gap:1rem; align-items:center; padding: 1.1rem 1.3rem; border-radius: 18px; background: linear-gradient(120deg, #EF4444, #B91C1C); color:#fff; box-shadow: 0 20px 40px -18px #EF4444; margin-top: 1rem; animation: fadeUp .6s both; }
.sos .si { width:46px; height:46px; border-radius:14px; background: rgba(255,255,255,.18); display:grid; place-items:center; flex-shrink:0; }
.sos .si svg { animation: shake 1s infinite; }
.sos b { font-size: 1.05rem; } .sos span { opacity:.92; font-size:.88rem; display:block; margin-top:2px; }
.sos .nums { margin-left:auto; display:flex; gap:.4rem; flex-wrap:wrap; }
.sos .nums em { font-style:normal; font-weight:800; background: rgba(255,255,255,.18); padding: .35rem .7rem; border-radius: 10px; font-size:.82rem; }

/* history cards */
.h-card { display:grid; grid-template-columns: 6px 1fr auto; gap: 1rem; align-items:center; padding: 1rem 1.1rem 1rem 0; border-radius: 16px; background: var(--surface); border:1px solid var(--line); box-shadow: var(--shadow); overflow:hidden; animation: fadeUp .55s both; }
.h-card .strip { height: 100%; min-height: 64px; background: var(--uc); }
.h-card .t { font-weight: 800; color: var(--ink); font-size: .98rem; }
.h-card .s { color: var(--muted); font-size: .8rem; margin-top: .25rem; display:flex; gap:.9rem; flex-wrap:wrap; }
.h-card .chips { display:flex; gap:.35rem; flex-wrap:wrap; margin-top:.5rem; }
.h-card .chips span { font-size:.72rem; font-weight:700; padding: .18rem .55rem; border-radius:999px; background: var(--chip); color: var(--brand); }
.u-pill { font-size: .72rem; font-weight:800; letter-spacing:.06em; padding: .35rem .7rem; border-radius:999px; color:#fff; background: var(--uc); }

/* concept / pipeline cards */
.grid { display:grid; gap: 1rem; }
.g4 { grid-template-columns: repeat(4, minmax(0,1fr)); } .g3 { grid-template-columns: repeat(3, minmax(0,1fr)); } .g2 { grid-template-columns: repeat(2, minmax(0,1fr)); }
.c-card { position:relative; padding: 1.15rem 1.15rem 1.1rem; border-radius: 18px; background: var(--surface); border:1px solid var(--line); box-shadow: var(--shadow); animation: fadeUp .6s both; transition: transform .3s cubic-bezier(.2,.7,.2,1), box-shadow .3s ease, border-color .3s ease; overflow:hidden; }
.c-card:hover { transform: translateY(-5px); box-shadow: var(--shadow_lg); border-color: color-mix(in srgb, var(--brand) 35%, var(--line)); }
.c-card .ci { width:40px; height:40px; border-radius:12px; display:grid; place-items:center; color:#fff; background: linear-gradient(135deg, var(--brand), var(--brand2)); margin-bottom:.8rem; box-shadow: 0 8px 18px -8px var(--brand); }
.c-card h5 { margin:0 0 .3rem !important; padding:0 !important; font-size: .98rem !important; font-weight:800 !important; color: var(--ink) !important; }
.c-card p { margin:0; color: var(--muted); font-size: .84rem; line-height: 1.55; }
.c-card code { font-size: .74rem; background: var(--chip); color: var(--brand); padding: .12rem .45rem; border-radius: 6px; }
.c-card .file { position:absolute; top: 1rem; right: 1rem; font-size:.68rem; font-weight:700; color: var(--muted); font-family:'JetBrains Mono', monospace; }
.flow { display:flex; align-items:stretch; gap:.35rem; flex-wrap:nowrap; }
.flow .fs { flex:1; min-width: 0; padding: .9rem .45rem; border-radius: 16px; background: var(--surface); border:1px solid var(--line); text-align:center; animation: fadeUp .6s both; position:relative; }
.flow .fs .fi { width:38px; height:38px; margin: 0 auto .5rem; border-radius: 12px; display:grid; place-items:center; color: var(--brand); background: var(--chip); }
.flow .fs b { display:block; font-size: .8rem; color: var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.flow .fs small { color: var(--muted); font-size:.68rem; display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.flow .arrow { display:grid; place-items:center; color: var(--muted); }

/* chat bubbles for the messages demo */
.msg { display:flex; gap:.75rem; margin-bottom:.75rem; animation: fadeUp .5s both; }
.msg .role { flex-shrink:0; font-size: .66rem; font-weight:800; letter-spacing:.08em; padding: .3rem .55rem; border-radius: 8px; height: fit-content; margin-top: 3px; font-family:'JetBrains Mono', monospace; }
.msg .bubble { padding: .75rem .95rem; border-radius: 14px; background: var(--surface); border:1px solid var(--line); color: var(--ink2); font-size: .9rem; line-height: 1.55; }
.msg.system .role { background: rgba(100,116,139,.15); color: var(--muted); }
.msg.human .role { background: var(--chip); color: var(--brand); }
.msg.ai .role { background: rgba(14,165,164,.14); color: #0D9488; }
.msg.reply .role { background: linear-gradient(135deg, var(--brand), var(--brand2)); color: #fff; }
.msg.reply .bubble { border-color: color-mix(in srgb, var(--brand) 40%, var(--line)); box-shadow: 0 10px 30px -16px var(--brand); }

/* empty state */
.empty { text-align:center; padding: 3.5rem 1rem; border-radius: 24px; border: 1.5px dashed var(--line); background: var(--surface); animation: fadeUp .6s both; }
.empty .ei { width:64px; height:64px; margin: 0 auto 1rem; border-radius: 20px; display:grid; place-items:center; color: var(--brand); background: var(--chip); animation: floaty 4s ease-in-out infinite; }
.empty h4 { margin: 0 0 .35rem !important; font-weight:800 !important; }
.empty p { color: var(--muted); margin: 0; }

/* bench bars */
.bench { display:flex; flex-direction:column; gap:.7rem; }
.bench .row { display:grid; grid-template-columns: 120px 1fr 80px; gap:.8rem; align-items:center; font-size:.85rem; color: var(--ink2); font-weight:700; }
.bench .track { height: 14px; border-radius: 999px; background: var(--surface2); border:1px solid var(--line); overflow:hidden; }
.bench .fill { height:100%; border-radius:999px; animation: grow 1.2s cubic-bezier(.2,.7,.2,1) both; transform-origin: left; }
@keyframes grow { from { transform: scaleX(0);} to { transform: scaleX(1);} }


/* =================== CHAT ASSISTANT =================== */
[class*="st-key-chatwrap"] { max-width: 860px; margin: 0 auto; width: 100%; }
[class*="st-key-scroller"] { display:none !important; }
[data-testid="stChatMessageAvatarCustom"] span { color:#fff !important; }
.welcome { text-align:center; padding: .4rem 0 .6rem; animation: fadeUp .7s both; }
.welcome .wl { position:relative; width:74px; height:74px; margin: 0 auto 1.1rem; border-radius: 24px; display:grid; place-items:center; color:#fff;
  background: linear-gradient(135deg, var(--brand), var(--brand2)); box-shadow: 0 18px 40px -16px var(--brand); }
.welcome .wl::after { content:""; position:absolute; inset:-7px; border-radius: 28px; border: 2px solid color-mix(in srgb, var(--brand) 40%, transparent); animation: ripple 2.6s ease-out infinite; }
.welcome .wl svg { animation: beat 2.2s ease-in-out infinite; }
.welcome h1 { font-size: clamp(1.7rem, 3vw, 2.35rem) !important; font-weight: 800 !important; letter-spacing: -.035em !important; margin: 0 0 .5rem !important; padding: 0 !important; }
.welcome p { color: var(--muted); max-width: 580px; margin: 0 auto; line-height: 1.65; font-size: .98rem; }
.welcome p b { color: var(--ink); }
.w-chips { display:flex; gap:.45rem; justify-content:center; flex-wrap:wrap; margin-top: 1rem; }
.w-chips span { display:inline-flex; align-items:center; gap:.35rem; font-size:.74rem; font-weight:700; padding:.32rem .7rem; border-radius:999px; background: var(--surface); border:1px solid var(--line); color: var(--ink2); }
.w-chips .sos-chip { color:#DC2626; background: rgba(239,68,68,.08); border-color: rgba(239,68,68,.3); }
.stepper { display:flex; justify-content:center; gap:.35rem; margin: 1.1rem 0 1.3rem; flex-wrap:wrap; animation: fadeIn .6s both; }
.stepper .sp { display:inline-flex; align-items:center; gap:.35rem; font-size:.74rem; font-weight:700; padding:.34rem .75rem; border-radius:999px; background: var(--surface2); border:1px solid var(--line); color: var(--muted); transition: all .35s ease; }
.stepper .sp b { width:17px; height:17px; border-radius:50%; display:grid; place-items:center; font-size:.64rem; background: var(--line); color: var(--muted); }
.stepper .sp.done { color:#059669; background: rgba(16,185,129,.10); border-color: rgba(16,185,129,.3); }
.stepper .sp.on { color:#fff; background: linear-gradient(135deg, var(--brand), var(--brand2)); border-color: transparent; box-shadow: 0 8px 18px -8px var(--brand); }
.stepper .sp.on b { background: rgba(255,255,255,.25); color:#fff; }

/* chat bubbles */
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: .3rem 0 !important; gap: .75rem; animation: fadeUp .45s both; }
[data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarCustom"] { background: linear-gradient(135deg, var(--brand), var(--brand2)) !important; color:#fff !important; border-radius: 12px !important; box-shadow: 0 8px 18px -8px var(--brand); }
[data-testid="stChatMessageAvatarUser"] { background: var(--surface2) !important; color: var(--brand) !important; border: 1px solid var(--line); border-radius: 12px !important; }
[data-testid="stChatMessage"]:has(:is([data-testid="stChatMessageAvatarAssistant"],[data-testid="stChatMessageAvatarCustom"])) [data-testid="stChatMessageContent"] {
  background: var(--surface); border: 1px solid var(--line); border-radius: 6px 20px 20px 20px; padding: .95rem 1.15rem; box-shadow: var(--shadow); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  margin-left: auto !important; margin-right: 0 !important; width: fit-content; max-width: 80%; flex-grow: 0; background: linear-gradient(135deg, var(--brand), var(--brand2));
  border-radius: 20px 6px 20px 20px; padding: .7rem 1rem; box-shadow: 0 12px 26px -14px var(--brand); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p { color: #fff !important; margin: 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stButton > button {
  background: rgba(255,255,255,.16) !important; color: #fff !important; border: 1px solid rgba(255,255,255,.28) !important; padding: .12rem .6rem !important; min-height: 0 !important; font-size: .74rem !important; box-shadow:none !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stButton > button p { font-size: .74rem !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stButton > button:hover { background: rgba(255,255,255,.28) !important; transform:none; }
.bot-q { font-size: 1.02rem; color: var(--ink); font-weight: 700; margin: 0 0 .2rem; }
.bot-s { color: var(--muted); font-size: .88rem; margin: 0 0 .5rem; line-height: 1.55; }
[class*="st-key-step_"] { margin-top: .5rem; }
/* result bubble: no extra card around the dashboard */
[data-testid="stChatMessageContent"]:has([class*="st-key-resultwrap"]) { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }

/* bottom chat bar */
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stBottom"] { background: linear-gradient(to top, color-mix(in srgb, var(--bg) 80%, transparent) 45%, transparent) !important; }
[data-testid="stBottomBlockContainer"] { max-width: 900px !important; padding-bottom: 1.2rem !important; }
[data-testid="stChatInput"] { background: var(--surface) !important; border: 1.5px solid var(--line) !important; border-radius: 18px !important; box-shadow: var(--shadow_lg) !important; transition: border-color .2s ease, box-shadow .2s ease; }
[data-testid="stChatInput"]:focus-within { border-color: var(--brand) !important; box-shadow: 0 0 0 4px var(--glow), var(--shadow_lg) !important; }
[data-testid="stChatInput"] textarea { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; font-size: .95rem !important; }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] [data-testid="stChatInputTextArea"] { background: transparent !important; border:none !important; }
[data-testid="stChatInputSubmitButton"] { background: linear-gradient(135deg, var(--brand), var(--brand2)) !important; color:#fff !important; border-radius: 12px !important; }
[data-testid="stChatInputSubmitButton"] svg { color:#fff !important; fill:#fff !important; }
[data-testid="stChatInput"] button:not([data-testid="stChatInputSubmitButton"]) svg { color: var(--muted) !important; fill: currentColor; }
[data-testid="stChatInput"] button:not([data-testid="stChatInputSubmitButton"]) { color: var(--muted) !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] { margin: 0 !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"] { gap: .45rem !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p { line-height: 1.5 !important; padding: 0 !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] p strong { color: var(--ink); font-weight: 700; }
.bottom-note { text-align:center; font-size:.72rem; color: var(--muted); margin-top: .35rem; }

/* sidebar recent list + links (tertiary buttons) */
[data-testid="stSidebar"] .stButton > button[kind="tertiary"], [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {
  width: 100%; justify-content: flex-start !important; text-align: left; border: 1px solid transparent !important; background: transparent !important;
  padding: .45rem .6rem !important; border-radius: 12px !important; box-shadow: none !important; color: var(--ink2) !important; min-height: 0 !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] > div, [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] [data-testid="stMarkdownContainer"] { justify-content: flex-start !important; text-align:left !important; width:100%; }
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] p { font-weight: 600 !important; font-size: .84rem !important; line-height: 1.35 !important; text-align: left !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover { background: var(--surface2) !important; border-color: var(--line) !important; transform: none !important; color: var(--brand) !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] details { box-shadow:none; }
.sb-empty { font-size:.8rem; color: var(--muted); padding: .3rem .2rem; }

@media (max-width: 900px) {
  .flow { flex-wrap: wrap; } .flow .fs { min-width: 110px; } .flow .arrow { display:none; }
  .hero, .urg { grid-template-columns: 1fr; padding: 1.5rem; }
  .orb-wrap { height: 240px; }
  .g4, .g3 { grid-template-columns: repeat(2, minmax(0,1fr)); }
}
@media (max-width: 560px) { .g4, .g3, .g2 { grid-template-columns: 1fr; } .bench .row { grid-template-columns: 90px 1fr 64px; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; } }
"""

GATE_CSS = r"""
/* ---------- API key "Connect" screen ---------- */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display:none !important; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1180px; padding-top: 3.2rem; }

.g-left { padding: 1.2rem 1.5rem 0 0; animation: fadeUp .8s both; }
.g-logo { display:flex; align-items:center; gap:.8rem; margin-bottom: 2.2rem; }
.g-logo .lg { width:52px; height:52px; border-radius:16px; display:grid; place-items:center; color:#fff; background: linear-gradient(135deg, var(--brand), var(--brand2)); box-shadow: 0 14px 30px -12px var(--brand); position:relative; }
.g-logo .lg::after { content:""; position:absolute; inset:-6px; border-radius: 20px; border: 2px solid color-mix(in srgb, var(--brand) 45%, transparent); animation: ripple 2.6s ease-out infinite; }
.g-logo .lg svg { animation: beat 2.2s ease-in-out infinite; }
.g-logo .nm { font-weight: 800; font-size: 1.35rem; color: var(--ink); letter-spacing:-.02em; }
.g-logo .nm small { display:block; font-size: .74rem; font-weight:600; color: var(--muted); letter-spacing: 0; }
.g-left h1 { font-size: clamp(2.2rem, 4.2vw, 3.5rem) !important; line-height: 1.04 !important; font-weight: 800 !important; letter-spacing: -.04em !important; margin: 1rem 0 1rem !important; padding:0 !important; }
.g-left p.lead { font-size: 1.05rem; color: var(--muted); line-height:1.7; max-width: 520px; margin:0; }
.g-feats { margin-top: 1.8rem; display:flex; flex-direction:column; gap: .8rem; }
.g-feat { display:flex; gap: .85rem; align-items:center; padding: .75rem .9rem; border-radius: 16px; background: var(--glass); backdrop-filter: blur(14px); border: 1px solid var(--line); max-width: 500px; animation: slideIn .6s both; transition: transform .3s ease, border-color .3s ease; }
.g-feat:hover { transform: translateX(6px); border-color: color-mix(in srgb, var(--brand) 40%, var(--line)); }
.g-feat .fi { width: 40px; height:40px; border-radius: 12px; display:grid; place-items:center; color:#fff; flex-shrink:0; }
.g-feat b { color: var(--ink); font-size: .92rem; display:block; } .g-feat span { color: var(--muted); font-size: .8rem; }
.g-ecg { margin-top: 1.6rem; max-width: 500px; }
.g-ecg path { stroke-dasharray: 1000; stroke-dashoffset: 1000; animation: ecg 3.4s linear infinite; }

[class*="st-key-gatecard"] { position:relative; background: var(--glass); backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%); border-radius: 28px; padding: 2rem 2rem 1.6rem; border: 1px solid var(--line); box-shadow: var(--shadow_lg); animation: fadeUp .9s .15s both; }
[class*="st-key-gatecard"]::before { content:""; position:absolute; inset:-1px; border-radius: 29px; padding: 1.5px; background: linear-gradient(135deg, var(--brand), transparent 35%, transparent 65%, var(--brand2)); -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0); -webkit-mask-composite: xor; mask-composite: exclude; pointer-events:none; }
.gc-top { display:flex; align-items:center; justify-content:space-between; margin-bottom: 1.4rem; }
.gc-key { width: 64px; height:64px; border-radius: 20px; display:grid; place-items:center; color: #fff; background: linear-gradient(135deg, var(--brand), var(--brand2)); box-shadow: 0 16px 34px -14px var(--brand); animation: floaty 4.5s ease-in-out infinite; }
.gc-secure { display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem; font-weight:800; color: #059669; background: rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.3); padding: .3rem .65rem; border-radius:999px; }
.gc-title { font-size: 1.55rem; font-weight: 800; color: var(--ink); letter-spacing:-.03em; margin: 0 0 .35rem; }
.gc-sub { color: var(--muted); font-size: .92rem; line-height: 1.6; margin: 0 0 1.2rem; }
.gc-note { display:flex; gap:.6rem; align-items:flex-start; font-size: .78rem; color: var(--muted); line-height:1.5; padding: .75rem .85rem; border-radius: 14px; background: var(--surface2); border: 1px solid var(--line); margin-top: .9rem; }
.gc-note svg { color: var(--accent); margin-top:1px; }
.gc-or { display:flex; align-items:center; gap:.7rem; color: var(--muted); font-size:.74rem; font-weight:700; margin: .7rem 0; text-transform: uppercase; letter-spacing:.1em; }
.gc-or::before, .gc-or::after { content:""; flex:1; height:1px; background: var(--line); }
[class*="st-key-gatecard"] [data-baseweb="input"] { height: 54px; border-radius: 14px !important; }
[class*="st-key-gatecard"] [data-baseweb="input"] input { font-family:'JetBrains Mono', monospace !important; font-size: .95rem !important; letter-spacing: .02em; }
[class*="st-key-gatecard"] .stButton > button { width:100%; height: 52px; font-size: 1rem !important; border-radius: 14px !important; }
.gc-success { text-align:center; padding: 1.6rem 0 .6rem; }
.gc-success .ck { width: 86px; height: 86px; margin: 0 auto 1rem; border-radius:50%; display:grid; place-items:center; background: linear-gradient(135deg, #10B981, #0EA5A4); box-shadow: 0 20px 40px -16px #10B981; animation: popIn .6s cubic-bezier(.3,1.5,.5,1) both; }
.gc-success .ck path { stroke-dasharray: 40; stroke-dashoffset: 40; animation: drawCheck .5s .35s ease forwards; }
.gc-success h3 { font-weight: 800 !important; margin: 0 0 .3rem !important; }
.gc-success p { color: var(--muted); margin:0; }
.loader-bar { height: 4px; border-radius: 99px; background: var(--surface2); overflow:hidden; margin-top: 1.2rem; }
.loader-bar i { display:block; height:100%; width: 40%; background: linear-gradient(90deg, var(--brand), var(--brand2)); border-radius:99px; animation: load 1.1s ease-in-out forwards; }
@keyframes load { from { width: 0; } to { width: 100%; } }
.trust { display:flex; gap: 1.2rem; flex-wrap:wrap; margin-top: 1.4rem; color: var(--muted); font-size: .76rem; font-weight:700; }
.trust span { display:inline-flex; align-items:center; gap:.35rem; }
"""


def build_css(dark: bool, gate: bool = False) -> str:
    """Return the full <style> block for the current theme / screen."""
    css = FONTS + theme_vars(dark) + BASE_CSS + (GATE_CSS if gate else "")
    return f"<style>{css}</style>"


# ============================================================================
# COMPONENTS — small HTML building blocks
# ============================================================================
ECG_PATH = ("M0 30 H120 L135 30 L145 8 L158 52 L170 18 L180 30 H260 L272 30 L280 22 L288 38 L296 30 "
            "H380 L395 30 L405 4 L418 56 L430 14 L440 30 H560")


def ecg_svg(cls: str = "ecg-line", color: str = "var(--brand)") -> str:
    return (f'<svg class="{cls}" viewBox="0 0 560 60" preserveAspectRatio="none" fill="none">'
            f'<path d="{ECG_PATH}" stroke="{color}" stroke-width="2.4" stroke-linecap="round" '
            f'stroke-linejoin="round" opacity=".85"/></svg>')


# --------------------------------------------------------------------------
# Connect (API key) screen
# --------------------------------------------------------------------------
def gate_background() -> str:
    return ('<div class="gate-bg"><div class="grid-l"></div><div class="b b1"></div>'
            '<div class="b b2"></div><div class="b b3"></div></div>')


def gate_left() -> str:
    feats = [
        ("brain-circuit", "linear-gradient(135deg,#4F46E5,#7C3AED)", "Structured AI guidance",
         "Summary, possible conditions, next steps & doctor questions"),
        ("gauge", "linear-gradient(135deg,#F97316,#EF4444)", "Instant urgency triage",
         "Colour-coded LOW → EMERGENCY with a built-in safety net"),
        ("languages", "linear-gradient(135deg,#0EA5A4,#22C55E)", "10 languages incl. Urdu",
         "Right-to-left layouts, voice input & PDF reports"),
    ]
    rows = "".join(
        f'<div class="g-feat d{i + 2}" style="animation-delay:{.25 + i * .12}s">'
        f'<div class="fi" style="background:{bg}">{icon(ic, 20)}</div>'
        f'<div><b>{t}</b><span>{s}</span></div></div>'
        for i, (ic, bg, t, s) in enumerate(feats)
    )
    return f"""
<div class="g-left">
  <div class="g-logo"><div class="lg">{icon("heart-pulse", 26, 2.2)}</div>
    <div class="nm">{APP_NAME}<small>Symptom assessment &amp; patient guidance</small></div></div>
  <span class="eyebrow"><span class="pill">NEW</span> LangChain · OpenAI · Streamlit</span>
  <h1>Your calm,<br/><span class="grad-text">AI&#8209;powered</span><br/>health guide.</h1>
  <p class="lead">Describe how you feel and get clear, structured, safety-first guidance in seconds —
  so you know what your symptoms may mean and what to ask your doctor.</p>
  <div class="g-feats">{rows}</div>
  <div class="g-ecg">{ecg_svg("ecg-line")}</div>
</div>"""


def gate_card_header() -> str:
    return f"""
<div class="gc-top">
  <div class="gc-key">{icon("key-round", 30, 2)}</div>
  <span class="gc-secure">{icon("lock", 13, 2.4)} End-to-end session only</span>
</div>
<div class="steps" style="margin-bottom:1.1rem">
  <div class="step on"><span class="n">1</span>Connect</div><div class="step-line"></div>
  <div class="step"><span class="n">2</span>Describe</div><div class="step-line"></div>
  <div class="step"><span class="n">3</span>Get guidance</div>
</div>
<div class="gc-title">Connect your OpenAI key</div>
<p class="gc-sub">MediGuide AI runs on your own OpenAI account. Paste your API key to unlock the assistant.</p>"""


def gate_card_note() -> str:
    return f"""
<div class="gc-note">{icon("shield-check", 18, 2)}<div>Your key is kept <b>only in this browser session's memory</b>.
It is never written to disk, logged, or stored in the code — closing the tab forgets it.</div></div>
<div class="trust"><span>{icon("lock", 13)} Not stored</span><span>{icon("eye-off", 13)} Masked input</span>
<span>{icon("zap", 13)} Verified live</span></div>"""


def gate_success(masked: str) -> str:
    return f"""
<div class="gc-success">
  <div class="ck"><svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3"
     stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg></div>
  <h3>You're connected</h3>
  <p>Key <code>{escape(masked)}</code> verified. Opening MediGuide AI…</p>
  <div class="loader-bar"><i></i></div>
</div>"""


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def sb_label(text: str, ic: str) -> str:
    return f'<div class="sb-label">{icon(ic, 13, 2.4)} {escape(text)}</div>'


def sb_connection(masked: str, model: str) -> str:
    return (f'<div class="sb-conn"><span class="dot"></span><div><b style="color:var(--ink)">Connected</b>'
            f' · {escape(model)}<br/><span class="mono" style="font-size:.74rem;color:var(--muted)">'
            f'{escape(masked)}</span></div></div>')


# --------------------------------------------------------------------------
# Main page pieces
# --------------------------------------------------------------------------


def section_title(kicker: str, title: str, sub: str = "", ic: str = "sparkles", right: str = "") -> str:
    sub_html = f"<p>{sub}</p>" if sub else ""
    return (f'<div class="section-title fade-up"><div><div class="kicker">{icon(ic, 14, 2.4)} {kicker}</div>'
            f'<h2>{title}</h2>{sub_html}</div>{right}</div>')


def card_head(ic: str, title: str, sub: str, tone: str = "") -> str:
    return (f'<div class="card-head"><div class="ib {tone}">{icon(ic, 19, 2.2)}</div>'
            f'<div><h4>{title}</h4><small>{sub}</small></div></div>')


def severity_meter(value: int, label: str, color: str) -> str:
    pct = value * 10
    return (f'<div class="sev"><div class="sev-bar"><i style="width:{pct}%;background:linear-gradient(90deg,{color}99,{color})"></i></div>'
            f'<span class="sev-chip" style="background:{color}">{value}/10 · {label}</span></div>')


def gauge_svg(level: str) -> str:
    """Semi-circular urgency gauge with an animated needle."""
    meta = URGENCY_LEVELS[level]
    angle = -90 + meta["score"] * 1.8
    segs = [("#10B981", 180, 225, "a1"), ("#F59E0B", 225, 270, "a2"),
            ("#F97316", 270, 315, "a3"), ("#EF4444", 315, 360, "a4")]
    import math

    def pt(a, r=100):
        rad = math.radians(a)
        return 130 + r * math.cos(rad), 130 + r * math.sin(rad)

    arcs = ""
    for col, a0, a1, cls in segs:
        x0, y0 = pt(a0 + 1.5)
        x1, y1 = pt(a1 - 1.5)
        arcs += (f'<path class="arc {cls}" d="M{x0:.1f} {y0:.1f} A100 100 0 0 1 {x1:.1f} {y1:.1f}" '
                 f'stroke="{col}" stroke-width="18" stroke-linecap="round" fill="none"/>')
    return f"""
<svg class="gauge" viewBox="0 0 260 160" style="--angle:{angle}deg">
  <path d="M30 130 A100 100 0 0 1 230 130" stroke="var(--surface2)" stroke-width="22" fill="none" stroke-linecap="round"/>
  {arcs}
  <g class="needle"><line x1="130" y1="130" x2="130" y2="46" stroke="var(--ink)" stroke-width="4" stroke-linecap="round"/></g>
  <circle cx="130" cy="130" r="11" fill="var(--ink)"/><circle cx="130" cy="130" r="4.5" fill="{meta['color']}"/>
  <text x="30" y="154" font-size="10" fill="var(--muted)" text-anchor="middle" font-weight="700">LOW</text>
  <text x="230" y="154" font-size="10" fill="var(--muted)" text-anchor="middle" font-weight="700">EMERG.</text>
</svg>"""


def urgency_banner(level: str, summary: str, tags: list[tuple[str, str, str]]) -> str:
    meta = URGENCY_LEVELS[level]
    tag_html = "".join(f'<span class="tag {cls}">{icon(ic, 13, 2.4)} {escape(t)}</span>' for ic, t, cls in tags)
    emergency = " emergency" if level == "EMERGENCY" else ""
    return f"""
<div class="urg{emergency}" style="--uc:{meta['color']};--us:{meta['soft']}">
  <div>
    <span class="urg-level"><span class="ui">{icon(meta['icon'], 15, 2.6)}</span>URGENCY · {level}</span>
    <h3>{escape(meta['meaning'])}</h3>
    <p dir="auto">{escape(summary)}</p>
    <div class="meta">{tag_html}</div>
  </div>
  <div style="display:grid;place-items:center">{gauge_svg(level)}</div>
</div>"""


def sos_box() -> str:
    return f"""
<div class="sos"><div class="si">{icon("siren", 24, 2.2)}</div>
<div><b>Seek emergency help immediately</b><span>Call your local emergency number or go to the nearest
emergency department now. Do not drive yourself.</span></div>
<div class="nums"><em>PK 1122</em><em>US 911</em><em>UK 999</em><em>EU 112</em></div></div>"""


def list_cards(items: list[str], color: str, numbered: bool = True, ic: str | None = None) -> str:
    if not items:
        return '<p style="color:var(--muted)">Nothing to show.</p>'
    out = ""
    for i, item in enumerate(items, 1):
        badge = str(i) if numbered else icon(ic or "circle-check", 15, 2.4)
        out += (f'<div class="li-card" style="--lc:{color};animation-delay:{i * .06:.2f}s">'
                f'<span class="num">{badge}</span><span class="tx" dir="auto">{escape(item)}</span></div>')
    return out


def stream_head(text: str) -> str:
    return (f'<div class="stream-head">{icon("bot", 18, 2.2)} {escape(text)}'
            f'<span class="live-dot"><i></i><i></i><i></i></span></div>')


def history_card(row: dict) -> str:
    meta = URGENCY_LEVELS.get(row.get("urgency") or "MEDIUM", URGENCY_LEVELS["MEDIUM"])
    chips = "".join(f"<span>{escape(s)}</span>" for s in (row.get("symptoms") or [])[:6])
    other = (row.get("form") or {}).get("other_symptoms", "")
    if other:
        chips += f"<span>{escape(other[:40])}</span>"
    when = str(row.get("created_at", "")).replace("T", " · ")
    cached = " · ⚡ cached" if row.get("cached") else ""
    return f"""
<div class="h-card" style="--uc:{meta['color']}"><div class="strip"></div>
<div><div class="t">{escape(str(row.get('gender', '')))}, {escape(str(row.get('age', '')))} years
 — severity {row.get('severity', '-')}/10</div>
<div class="s"><span>{icon("clock", 12)} {escape(when)}</span><span>{icon("languages", 12)} {escape(str(row.get('language', '')))}</span>
<span>{icon("cpu", 12)} {escape(str(row.get('model', '')))} · {row.get('elapsed', 0):.2f}s{cached}</span></div>
<div class="chips">{chips}</div></div>
<span class="u-pill" style="--uc:{meta['color']};margin-right:.2rem">{escape(str(row.get('urgency', '')))}</span></div>"""


def empty_state(ic: str, title: str, text: str) -> str:
    return f'<div class="empty"><div class="ei">{icon(ic, 30, 2)}</div><h4>{title}</h4><p>{text}</p></div>'


def concept_cards(items: list[tuple[str, str, str, str]], cols: str = "g4") -> str:
    cards = "".join(
        f'<div class="c-card" style="animation-delay:{i * .06:.2f}s"><span class="file">{escape(f)}</span>'
        f'<div class="ci">{icon(ic, 19, 2.2)}</div><h5>{t}</h5><p>{d}</p></div>'
        for i, (ic, t, d, f) in enumerate(items)
    )
    return f'<div class="grid {cols}">{cards}</div>'


def flow_steps(steps: list[tuple[str, str, str]]) -> str:
    parts = []
    for i, (ic, t, s) in enumerate(steps):
        parts.append(f'<div class="fs" style="animation-delay:{i * .07:.2f}s"><div class="fi">{icon(ic, 19, 2.2)}</div>'
                     f'<b>{t}</b><small>{s}</small></div>')
        if i < len(steps) - 1:
            parts.append(f'<div class="arrow">{icon("arrow-right", 16, 2.4)}</div>')
    return f'<div class="flow">{"".join(parts)}</div>'


def message_bubble(kind: str, role: str, text: str) -> str:
    return (f'<div class="msg {kind}"><span class="role">{role}</span>'
            f'<div class="bubble">{escape(text)}</div></div>')


def bench_bars(rows: list[tuple[str, float, str]]) -> str:
    top = max((v for _, v, _ in rows), default=1) or 1
    out = ""
    for label, val, color in rows:
        pct = max(2, val / top * 100)
        out += (f'<div class="row"><span>{escape(label)}</span><div class="track">'
                f'<div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
                f'<span style="text-align:right">{val:.3f}s</span></div>')
    return f'<div class="bench">{out}</div>'


# --------------------------------------------------------------------------
# Chat assistant pieces
# --------------------------------------------------------------------------
CHAT_STEPS = ["Symptoms", "Timing", "About you", "Details", "Guidance"]


def chat_welcome() -> str:
    return f"""
<div class="welcome">
  <div class="wl">{icon("heart-pulse", 34, 2.1)}</div>
  <h1>How are you feeling today?</h1>
  <p>I'm <b>MediGuide AI</b>. Answer a few quick questions and I'll explain what your symptoms may mean,
  how urgent they look and what to do next.</p>
  <div class="w-chips">
    <span>{icon("shield-check", 13, 2.4)} Educational only</span>
    <span>{icon("info", 13, 2.4)} Not a diagnosis</span>
    <span class="sos-chip">{icon("siren", 13, 2.4)} Emergency? Call 1122 · 911 · 999</span>
  </div>
</div>"""


def chat_stepper(active: int) -> str:
    parts = []
    for i, name in enumerate(CHAT_STEPS):
        cls = "done" if i < active else ("on" if i == active else "")
        mark = icon("circle-check", 13, 2.6) if i < active else f"<b>{i + 1}</b>"
        parts.append(f'<span class="sp {cls}">{mark}{name}</span>')
    return f'<div class="stepper">{"".join(parts)}</div>'


def recent_label(row: dict) -> str:
    """Markdown label for a sidebar 'recent assessment' button."""
    color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "orange", "EMERGENCY": "red"}.get(row.get("urgency"), "gray")
    syms = list(row.get("symptoms") or [])
    other = (row.get("form") or {}).get("other_symptoms", "")
    text = ", ".join(syms[:2]) or other[:28] or "Assessment"
    if len(syms) > 2:
        text += f" +{len(syms) - 2}"
    when = str(row.get("created_at", ""))[5:16].replace("T", " ")
    return f":{color}[●] **{text}**  \n{when} · {row.get('urgency', '').title()}"


def sidebar_brand_compact() -> str:
    return (f'<div class="sb-brand"><div class="sb-logo">{icon("heart-pulse", 24, 2.2)}</div>'
            f'<div><div class="sb-name">{APP_NAME}</div><span class="sb-ver">v{APP_VERSION} · Educational</span>'
            f'</div></div>')


def sidebar_disclaimer() -> str:
    return (f'<div class="sb-disc"><b>{icon("triangle-alert", 14, 2.4)} Medical disclaimer</b>'
            f'{escape(DISCLAIMER_SHORT)} In an emergency call '
            f'<b style="display:inline;text-transform:none;letter-spacing:0">1122 / 911 / 999</b>.</div>')


# ============================================================================
# STREAMLIT UI — Connect screen, chat assistant, history, analytics, how it works
# ============================================================================
# ==========================================================================
# Page setup & session state
# ==========================================================================
# Brand colour for Streamlit's own widgets (sliders, toggles, tags, focus rings).
# Set here so the app needs no separate .streamlit/config.toml file.
for _opt, _val in {"theme.primaryColor": "#4F46E5", "theme.base": "light",
                   "client.toolbarMode": "minimal", "browser.gatherUsageStats": False}.items():
    try:
        st._config.set_option(_opt, _val)
    except Exception:
        pass

st.set_page_config(
    page_title=f"{APP_NAME} · Symptom Assistant",
    page_icon=":material/ecg_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

STAGES = ["symptoms", "timing", "about", "context", "done"]

DEFAULTS = {
    # connection
    "api_key": None, "connected": False, "gate_ok": False,
    # settings
    "dark": False, "page": "chat", "model": DEFAULT_MODEL, "temperature": DEFAULT_TEMPERATURE,
    "cache_mode": DEFAULT_CACHE_MODE, "language": DEFAULT_LANGUAGE, "f_language": DEFAULT_LANGUAGE,
    # the guided intake (one value per question)
    "stage": "symptoms", "f_age": "", "f_gender": GENDERS[0], "f_symptoms": [], "f_other": "",
    "f_duration": DURATIONS[1], "f_severity": 3, "f_conditions": "", "f_meds": "", "f_notes": "",
    "step_error": None, "run_pending": False, "pending_q": None, "scroll": False,
    # results + extras
    "result": None, "bench": None, "demo": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.markdown(build_css(st.session_state.dark, gate=not st.session_state.connected), unsafe_allow_html=True)

FORM_KEYS = {  # session key -> default (used when starting a new chat)
    "f_age": "", "f_gender": GENDERS[0], "f_symptoms": [], "f_other": "", "f_duration": DURATIONS[1],
    "f_severity": 3, "f_conditions": "", "f_meds": "", "f_notes": "",
}


# ==========================================================================
# Small helpers
# ==========================================================================
def html(markup: str) -> None:
    """Render trusted HTML built in src/components.py."""
    st.markdown(markup, unsafe_allow_html=True)


def get_llm(json_mode: bool = False):
    s = st.session_state
    return build_llm(s.api_key, s.model, s.temperature, json_mode=json_mode)


def friendly_error(exc: Exception) -> str:
    name, text = type(exc).__name__, str(exc)
    if "AuthenticationError" in name or "401" in text:
        return "OpenAI rejected the API key. Use **Change API key** in the sidebar and connect again."
    if "RateLimit" in name or "429" in text or "quota" in text.lower():
        return "OpenAI's rate limit or quota was reached. Wait a moment or check your billing, then retry."
    if "NotFound" in name:
        return "This model isn't available for your key. Choose another model in the sidebar settings."
    if "Connection" in name or "Timeout" in name:
        return "Couldn't reach OpenAI. Check your internet connection and try again."
    return f"Something went wrong while contacting the AI ({name}). Please try again."


def lang_label(x: str) -> str:
    return x if x == "English" else f"{x} · {LANGUAGES[x]['native']}"


def plotly_layout(fig, height=320):
    dark = st.session_state.dark
    ink, muted, grid = ("#EEF1FA", "#8D96B2", "#232C47") if dark else ("#0E1325", "#66708A", "#E4E7F1")
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans, sans-serif", color=ink, size=12),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=muted)),
        hoverlabel=dict(font_family="Plus Jakarta Sans, sans-serif"),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, color=muted)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, color=muted)
    return fig


def scroll_to_bottom():
    """Gently scroll the chat to the newest message (runs only when something new appears)."""
    js = ("<script>setTimeout(()=>{const d=window.parent.document;"
          "const m=d.querySelector('[data-testid=\"stAppScrollToBottomContainer\"]')||d.querySelector('[data-testid=\"stMain\"]');"
          "if(m){m.scrollTo({top:m.scrollHeight,behavior:'smooth'});}},400);</script>")
    with st.container(key="scroller"):
        st.iframe(js, height=1)


def collect_form() -> dict:
    s = st.session_state
    form = {
        "age": str(s.f_age).strip(), "gender": s.f_gender, "symptoms": list(s.f_symptoms),
        "other_symptoms": s.f_other.strip(), "duration": s.f_duration, "severity": int(s.f_severity),
        "existing_conditions": s.f_conditions.strip(), "medications": s.f_meds.strip(),
        "notes": s.f_notes.strip(),
    }
    form["symptoms_text"] = build_symptoms_text(form)
    return form


# ==========================================================================
# Callbacks (they run BEFORE the next rerun, so they may change widget values)
# ==========================================================================
def sync_language_from_sidebar():
    st.session_state.f_language = st.session_state.language


def _sync_answer(name: str):
    """Copy a widget's value into the persistent answer store (f_<name>)."""
    s = st.session_state
    s[f"f_{name}"] = s[f"w_{name}"]
    if name == "language":                 # keep the sidebar selector in step
        s.language = s.w_language


def bind(name: str) -> dict:
    """
    Widget keys are dropped by Streamlit once a widget stops being drawn
    (e.g. when a chat step collapses). So each answer lives in a permanent
    key `f_<name>`, and the on-screen widget uses a mirror key `w_<name>`.
    """
    s = st.session_state
    val = s[f"f_{name}"]
    s[f"w_{name}"] = list(val) if isinstance(val, list) else val
    return {"key": f"w_{name}", "on_change": _sync_answer, "args": (name,)}


def go_to_stage(stage: str):
    s = st.session_state
    s.stage, s.step_error, s.scroll = stage, None, True
    if stage != "done":
        s.result = None


def next_from_symptoms():
    s = st.session_state
    if not s.f_symptoms and not s.f_other.strip():
        s.step_error = ("Please pick or describe at least one symptom — "
                        "I won't contact the AI without symptoms.")      # test scenario 5
        return
    go_to_stage("timing")


def next_from_about():
    s = st.session_state
    age = str(s.f_age).strip()
    if not age.isdigit() or not (0 < int(age) <= 120):
        s.step_error = "Please enter an age between 1 and 120 (numbers only)."
        return
    go_to_stage("context")


def start_analysis():
    s = st.session_state
    problems = validate_form(collect_form())
    if problems:
        s.step_error = problems[0]
        return
    s.result, s.pending_q = None, None
    s.run_pending = True
    go_to_stage("done")


def skip_details():
    s = st.session_state
    s.f_conditions = s.f_meds = s.f_notes = ""
    start_analysis()


def load_example(name: str):
    ex = EXAMPLE_SCENARIOS[name]
    s = st.session_state
    s.f_age, s.f_gender, s.f_symptoms = ex["age"], ex["gender"], list(ex["symptoms"])
    s.f_duration, s.f_severity, s.f_other = ex["duration"], ex["severity"], ex["other"]
    s.f_conditions, s.f_meds, s.f_notes = ex["conditions"], ex["medications"], ex["notes"]
    go_to_stage("context")          # everything filled in — the user just presses "Analyse"


def new_chat():
    s = st.session_state
    for k, v in FORM_KEYS.items():
        s[k] = list(v) if isinstance(v, list) else v
    s.result, s.pending_q, s.run_pending, s.step_error = None, None, False, None
    s.stage, s.page = "symptoms", "chat"


def open_history_item(row: dict):
    s = st.session_state
    f = row.get("form") or {}
    s.f_age, s.f_gender = str(f.get("age", "")), f.get("gender", GENDERS[0])
    s.f_symptoms = [x for x in f.get("symptoms", []) if x in COMMON_SYMPTOMS]
    s.f_other, s.f_duration = f.get("other_symptoms", ""), f.get("duration", DURATIONS[1])
    s.f_severity = int(f.get("severity", 3))
    s.f_conditions, s.f_meds, s.f_notes = (f.get("existing_conditions", ""), f.get("medications", ""),
                                           f.get("notes", ""))
    lang = row.get("language") or DEFAULT_LANGUAGE
    s.result = {
        "form": {**f, "symptoms_text": build_symptoms_text(f)}, "assessment": row["result"], "raw": "",
        "error": None, "elapsed": row.get("elapsed") or 0.0, "cached": bool(row.get("cached")),
        "safety_note": None, "narrative": row.get("narrative") or "", "needs_stream": False, "chat": [],
        "language": lang, "model": row.get("model") or DEFAULT_MODEL, "history_id": row["id"],
        "prompt_inputs": None, "reopened": True, "narrative_stats": {},
    }
    s.stage, s.page, s.run_pending, s.pending_q = "done", "chat", False, None


def ask_suggestion(q: str):
    st.session_state.pending_q = q
    st.session_state.scroll = True


def set_page(p: str):
    st.session_state.page = p


def disconnect():
    for k in ["api_key", "result", "bench", "demo"]:
        st.session_state[k] = None
    st.session_state.connected = False
    st.session_state.gate_ok = False


# ==========================================================================
# 1) CONNECT SCREEN (API key)
# ==========================================================================
@st.fragment(run_every=0.7)
def gate_advance():
    """After the success animation has played (~1.4 s), switch to the main app."""
    if time.time() - st.session_state.get("gate_ok_at", 0) > 1.3:
        st.session_state.connected = True
        st.session_state.gate_ok = False
        st.rerun(scope="app")


def render_gate():
    html(gate_background())
    left, right = st.columns([1.12, 1], gap="large", vertical_alignment="center")
    with left:
        html(gate_left())
    with right:
        with st.container(key="gatecard"):
            if st.session_state.gate_ok:
                html(gate_success(mask_key(st.session_state.api_key)))
                gate_advance()          # opens the app after a short success animation
                return

            html(gate_card_header())
            key = st.text_input("OpenAI API key", type="password", placeholder="sk-proj-••••••••••••••••",
                                key="gate_key", help="Find it at platform.openai.com → API keys")
            if st.button("Connect securely", type="primary", icon=":material/arrow_forward:",
                         use_container_width=True, key="connect_btn"):
                with st.spinner("Verifying your key with OpenAI…"):
                    ok, msg = verify_api_key(key)
                if ok:
                    st.session_state.api_key = key.strip()
                    st.session_state.gate_ok = True
                    st.session_state.gate_ok_at = time.time()
                    st.rerun()
                else:
                    st.error(msg, icon=":material/error:")

            env_key = get_env_api_key()
            if env_key:
                html('<div class="gc-or">or</div>')
                if st.button("Use the key from my .env file", icon=":material/description:",
                             use_container_width=True, key="env_btn"):
                    with st.spinner("Verifying key from .env…"):
                        ok, msg = verify_api_key(env_key)
                    if ok:
                        st.session_state.api_key = env_key
                        st.session_state.gate_ok = True
                        st.session_state.gate_ok_at = time.time()
                        st.rerun()
                    else:
                        st.error(msg, icon=":material/error:")

            with st.expander("How do I get an OpenAI API key?", icon=":material/help:"):
                st.markdown(
                    "1. Sign in at **[platform.openai.com](https://platform.openai.com/api-keys)**\n"
                    "2. Open **API keys → Create new secret key**\n"
                    "3. Copy the key (it starts with `sk-`) and paste it above\n"
                    "4. Optional: save it in a local `.env` file as `OPENAI_API_KEY=sk-...` "
                    "(never commit `.env` to GitHub)"
                )
            html(gate_card_note())


# ==========================================================================
# 2) SIDEBAR — new chat, recent assessments, settings
# ==========================================================================
def render_sidebar():
    s = st.session_state
    with st.sidebar:
        html(sidebar_brand_compact())
        st.button("New assessment", type="primary", icon=":material/add_comment:", on_click=new_chat,
                  use_container_width=True, key="sb_new")

        html(sb_label("Recent", "history"))
        rows = list_assessments(limit=6)
        if rows:
            for row in rows:
                st.button(recent_label(row), key=f"recent_{row['id']}", type="tertiary",
                          on_click=open_history_item, args=(row,), use_container_width=True)
            st.button("View all history", icon=":material/arrow_forward:", type="tertiary",
                      on_click=set_page, args=("history",), use_container_width=True, key="sb_all")
        else:
            html('<div class="sb-empty">Your assessments will appear here.</div>')

        html(sb_label("Explore", "layers"))
        if s.page != "chat":
            st.button("Back to the assistant", icon=":material/forum:", type="tertiary", on_click=set_page,
                      args=("chat",), use_container_width=True, key="sb_chat")
        st.button("Analytics", icon=":material/monitoring:", type="tertiary", on_click=set_page,
                  args=("analytics",), use_container_width=True, key="sb_analytics")
        st.button("How it works", icon=":material/account_tree:", type="tertiary", on_click=set_page,
                  args=("how",), use_container_width=True, key="sb_how")

        html(sb_label("Preferences", "sparkles"))
        st.selectbox("Answer language", list(LANGUAGES), key="language", on_change=sync_language_from_sidebar,
                     format_func=lang_label)
        st.toggle("Dark mode", key="dark")
        with st.expander("AI model settings", icon=":material/tune:"):
            st.selectbox("Model", AVAILABLE_MODELS, key="model")
            st.slider("Creativity (temperature)", 0.0, 1.0, step=0.1, key="temperature",
                      help="Lower = more consistent answers. 0.2 is recommended for medical guidance.")
            st.selectbox("LLM cache", CACHE_MODES, key="cache_mode",
                         help="Cached answers return instantly for identical requests.")
            if s.cache_mode != "Off" and st.button("Clear this cache", icon=":material/delete_sweep:",
                                                    use_container_width=True):
                clear_cache(s.cache_mode)
                st.toast("Cache cleared — the next request will call the API again.", icon=":material/check:")

        st.write("")
        html(sidebar_disclaimer())
        st.write("")
        html(sb_connection(mask_key(s.api_key or ""), s.model))
        st.write("")
        st.button("Change API key", icon=":material/logout:", on_click=disconnect, use_container_width=True)


# ==========================================================================
# 3) CHAT ASSISTANT
# ==========================================================================
BOT = ":material/ecg_heart:"


def bot_say(question: str, sub: str = "") -> None:
    html(f'<div class="bot-q">{question}</div>' + (f'<div class="bot-s">{sub}</div>' if sub else ""))


def user_bubble(text: str, edit_stage: str | None) -> None:
    with st.chat_message("user"):
        st.markdown(text)
        if edit_stage:
            st.button("Edit", key=f"edit_{edit_stage}", icon=":material/edit:", on_click=go_to_stage,
                      args=(edit_stage,))


def step_error():
    if st.session_state.step_error:
        st.warning(st.session_state.step_error, icon=":material/info:")


def handle_chat_input(msg) -> None:
    """Text (or a voice note) typed in the bottom bar — meaning depends on the current step."""
    s = st.session_state
    text = (getattr(msg, "text", None) if not isinstance(msg, str) else msg) or ""
    audio = getattr(msg, "audio", None) if not isinstance(msg, str) else None
    if audio is not None:
        try:
            with st.spinner("Transcribing your voice note…"):
                spoken = transcribe_audio(s.api_key, audio.getvalue(), getattr(audio, "name", "voice.wav"))
            text = f"{text} {spoken}".strip()
            if spoken:
                st.toast("Voice note transcribed.", icon=":material/mic:")
        except Exception as exc:
            st.toast(f"Voice transcription failed: {friendly_error(exc)}")
    text = text.strip()
    if not text:
        return

    s.scroll = True
    if s.stage == "symptoms":
        found = [sym for sym in COMMON_SYMPTOMS if sym.lower() in text.lower()]
        s.f_symptoms = sorted(set(s.f_symptoms) | set(found))
        s.f_other = (s.f_other + " " + text).strip()
        go_to_stage("timing")
    elif s.stage == "about":
        num = re.search(r"\b(\d{1,3})\b", text)
        if num:
            s.f_age = num.group(1)
        low = text.lower()
        if re.search(r"\b(female|woman|girl)\b", low):
            s.f_gender = "Female"
        elif re.search(r"\b(male|man|boy)\b", low):
            s.f_gender = "Male"
        if s.f_age:
            next_from_about()
    elif s.stage in ("timing", "context"):
        s.f_notes = (s.f_notes + " " + text).strip()
        st.toast("Added to your notes.", icon=":material/note_add:")
        if s.stage == "context":
            start_analysis()
    elif s.stage == "done" and s.result and s.result.get("assessment"):
        s.pending_q = text


def run_pipeline(form: dict) -> None:
    """form -> prompt inputs -> cache -> LLM + LLMChain -> run -> parse JSON."""
    s = st.session_state
    language = s.f_language
    configure_cache(s.cache_mode)                                     # select cache
    prompt_inputs = build_prompt_inputs(form, language)               # PromptTemplate
    result = {
        "form": form, "assessment": None, "raw": "", "error": None, "elapsed": 0.0, "cached": False,
        "safety_note": None, "narrative": "", "needs_stream": True, "chat": [], "language": language,
        "model": s.model, "history_id": None, "prompt_inputs": prompt_inputs, "reopened": False,
        "narrative_stats": {},
    }
    with st.status("Analysing your symptoms…", expanded=True) as status:
        st.write(":material/edit_note: Building your patient profile (**PromptTemplate**)")
        st.write(f":material/database: Checking the **{s.cache_mode}**")
        st.write(f":material/hub: Running the **LLMChain** on **{s.model}**")
        try:
            chain = build_assessment_chain(get_llm(json_mode=True))   # ChatOpenAI + LLMChain
            out = run_assessment(chain, prompt_inputs)
        except Exception as exc:
            status.update(label="Couldn't reach the AI", state="error", expanded=False)
            st.error(friendly_error(exc), icon=":material/cloud_off:")
            st.button("Try again", icon=":material/refresh:", on_click=start_analysis, key="retry_btn")
            return
        result.update(raw=out["raw"], elapsed=out["elapsed"], cached=out["cached"])
        st.write(":material/data_object: Reading the structured **JSON** safely")
        data, err = safe_parse_json(out["raw"])
        if err:
            result["error"] = err
            status.update(label="I couldn't read the AI's answer", state="error", expanded=False)
        else:
            data, note = apply_safety_floor(data, form)
            result.update(assessment=data, safety_note=note)
            where = "from cache ⚡" if out["cached"] else "live"
            status.update(label=f"Done in {out['elapsed']:.2f}s ({where})", state="complete", expanded=False)
    if result["assessment"]:
        result["history_id"] = save_assessment(form, result["assessment"], "", language, s.model,
                                               result["elapsed"], result["cached"])
    s.result = result


def render_results(res: dict) -> None:
    """The results card shown inside the assistant's chat bubble."""
    rtl = LANGUAGES.get(res["language"], {}).get("rtl", False)
    if res["error"]:
        st.error(f"**The AI's answer came back in an unexpected format.** {res['error']} "
                 "Please try again — this is usually a one-off.", icon=":material/error:")
        with st.expander("Raw model output (for debugging)", icon=":material/code:"):
            st.code(res["raw"] or "(empty)", language="json")
        st.button("Try again", icon=":material/refresh:", on_click=start_analysis, key="retry_parse")
        return

    a, form = res["assessment"], res["form"]
    level = a["urgency_level"]
    meta = URGENCY_LEVELS[level]

    tags = [("cpu", res["model"], "")]
    if res["reopened"]:
        tags.append(("history", "Opened from history", ""))
    elif res["cached"]:
        tags.append(("zap", f"From cache in {res['elapsed']:.2f}s", "cache"))
    else:
        tags.append(("activity", f"Live · {res['elapsed']:.2f}s", "live"))
    tags.append(("languages", res["language"], ""))

    with st.container(key="resultwrap"):
        if res["safety_note"]:
            st.warning(res["safety_note"], icon=":material/shield:")
        html(urgency_banner(level, a["summary"], tags))
        if level == "EMERGENCY":
            html(sos_box())
            st.error("**EMERGENCY:** Please call your local emergency number or go to the nearest emergency "
                     "department immediately. Do not wait for symptoms to get worse.", icon=":material/emergency:")
        st.write("")

        short = {"LOW": "Self-care & monitor", "MEDIUM": "See a doctor soon",
                 "HIGH": "Get care today", "EMERGENCY": "Call emergency now"}[level]
        sev_label, _ = severity_label(int(form["severity"]))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Urgency", meta["label"], delta=short, delta_color="off", delta_arrow="off")
        m2.metric("Severity", f"{form['severity']}/10", delta=sev_label, delta_color="off", delta_arrow="off")
        m3.metric("Possibilities", len(a["possible_conditions"]), delta="education only", delta_color="off",
                  delta_arrow="off")
        m4.metric("Response", f"{res['elapsed']:.2f}s", delta="⚡ cached" if res["cached"] else "live API",
                  delta_color="normal" if res["cached"] else "off", delta_arrow="off")
        st.write("")

        with st.container(key="rtl_results" if rtl else "ltr_results"):
            tabs = st.tabs([":material/auto_awesome: Guidance", ":material/checklist: Next steps",
                            ":material/biotech: Possible causes", ":material/contact_support: Ask your doctor",
                            ":material/warning: Warning signs"])
            with tabs[1]:
                st.success("**Recommended next steps** — practical things you can do now.", icon=":material/task_alt:")
                html(list_cards(a["recommended_next_steps"], "#10B981"))
            with tabs[2]:
                st.info("Listed **for education only** — not a diagnosis. Only a healthcare professional can "
                        "diagnose you.", icon=":material/school:")
                for i, cond in enumerate(a["possible_conditions"], 1):
                    with st.expander(f"{i}.  {cond['name']}", expanded=(i == 1), icon=":material/biotech:"):
                        st.write(cond["reason"] or "—")
            with tabs[3]:
                st.info("**Questions to ask a healthcare professional** — take these to your appointment.",
                        icon=":material/contact_support:")
                html(list_cards(a["questions_for_doctor"], "#0EA5E9"))
            with tabs[4]:
                st.error("**Get help immediately** if you notice any of these:", icon=":material/emergency:")
                html(list_cards(a["warning_signs"], "#EF4444", numbered=False, ic="triangle-alert"))
            # Guidance tab is filled LAST so the other tabs are ready while it streams
            with tabs[0]:
                st.info(f"**Summary —** {a['summary']}", icon=":material/summarize:")
                with st.container(key="stream_box"):
                    html(stream_head("Your guidance note"))
                    if res["needs_stream"]:
                        stats: dict = {}
                        text = st.write_stream(stream_narrative(get_llm(), res["prompt_inputs"], level, stats))
                        res["narrative"] = text if isinstance(text, str) else "".join(map(str, text))
                        res["needs_stream"], res["narrative_stats"] = False, stats
                        if res["history_id"]:
                            update_narrative(res["history_id"], res["narrative"])
                    else:
                        st.markdown(res["narrative"] or "_No guidance note saved for this assessment._")
                ns = res.get("narrative_stats") or {}
                if ns:
                    st.caption(f"Streamed with `llm.stream()` → `st.write_stream()` in {ns.get('elapsed', 0):.2f}s"
                               + (" · replayed from cache ⚡" if ns.get("cached") else " · live"))

        st.write("")
        c1, c2, c3 = st.columns([1.25, 1, 1])
        try:
            pdf = build_pdf_report(form, a, res["narrative"], res["language"], res["model"], form["symptoms_text"])
            c1.download_button("Download PDF report", data=pdf, type="primary", icon=":material/picture_as_pdf:",
                               file_name=f"MediGuide_Report_{time.strftime('%Y%m%d_%H%M')}.pdf",
                               mime="application/pdf", use_container_width=True)
        except Exception as exc:  # the PDF must never break the dashboard
            c1.warning(f"PDF could not be created ({type(exc).__name__}).")
        with c2.popover("View raw JSON", icon=":material/data_object:", use_container_width=True):
            st.json(a)
        c3.button("New assessment", icon=":material/add_comment:", on_click=new_chat, use_container_width=True,
                  key="res_new")
        st.warning("This is educational information, **not a medical diagnosis**. Please consult a qualified "
                   "healthcare professional.", icon=":material/medical_information:")


def page_chat():
    s = st.session_state

    # ---- bottom chat bar (pinned) — handled first so it can update the steps
    placeholders = {
        "symptoms": "Describe your symptoms… e.g. “sore throat and fever since yesterday”",
        "timing": "Add a note, or use the options above…",
        "about": "Type your age and gender, e.g. “32, female”",
        "context": "Anything else? e.g. “I have asthma” — or press Analyse",
        "done": "Ask a follow-up question about your guidance…",
    }
    msg = st.chat_input(placeholders[s.stage], accept_audio=True, key="chatbar")
    if msg:
        handle_chat_input(msg)

    stage_idx = STAGES.index(s.stage)
    with st.container(key="chatwrap"):
        html(chat_welcome())
        html(chat_stepper(stage_idx))

        # ---- Step 1: symptoms ------------------------------------------------
        with st.chat_message("assistant", avatar=BOT):
            bot_say("Hi there 👋 What symptoms are you experiencing?",
                    "Pick everything that applies, add your own words — or just type or speak in the bar below.")
            if s.stage == "symptoms":
                with st.container(key="step_symptoms"):
                    st.multiselect("Symptoms", COMMON_SYMPTOMS, **bind("symptoms"),
                                   placeholder="Search or pick symptoms…", label_visibility="collapsed")
                    st.text_area("Describe it in your own words (optional)", **bind("other"), height=80,
                                 placeholder="e.g. a dull ache behind my eyes that gets worse in the evening…")
                    step_error()
                    c1, c2 = st.columns([1, 2.2], vertical_alignment="center")
                    c1.button("Continue", type="primary", icon=":material/arrow_forward:",
                              on_click=next_from_symptoms, use_container_width=True, key="go_timing")
                    with c2:
                        ex = st.columns(3)
                        icons = {"Common cold": ":material/sick:", "Fever & cough": ":material/thermometer:",
                                 "Chest pain": ":material/cardiology:"}
                        for col, name in zip(ex, EXAMPLE_SCENARIOS):
                            col.button(name, icon=icons[name], on_click=load_example, args=(name,),
                                       use_container_width=True, key=f"ex_{name}",
                                       help="Fill in this example scenario")
        if stage_idx > 0:
            user_bubble(build_symptoms_text(collect_form()), "symptoms")

        # ---- Step 2: timing & severity --------------------------------------
        if stage_idx >= 1:
            with st.chat_message("assistant", avatar=BOT):
                bot_say("Thanks. How long have you had them, and how bad are they?")
                if s.stage == "timing":
                    with st.container(key="step_timing"):
                        c1, c2 = st.columns([1, 1.25])
                        c1.selectbox("Duration", DURATIONS, **bind("duration"))
                        with c2:
                            st.slider("Severity (1 = mild, 10 = worst)", 1, 10, **bind("severity"))
                        label, color = severity_label(s.f_severity)
                        html(severity_meter(s.f_severity, label, color))
                        st.button("Continue", type="primary", icon=":material/arrow_forward:",
                                  on_click=go_to_stage, args=("about",), key="go_about")
        if stage_idx > 1:
            label, _ = severity_label(s.f_severity)
            user_bubble(f"{s.f_duration} · severity **{s.f_severity}/10** ({label})", "timing")

        # ---- Step 3: about you ------------------------------------------------
        if stage_idx >= 2:
            with st.chat_message("assistant", avatar=BOT):
                bot_say("A little about you", "Age and gender help me tailor the guidance safely.")
                if s.stage == "about":
                    with st.container(key="step_about"):
                        c1, c2, c3 = st.columns(3)
                        c1.text_input("Age", **bind("age"), placeholder="e.g. 32", max_chars=3)
                        c2.selectbox("Gender", GENDERS, **bind("gender"))
                        c3.selectbox("Reply language", list(LANGUAGES), **bind("language"), format_func=lang_label)
                        step_error()
                        st.button("Continue", type="primary", icon=":material/arrow_forward:",
                                  on_click=next_from_about, key="go_context")
        if stage_idx > 2:
            user_bubble(f"{s.f_gender}, {s.f_age} years · replies in {s.f_language}", "about")

        # ---- Step 4: optional details -------------------------------------------
        if stage_idx >= 3:
            with st.chat_message("assistant", avatar=BOT):
                bot_say("Anything else I should know?",
                        "Optional — existing conditions, medicines or other notes make the guidance more accurate.")
                if s.stage == "context":
                    with st.container(key="step_context"):
                        st.text_area("Existing medical conditions", **bind("conditions"), height=68,
                                     placeholder="e.g. asthma, diabetes, high blood pressure, pregnancy…")
                        c1, c2 = st.columns(2)
                        c1.text_area("Current medications", **bind("meds"), height=68,
                                     placeholder="e.g. inhaler, metformin 500 mg…")
                        c2.text_area("Additional notes", **bind("notes"), height=68,
                                     placeholder="e.g. temperature 38.5°C, recent travel…")
                        step_error()
                        b1, b2 = st.columns([2, 1])
                        b1.button("Analyse my symptoms", type="primary", icon=":material/neurology:",
                                  on_click=start_analysis, use_container_width=True, key="analyze_btn")
                        b2.button("Skip & analyse", icon=":material/fast_forward:", on_click=skip_details,
                                  use_container_width=True, key="skip_btn")
        if stage_idx > 3:
            extra = [x for x in [s.f_conditions, s.f_meds, s.f_notes] if x.strip()]
            user_bubble(" · ".join(extra) if extra else "No extra details", "context")

        # ---- Step 5: results + follow-up chat ------------------------------------
        if s.stage == "done":
            with st.chat_message("assistant", avatar=BOT):
                if s.run_pending:
                    s.run_pending = False
                    run_pipeline(collect_form())
                if s.result:
                    render_results(s.result)

            res = s.result
            if res and res.get("assessment"):
                for turn in res["chat"]:
                    with st.chat_message(turn["role"], avatar=BOT if turn["role"] == "assistant" else None):
                        st.markdown(turn["content"])
                if s.pending_q:
                    q, s.pending_q = s.pending_q, None
                    with st.chat_message("user"):
                        st.markdown(q)
                    with st.chat_message("assistant", avatar=BOT):
                        try:
                            reply = st.write_stream(stream_followup(get_llm(), res["assessment"]["summary"],
                                                                    res["chat"], q, res["language"]))
                            reply = reply if isinstance(reply, str) else "".join(map(str, reply))
                        except Exception as exc:
                            reply = f"⚠️ {friendly_error(exc)}"
                            st.markdown(reply)
                    res["chat"] += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
                with st.chat_message("assistant", avatar=BOT):
                    bot_say("Any questions about this?", "Ask me anything below — or tap a suggestion.")
                    sug = st.columns(3)
                    for col, q in zip(sug, ["What can I do at home?", "When should I see a doctor?",
                                            "What should I tell my doctor?"]):
                        col.button(q, key=f"sug_{q}", on_click=ask_suggestion, args=(q,), use_container_width=True)

    if s.scroll:
        s.scroll = False
        scroll_to_bottom()


# ==========================================================================
# 4) HISTORY PAGE
# ==========================================================================
def page_history():
    rows = list_assessments()
    html(section_title("Patient session history", "Your past assessments",
                          "Saved locally in SQLite — open any assessment to see the full guidance again.", "history"))
    if not rows:
        html(empty_state("history", "No assessments yet", "Your first assessment will appear here."))
        return
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    chosen = c1.multiselect("Filter by urgency", list(URGENCY_LEVELS), default=list(URGENCY_LEVELS))
    c2.metric("Saved assessments", len(rows))
    with c3.popover("Clear history", icon=":material/delete:", use_container_width=True):
        st.write("This permanently deletes all saved assessments.")
        if st.button("Yes, delete everything", type="primary"):
            clear_history()
            st.rerun()
    st.write("")
    for row in rows:
        if row.get("urgency") not in chosen:
            continue
        a, b = st.columns([6, 1], vertical_alignment="center")
        a.markdown(history_card(row), unsafe_allow_html=True)
        b.button("Open", key=f"open_{row['id']}", icon=":material/open_in_new:", on_click=open_history_item,
                 args=(row,), use_container_width=True)


# ==========================================================================
# 5) ANALYTICS PAGE
# ==========================================================================
def page_analytics():
    import plotly.express as px
    import plotly.graph_objects as go

    html(section_title("Analytics", "Assessment insights",
                          "A small dashboard built from your saved session history.", "chart-pie"))
    df = history_dataframe()
    if df.empty:
        html(empty_state("chart-pie", "Nothing to analyse yet", "Charts appear after your first assessment."))
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total assessments", len(df))
    k2.metric("Average severity", f"{df['severity'].mean():.1f}/10")
    k3.metric("High + emergency", int(df["urgency"].isin(["HIGH", "EMERGENCY"]).sum()))
    k4.metric("Cache hit rate", f"{df['cached'].mean() * 100:.0f}%",
              help=f"Average response time: {df['elapsed'].mean():.2f}s")
    st.write("")

    order = list(URGENCY_LEVELS)
    colors = {k: v["color"] for k, v in URGENCY_LEVELS.items()}
    c1, c2 = st.columns([1, 1.4], gap="medium")
    with c1, st.container(key="card_chart_urgency"):
        html(card_head("chart-pie", "Urgency distribution", "Share of assessments per level"))
        counts = df["urgency"].value_counts().reindex(order).fillna(0)
        counts = counts[counts > 0]
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.64, sort=False,
                               marker=dict(colors=[colors[l] for l in counts.index], line=dict(width=0)),
                               textinfo="percent", textfont=dict(color="#fff", size=12)))
        fig.add_annotation(text=f"<b>{len(df)}</b><br><span style='font-size:11px'>assessments</span>",
                           showarrow=False, font=dict(size=22))
        st.plotly_chart(plotly_layout(fig), use_container_width=True, config={"displayModeBar": False})
    with c2, st.container(key="card_chart_time"):
        html(card_head("activity", "Assessments over time", "Daily count by urgency", "sky"))
        t = (df.assign(day=df["created_at"].dt.strftime("%d %b"))
             .groupby(["day", "urgency"]).size().reset_index(name="n"))
        fig = px.bar(t, x="day", y="n", color="urgency", color_discrete_map=colors,
                     category_orders={"urgency": order})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=.55, xaxis_title=None, yaxis_title=None, legend_title=None,
                          xaxis_type="category")
        st.plotly_chart(plotly_layout(fig), use_container_width=True, config={"displayModeBar": False})

    c3, c4 = st.columns([1.4, 1], gap="medium")
    with c3, st.container(key="card_chart_symptoms"):
        html(card_head("thermometer", "Most reported symptoms", "Across all assessments", "rose"))
        sym = df["symptoms"].explode().dropna().value_counts().head(8).sort_values()
        if sym.empty:
            st.caption("No listed symptoms yet.")
        else:
            fig = go.Figure(go.Bar(x=sym.values, y=sym.index, orientation="h",
                                   marker=dict(color=sym.values, colorscale=[[0, "#A5B4FC"], [1, "#4F46E5"]])))
            st.plotly_chart(plotly_layout(fig, 300), use_container_width=True, config={"displayModeBar": False})
    with c4, st.container(key="card_chart_sev"):
        html(card_head("gauge", "Severity vs. urgency", "Average severity per level", "teal"))
        sv = df.groupby("urgency")["severity"].mean().reindex(order).dropna()
        fig = go.Figure(go.Bar(x=sv.index, y=sv.values, marker=dict(color=[colors[l] for l in sv.index]),
                               text=[f"{v:.1f}" for v in sv.values], textposition="outside"))
        fig.update_yaxes(range=[0, 11])
        st.plotly_chart(plotly_layout(fig, 300), use_container_width=True, config={"displayModeBar": False})


# ==========================================================================
# 6) HOW IT WORKS PAGE
# ==========================================================================
def page_how():
    s = st.session_state
    html(section_title("Under the hood", "How MediGuide AI works",
                          "Every LangChain concept from the brief — and where to find it in the code.", "workflow"))
    html(flow_steps([
        ("clipboard-list", "Chat input", "app.py"), ("file-text", "Prompt inputs", "PromptTemplate"),
        ("database", "Select cache", "set_llm_cache"), ("cpu", "LLM + chain", "ChatOpenAI · LLMChain"),
        ("rocket", "Run chain", "chain.invoke"), ("list-checks", "Parse JSON", "utils.py"),
        ("activity", "Stream", "llm.stream()"), ("chart-pie", "Results card", "st.metric · tabs"),
    ]))
    st.write("")
    html(concept_cards([
        ("cpu", "ChatOpenAI", "The OpenAI chat model, created with your session key — never hard-coded.", "chains.py"),
        ("file-text", "PromptTemplate", "Reusable single-string template that turns your answers into a patient profile.", "prompts.py"),
        ("message-circle-question", "ChatPromptTemplate", "System (safety rules + JSON schema) + Human (patient data).", "prompts.py"),
        ("bot", "System / Human / AI", "Roles, user input and conversation memory — used by the follow-up chat.", "chains.py"),
        ("workflow", "LLMChain", "Reusable prompt → model pipeline that produces the structured assessment.", "chains.py"),
        ("list-checks", "Structured JSON", "Exact schema, fence-stripping and try/except parsing — never crashes.", "utils.py"),
        ("activity", "Streaming", "<code>llm.stream()</code> chunks rendered live with <code>st.write_stream</code>.", "chains.py"),
        ("database", "Caching", "InMemoryCache &amp; SQLiteCache, switchable in the sidebar.", "cache_manager.py"),
    ]))

    html(section_title("Caching", "InMemoryCache vs SQLiteCache",
                          "Run the same prompt twice — the second call is served from the cache.", "database"))
    c1, c2 = st.columns([1, 1.15], gap="medium")
    with c1:
        st.table(pd.DataFrame(CACHE_COMPARISON).set_index("Feature"))
        st.caption("Register a cache once with `set_llm_cache(...)`; LangChain checks it automatically "
                   "before every model call.")
    with c2, st.container(key="card_bench"):
        html(card_head("zap", "Live cache benchmark", f"Current setting: {s.cache_mode}"))
        if st.button("Run benchmark (2 identical calls)", type="primary", icon=":material/speed:",
                     use_container_width=True):
            configure_cache(s.cache_mode)
            chain = LLMChain(llm=get_llm(), prompt=PromptTemplate.from_template(
                "In one short sentence, explain what {topic} means for patients. (benchmark id {nonce})"))
            inputs = {"topic": "staying hydrated", "nonce": uuid.uuid4().hex[:8]}
            times = []
            try:
                with st.spinner("Calling the model twice…"):
                    for _ in range(2):
                        t0 = time.perf_counter()
                        chain.invoke(inputs)
                        times.append(time.perf_counter() - t0)
                s.bench = {"mode": s.cache_mode, "t1": times[0], "t2": times[1]}
            except Exception as exc:
                st.error(friendly_error(exc))
        if s.bench:
            b = s.bench
            html(bench_bars([("1st call (miss)", b["t1"], "linear-gradient(90deg,#F97316,#EF4444)"),
                                ("2nd call", b["t2"], "linear-gradient(90deg,#10B981,#0EA5A4)")]))
            if b["mode"] == "Off":
                st.info("Cache is **Off**, so both calls hit the API. Turn the cache on in the sidebar settings and "
                        "run again to see the difference.", icon=":material/info:")
            else:
                speed = b["t1"] / max(b["t2"], 1e-4)
                st.success(f"**{speed:,.0f}× faster** on the second call using **{b['mode']}**.",
                           icon=":material/bolt:")

    html(section_title("Messages", "System, Human & AI messages",
                          "How a conversation is represented when it is sent to the model.", "bot"))
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1, st.container(key="card_msgs"):
        html(card_head("bot", "Live conversation demo", "SystemMessage → HumanMessage → AIMessage → HumanMessage"))
        if st.button("Send this conversation to the model", icon=":material/play_arrow:", use_container_width=True):
            try:
                with st.spinner("Asking the model…"):
                    s.demo = demo_messages(get_llm(), s.language)
            except Exception as exc:
                st.error(friendly_error(exc))
        if s.demo:
            kinds = {"system": ("system", "SYSTEM"), "human": ("human", "HUMAN"), "ai": ("ai", "AI")}
            for m in s.demo["messages"]:
                kind, role = kinds.get(m.type, ("human", m.type.upper()))
                text = m.content if kind != "system" else m.content[:260] + " …"
                html(message_bubble(kind, role, text))
            html(message_bubble("reply", "AI REPLY", s.demo["reply"]))
        else:
            st.caption("Press the button to see the four messages and the model's reply.")
    with c2:
        with st.expander("PromptTemplate (patient profile)", expanded=True, icon=":material/description:"):
            st.code(PATIENT_PROFILE_TEMPLATE.template, language="text")
        with st.expander("Rendered with your current answers", icon=":material/preview:"):
            preview = collect_form()
            preview["symptoms_text"] = preview["symptoms_text"] or "(none yet)"
            preview["age"] = preview["age"] or "(not set)"
            st.code(build_patient_profile(preview), language="text")


# ==========================================================================
# Router
# ==========================================================================
if not st.session_state.connected:
    render_gate()
else:
    html(gate_background())          # same animated background as the Connect screen
    render_sidebar()
    page = st.session_state.page
    if page == "history":
        page_history()
    elif page == "analytics":
        page_analytics()
    elif page == "how":
        page_how()
    else:
        page_chat()
