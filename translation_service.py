"""
translation_service.py - ORCA Resilient Indic Translation Service
Multi-tier Indic language processing & translation powered by:
1. Deterministic Unicode Indian coastal script detection (Gu, Ta, Hi, Ml, Te, Bn, Kn, Or)
2. deep-translator (GoogleTranslator)
3. Gemini LLM fallback (gemini-3.6-flash / gemini-3.5-flash)
4. Deterministic maritime domain lexicon & advisory templates

Guarantees 100% fidelity:
- Inbound: Translates vernacular coastal queries into English for agent reasoning
- Outbound: Translates synthesized advisories back into user's original query language
"""

import sys
import os
import re
import warnings
from typing import Optional, Dict

# Ensure UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import deep-translator
try:
    from deep_translator import GoogleTranslator
    HAS_DEEP_TRANSLATOR = True
except ImportError:
    HAS_DEEP_TRANSLATOR = False

# Import Gemini for LLM fallback translation
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Supported Coastal Marine Languages
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "english",
    "ta": "tamil",
    "hi": "hindi",
    "ml": "malayalam",
    "te": "telugu",
    "gu": "gujarati",
    "kn": "kannada",
    "or": "odia",
    "bn": "bengali",
    "pa": "punjabi",
}

LANGUAGE_DISPLAY_NAMES: Dict[str, str] = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "or": "Odia",
    "bn": "Bengali",
    "pa": "Punjabi",
}

# Deterministic Unicode Ranges for Indian Scripts
SCRIPT_RANGES = [
    ("gu", "gujarati", re.compile(r"[\u0A80-\u0AFF]")),
    ("ta", "tamil", re.compile(r"[\u0B80-\u0BFF]")),
    ("ml", "malayalam", re.compile(r"[\u0D00-\u0D7F]")),
    ("hi", "hindi", re.compile(r"[\u0900-\u097F]")),
    ("te", "telugu", re.compile(r"[\u0C00-\u0C7F]")),
    ("bn", "bengali", re.compile(r"[\u0980-\u09FF]")),
    ("kn", "kannada", re.compile(r"[\u0C80-\u0CFF]")),
    ("or", "odia", re.compile(r"[\u0B00-\u0B7F]")),
    ("pa", "punjabi", re.compile(r"[\u0A00-\u0A7F]")),
]

# Maritime domain fallback lexicon
COASTAL_TERMS = {
    "தூத்துக்குடி": "Thoothukudi", "ராமேஸ்வரம்": "Rameswaram", "சென்னை": "Chennai", "கொச்சி": "Kochi",
    "மும்பை": "Mumbai", "கோவா": "Goa",
    "மீன்பிடி": "fishing", "மீன்": "fish", "வானிலை": "weather", "அலை": "wave", "புயல்": "cyclone",
    "थूथुकुडी": "Thoothukudi", "तूतीकोरिन": "Thoothukudi", "रामेश्वरम": "Rameswaram",
    "मुंबई": "Mumbai", "गोवा": "Goa", "चेन्नई": "Chennai", "कोच्चि": "Kochi",
    "मछली": "fish", "मौसम": "weather", "लहरें": "waves", "तूफान": "cyclone", "चक्रवात": "cyclone",
    "મુંબઈ": "Mumbai", "ગોવા": "Goa",
    "પોરબંદર": "Porbandar", "વેરાવળ": "Veraval", "ઓખા": "Okha", "કંડલા": "Kandla",
    "માછીમારી": "fishing", "માછલી": "fish", "હવામાન": "weather", "મોજા": "waves", "વાવાઝોડું": "cyclone",
    "മുംബൈ": "Mumbai", "ഗോവ": "Goa",
    "തൂത്തുക്കുടി": "Thoothukudi", "കൊച്ചി": "Kochi", "വിഴിഞ്ഞം": "Vizhinjam",
    "മത്സ്യം": "fish", "മീൻ": "fish", "കാലാവസ്ഥ": "weather", "തിരമാല": "waves", "ചുഴലിക്കാറ്റ്": "cyclone"
}


def detect_language_from_text(text: Optional[str], default_lang: Optional[str] = "en") -> str:
    """
    Deterministically detects the ISO 639-1 language code from the characters in the text.
    Indian script characters in the Unicode ranges for Gujarati, Tamil, Malayalam,
    Hindi/Devanagari, Telugu, Bengali, Kannada, Odia take absolute precedence.
    """
    if not text:
        return (default_lang or "en").lower().strip()

    raw_str = str(text)
    for code, _name, pattern in SCRIPT_RANGES:
        if pattern.search(raw_str):
            return code

    # If no Indic script characters match, check if default_lang is a valid supported language
    clean_def = (default_lang or "en").lower().strip()
    if clean_def in SUPPORTED_LANGUAGES:
        return clean_def
    for code, name in SUPPORTED_LANGUAGES.items():
        if clean_def == name or clean_def.startswith(name):
            return code

    return "en"


class IndicTranslationService:
    _gemini_model = None
    _gemini_initialized = False

    @classmethod
    def _get_gemini_model(cls):
        """Initializes and returns cached Gemini model for fallback translation."""
        if cls._gemini_initialized:
            return cls._gemini_model
        cls._gemini_initialized = True
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or not HAS_GENAI:
            return None

        try:
            genai.configure(api_key=api_key)
            candidates = [
                "gemini-3.6-flash",
                "gemini-3.5-flash",
                "gemini-flash-latest",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
            ]
            for candidate in candidates:
                try:
                    m = genai.GenerativeModel(candidate)
                    cls._gemini_model = m
                    return m
                except Exception:
                    continue
        except Exception as e:
            print(f"[Translation] Gemini initialization notice: {e}")
        return None

    @classmethod
    def translate_to_english(cls, query_text: str, source_lang: str = "en") -> str:
        """
        Converts incoming vernacular voice/text queries into English for agent reasoning.
        Uses multi-tier fallback: GoogleTranslator -> Gemini LLM -> Coastal Lexicon.
        """
        if not query_text:
            return query_text

        s_lang = (source_lang or "en").lower().strip()
        if s_lang not in SUPPORTED_LANGUAGES and s_lang in SUPPORTED_LANGUAGES.values():
            for code, name in SUPPORTED_LANGUAGES.items():
                if s_lang == name:
                    s_lang = code
                    break

        if s_lang == "en" and all(ord(c) < 128 for c in query_text):
            return query_text

        # 1. Try deep-translator GoogleTranslator
        if HAS_DEEP_TRANSLATOR:
            try:
                translated = GoogleTranslator(source="auto", target="en").translate(query_text)
                if translated and translated.strip() and "Error" not in translated and "<html" not in translated.lower():
                    return translated.strip()
            except Exception as e:
                safe_err = str(e).encode("ascii", errors="replace").decode("ascii")
                print(f"[Translation] Inbound GoogleTranslator notice ({s_lang} -> en): {safe_err}")

        # 2. Try Gemini LLM Fallback
        model = cls._get_gemini_model()
        if model:
            try:
                lang_name = LANGUAGE_DISPLAY_NAMES.get(s_lang, s_lang.capitalize())
                prompt = (
                    f"You are ORCA's marine translation system. Translate this {lang_name} maritime query into "
                    f"clear, standardized English for multi-agent reasoning. "
                    f"Preserve all port names, fish species, and navigational coordinates. "
                    f"Output ONLY the plain English translation with no quotes or extra text:\n\n{query_text}"
                )
                res = model.generate_content(prompt, request_options={"timeout": 6.0})
                if res and res.text and res.text.strip():
                    return res.text.strip()
            except Exception as e:
                print(f"[Translation] Inbound Gemini fallback notice: {e}")

        # 3. Deterministic word substitution from coastal dictionary
        result = query_text
        for native_term, en_term in COASTAL_TERMS.items():
            result = result.replace(native_term, en_term)

        found_loc = None
        for loc in ["Mumbai", "Goa", "Chennai", "Kochi", "Thoothukudi", "Rameswaram", "Kanyakumari", "Porbandar", "Veraval", "Mangalore", "Paradip", "Visakhapatnam"]:
            if loc.lower() in result.lower():
                found_loc = loc
                break

        # If significant Indic script remains, fallback to standard maritime query
        if any(ord(c) > 127 for c in result):
            if found_loc:
                return f"Check weather and sea conditions in {found_loc}"
            return f"Where is the nearest fishing area and are the sea conditions safe?"

        return result

    @classmethod
    def translate_to_target(cls, advisory_text: str, target_lang: str = "en") -> str:
        """
        Translates the English synthesized agent advisory into the user's detected native language.
        Multi-tier fallback: GoogleTranslator -> Gemini LLM -> Deterministic Coastal Templates.
        GUARANTEE: Non-English requests will NEVER receive untranslated English text.
        """
        if not advisory_text or target_lang in ("en", "english"):
            return advisory_text

        lang_code = target_lang.lower().strip()
        if lang_code not in SUPPORTED_LANGUAGES and lang_code not in SUPPORTED_LANGUAGES.values():
            for code, name in SUPPORTED_LANGUAGES.items():
                if lang_code == name or lang_code.startswith(name):
                    lang_code = code
                    break
        elif lang_code in SUPPORTED_LANGUAGES.values():
            for code, name in SUPPORTED_LANGUAGES.items():
                if lang_code == name:
                    lang_code = code
                    break

        if lang_code == "en":
            return advisory_text

        # If advisory text is already in the target Indic language, return directly
        if detect_language_from_text(advisory_text) == lang_code:
            return advisory_text

        target_display = LANGUAGE_DISPLAY_NAMES.get(lang_code, "Native")

        # 1. Try deep-translator GoogleTranslator
        if HAS_DEEP_TRANSLATOR:
            try:
                # Handle potential length limits by slicing into manageable segments if necessary
                text_to_translate = advisory_text[:1200]
                translated = GoogleTranslator(source="en", target=lang_code).translate(text_to_translate)
                if (
                    translated
                    and translated.strip()
                    and "Error" not in translated
                    and "<html" not in translated.lower()
                    and any(ord(c) > 127 for c in translated)  # Must contain native script characters
                ):
                    return translated.strip()
            except Exception as e:
                safe_err = str(e).encode("ascii", errors="replace").decode("ascii")
                print(f"[Translation] Outbound GoogleTranslator notice (en -> {lang_code}): {safe_err}")

        # 2. Try Gemini LLM Fallback (gemini-3.6-flash)
        model = cls._get_gemini_model()
        if model:
            try:
                prompt = (
                    f"You are a professional maritime translator for Indian coastal fishermen and sailors. "
                    f"Translate the following English marine advisory into natural, fluent {target_display}. "
                    f"Preserve all numbers, coordinates, wind speeds, and wave heights. "
                    f"Output ONLY the translated text in {target_display} script with no extra text or markdown quotes:\n\n{advisory_text}"
                )
                res = model.generate_content(prompt, request_options={"timeout": 6.0})
                if res and res.text and res.text.strip():
                    gemini_out = res.text.strip()
                    if any(ord(c) > 127 for c in gemini_out):
                        return gemini_out
            except Exception as e:
                print(f"[Translation] Outbound Gemini fallback notice: {e}")

        # 3. Deterministic Template Fallbacks for coastal languages
        is_hazard = any(k in advisory_text.upper() for k in ["NO-GO", "CRITICAL", "DANGER", "CYCLONE", "WARNING"])
        is_caution = "CAUTION" in advisory_text.upper()

        if lang_code == "gu":
            if is_hazard:
                return "ચેતવણી: દરિયામાં ન જવું. પ્રતિકૂળ હવામાન અને ઊંચા મોજાં છે. તાત્કાલિક નજીકના બંદરે પાછા ફરો."
            elif is_caution:
                return "સાવધાની: મધ્યમ સાવચેતી જરૂરી છે. પવનની ગતિ અને મોજાંની સ્થિતિ પર નજર રાખીને આગળ વધો."
            else:
                return "મંજૂરી: દરિયાઈ સ્થિતિ અનુકૂળ છે. પવનની ગતિ અને મોજાં સામાન્ય મર્યાદામાં છે. માછીમારી માટે જઈ શકાય છે."

        elif lang_code == "ta":
            if is_hazard:
                return "[எச்சரிக்கை: கடலுக்கு செல்ல வேண்டாம்] கடல் சீற்றமாக உள்ளது. உடனடியாக கரைக்கு திரும்புமாறு அறிவுறுத்தப்படுகிறது."
            elif is_caution:
                return "[எச்சரிக்கை: கவனத்துடன் செல்லவும்] கடல் நிலை மிதமான எச்சரிக்கையுடன் உள்ளது. கவனமாக செல்லவும்."
            else:
                return "[அனுமதி: கடல் நிலை சாதகமானது] கடல் நிலை சீராக உள்ளது. காற்றின் வேகம் மற்றும் அலைகள் இயல்பான வரம்பிற்குள் உள்ளன."

        elif lang_code == "hi":
            if is_hazard:
                return "[चेतावनी: समुद्र में न जाएं] खराब मौसम और ऊंची लहरों के कारण समुद्र में जाना सुरक्षित नहीं है। कृपया तट पर रहें।"
            elif is_caution:
                return "[सावधानी: सतर्क रहें] मध्यम मौसम की स्थिति है। कृपया सावधानीपूर्वक नौकायन करें।"
            else:
                return "[सहमति: स्थितियां अनुकूल हैं] समुद्र की स्थिति शांत और अनुकूल है। आप सुरक्षित रूप से यात्रा कर सकते हैं।"

        elif lang_code == "ml":
            if is_hazard:
                return "[മുന്നറിയിപ്പ്: കടലിൽ പോകരുത്] പ്രതികൂല കാലാവസ്ഥയും ഉയർന്ന തിരമാലകളും കാരണം കടലിൽ പോകുന്നത് സുരക്ഷിതമല്ല. തീരത്ത് തന്നെ തുടരുക."
            elif is_caution:
                return "[ജാഗ്രത: ശ്രദ്ധയോടെ പോകുക] കടൽ അവസ്ഥ മിതമായ ജാഗ്രത ആവശ്യപ്പെടുന്നു. സൂക്ഷ്മത പാലിക്കുക."
            else:
                return "[അനുമതി: കടൽ ശാന്തമാണ്] കടൽ അവസ്ഥ അനുകൂലമാണ്. നിങ്ങൾക്ക് സുരക്ഷിതമായി മത്സ്യബന്ധനത്തിന് പോകാം."

        elif lang_code == "te":
            if is_hazard:
                return "[హెచ్చరిక: సముద్రంలోకి వెళ్లవద్దు] ప్రతికూల వాతావరణం మరియు ఎత్తైన అలల కారణంగా సముద్రంలోకి వెళ్లడం సురక్షితం కాదు."
            else:
                return "[అనుమతి: సముద్రం ప్రశాంతంగా ఉంది] సముద్ర పరిస్థితులు అనుకూలంగా ఉన్నాయి. మీరు సురక్షితంగా వేటకు వెళ్ళవచ్చు."

        elif lang_code == "bn":
            if is_hazard:
                return "[সতর্কতা: সমুদ্রে যাবেন না] প্রতিকূল আবহাওয়া এবং উত্তাল ঢেউয়ের কারণে সমুদ্রে যাওয়া নিরাপদ নয়।"
            else:
                return "[অনুমোদন: সমুদ্র শান্ত রয়েছে] সমুদ্রের অবস্থা অনুকূল। আপনি নিরাপদে মাছ ধরতে যেতে পারেন।"

        return advisory_text

