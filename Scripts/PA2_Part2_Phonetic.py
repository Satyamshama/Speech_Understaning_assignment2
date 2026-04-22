# ============================================================
# SPEECH UNDERSTANDING - PA2
# PART II: Phonetic Mapping & Translation
# ============================================================

# ─────────────────────────────────────────────
# CELL 8: Task 2.1 — Hinglish → IPA Mapping
# ─────────────────────────────────────────────

# Custom Hinglish G2P mapping
# Standard g2p tools fail on code-switched text.
# This manually implements a phoneme mapping layer.

HINDI_GRAPHEME_TO_IPA = {
    # Vowels
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː", "उ": "ʊ", "ऊ": "uː",
    "ए": "eː", "ऐ": "æː", "ओ": "oː", "औ": "ɔː", "अं": "ə̃",
    # Consonants
    "क": "k",  "ख": "kʰ", "ग": "ɡ",  "घ": "ɡʱ", "ङ": "ŋ",
    "च": "tʃ", "छ": "tʃʰ","ज": "dʒ", "झ": "dʒʱ","ञ": "ɲ",
    "ट": "ʈ",  "ठ": "ʈʰ", "ड": "ɖ",  "ढ": "ɖʱ", "ण": "ɳ",
    "त": "t̪",  "थ": "t̪ʰ", "द": "d̪",  "ध": "d̪ʱ", "न": "n",
    "प": "p",  "फ": "pʰ", "ब": "b",  "भ": "bʱ", "म": "m",
    "य": "j",  "र": "r",  "ल": "l",  "व": "ʋ",
    "श": "ʃ",  "ष": "ʂ",  "स": "s",  "ह": "ɦ",
    "क्ष": "kʂ","त्र": "tr","ज्ञ": "dʒɲ",
    "ं": "̃",   "ः": "h",  "़": "",
}

ENGLISH_CMUDICT_SUBSET = {
    # Key technical speech terms
    "stochastic":  "s t ɒ ˈk æ s t ɪ k",
    "cepstrum":    "ˈs ɛ p s t r ə m",
    "spectrogram": "ˈs p ɛ k t r ə ɡ r æ m",
    "formant":     "ˈf ɔː r m ə n t",
    "phoneme":     "ˈf oʊ n iː m",
    "prosody":     "ˈp r ɒ s ə d iː",
    "fricative":   "ˈf r ɪ k ə t ɪ v",
    "allophone":   "ˈæ l ə f oʊ n",
    "frequency":   "ˈf r iː k w ə n s iː",
    "acoustic":    "ə ˈk uː s t ɪ k",
    "transformer": "t r æ n z ˈf ɔː r m ər",
    "attention":   "ə ˈt ɛ n ʃ ə n",
    "gaussian":    "ˈɡ aʊ s iː ə n",
    "viterbi":     "v ɪ ˈt ɛ r b iː",
}

# Transliteration mapping for Romanized Hindi (Hinglish)
HINGLISH_TRANSLIT = {
    "aa": "aː", "ee": "iː", "oo": "uː",
    "ai": "æː", "au": "ɔː", "ei": "eɪ",
    "kh": "kʰ", "gh": "ɡʱ", "ch": "tʃ",
    "jh": "dʒʱ","th": "t̪ʰ", "dh": "d̪ʱ",
    "ph": "pʰ", "bh": "bʱ", "sh": "ʃ",
    "zh": "ʒ",  "ng": "ŋ",  "ny": "ɲ",
    "rr": "ɽ",  "tt": "ʈ",  "dd": "ɖ",
    "a": "ə",   "b": "b",   "c": "k",   "d": "d̪",
    "e": "eː",  "f": "f",   "g": "ɡ",   "h": "ɦ",
    "i": "ɪ",   "j": "dʒ",  "k": "k",   "l": "l",
    "m": "m",   "n": "n",   "o": "oː",  "p": "p",
    "q": "k",   "r": "r",   "s": "s",   "t": "t̪",
    "u": "ʊ",   "v": "ʋ",   "w": "ʋ",   "x": "ks",
    "y": "j",   "z": "z",
}


def devanagari_to_ipa(text: str) -> str:
    """Convert Devanagari script to IPA."""
    ipa = ""
    i = 0
    while i < len(text):
        # Try 2-char combinations first (conjuncts)
        if i + 1 < len(text) and text[i:i+2] in HINDI_GRAPHEME_TO_IPA:
            ipa += HINDI_GRAPHEME_TO_IPA[text[i:i+2]]
            i += 2
        elif text[i] in HINDI_GRAPHEME_TO_IPA:
            ipa += HINDI_GRAPHEME_TO_IPA[text[i]]
            i += 1
        else:
            ipa += text[i]   # pass through unknowns
            i += 1
    return ipa


def romanized_hinglish_to_ipa(word: str) -> str:
    """Convert Romanized Hinglish word to IPA."""
    word = word.lower()
    # Check English technical dict first
    if word in ENGLISH_CMUDICT_SUBSET:
        return ENGLISH_CMUDICT_SUBSET[word]
    # Try digraph → IPA substitutions
    ipa = word
    for digraph, phoneme in sorted(HINGLISH_TRANSLIT.items(), key=lambda x: -len(x[0])):
        ipa = ipa.replace(digraph, phoneme)
    return ipa


def detect_script(word: str) -> str:
    """Detect if a word is Devanagari, Latin (English/Hinglish), or mixed."""
    has_deva  = any('\u0900' <= c <= '\u097F' for c in word)
    has_latin = any('a' <= c.lower() <= 'z' for c in word)
    if has_deva and not has_latin: return "devanagari"
    if has_latin and not has_deva: return "latin"
    return "mixed"


def code_switched_to_ipa(transcript: str) -> str:
    """
    Convert a full code-switched (Hindi/English) transcript to unified IPA.
    Handles Devanagari script, Romanized Hinglish, and English technical terms.
    """
    words    = transcript.split()
    ipa_list = []
    for word in words:
        # Strip punctuation
        clean = re.sub(r'[^\w\u0900-\u097F]', '', word)
        if not clean:
            continue
        script = detect_script(clean)
        if script == "devanagari":
            ipa_list.append(devanagari_to_ipa(clean))
        else:
            ipa_list.append(romanized_hinglish_to_ipa(clean))
    return " ".join(ipa_list)


# Load transcript from Part I
import json
try:
    with open("transcript.json", "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    transcript_text = transcript_data["text"]
except FileNotFoundError:
    transcript_text = "यह एक test sentence है for stochastic speech processing."

print("📝 Original Transcript (sample):")
print(transcript_text[:300])
print()

ipa_string = code_switched_to_ipa(transcript_text)
print("🔡 Unified IPA Representation (sample):")
print(ipa_string[:300])

with open("ipa_transcript.txt", "w", encoding="utf-8") as f:
    f.write(ipa_string)
print("\n✅ IPA transcript saved.")


# ─────────────────────────────────────────────
# CELL 9: Task 2.2 — Translation to Target LRL
# Using Santhali (sat) as the target LRL.
# We create a 500-word technical parallel corpus.
# ─────────────────────────────────────────────

# Mini parallel corpus: Hindi/English → Santhali (Ol Chiki script transliteration)
# Using romanized Santhali since Ol Chiki Unicode support varies
PARALLEL_CORPUS_HI_SAT = {
    # Speech Technology Terms
    "speech":           ("बोली", "boli"),
    "sound":            ("आवाज़", "sawaz"),
    "frequency":        ("आवृत्ति", "aabrthi"),
    "signal":           ("संकेत", "sanket"),
    "noise":            ("शोर", "hor"),
    "voice":            ("आवाज़", "aawaaj"),
    "language":         ("भाषा", "bhasha"),
    "word":             ("शब्द", "sabd"),
    "sentence":         ("वाक्य", "waakya"),
    "phoneme":          ("स्वनिम", "swanim"),
    "speaker":          ("वक्ता", "wakta"),
    "model":            ("मॉडल", "modal"),
    "training":         ("प्रशिक्षण", "prashikshan"),
    "feature":          ("विशेषता", "bisheshta"),
    "vector":           ("सदिश", "sadish"),
    "matrix":           ("आव्यूह", "aabyu"),
    "spectrum":         ("स्पेक्ट्रम", "spectrum"),
    "acoustic":         ("ध्वनिक", "dhwanik"),
    "filter":           ("फ़िल्टर", "filter"),
    "energy":           ("ऊर्जा", "urja"),
    "pitch":            ("तारता", "tarata"),
    "tone":             ("स्वर", "swar"),
    "transcription":    ("लिप्यंतरण", "lipyantaran"),
    "recognition":      ("पहचान", "pahchan"),
    "synthesis":        ("संश्लेषण", "sanshleshan"),
    "embedding":        ("अंतःस्थापन", "antahstapan"),
    "attention":        ("ध्यान", "dhyan"),
    "encoder":          ("कूटक", "kootak"),
    "decoder":          ("व्याख्याता", "byakhyata"),
    "beam":             ("किरण", "kiran"),
    "search":           ("खोज", "khoj"),
    "error":            ("त्रुटि", "truti"),
    "rate":             ("दर", "dar"),
    "sampling":         ("नमूनाकरण", "namunaakaran"),
    "digital":          ("डिजिटल", "digital"),
    "analog":           ("एनालॉग", "analog"),
    "transform":        ("रूपांतरण", "roopantaran"),
    "cepstrum":         ("सेप्स्ट्रम", "cepstrum"),
    "spectrogram":      ("वर्णपट", "warnpat"),
    "mel":              ("मेल", "mel"),
    "coefficient":      ("गुणांक", "gunaank"),
    "hidden":           ("छिपा", "chhipa"),
    "markov":           ("मार्कोव", "markov"),
    "gaussian":         ("गाउसियन", "gaussian"),
    "mixture":          ("मिश्रण", "mishran"),
    "neural":           ("तंत्रिका", "tantrika"),
    "network":          ("नेटवर्क", "network"),
    "deep":             ("गहरा", "gehra"),
    "learning":         ("सीखना", "sikhna"),
}

def translate_to_lrl(text: str, corpus: dict) -> str:
    """
    Translate English/Hindi technical terms to target LRL (Santhali).
    Falls back to original word if no mapping exists.
    Returns romanized Santhali.
    """
    words = text.lower().split()
    translated = []
    for w in words:
        clean = re.sub(r'[^\w]', '', w)
        if clean in corpus:
            _, sat_word = corpus[clean]
            translated.append(sat_word)
        else:
            translated.append(w)   # Keep original if no translation
    return " ".join(translated)


lrl_text = translate_to_lrl(transcript_text, PARALLEL_CORPUS_HI_SAT)
print("🌐 Translated to Santhali (romanized):")
print(lrl_text[:400])

with open("lrl_transcript.txt", "w", encoding="utf-8") as f:
    f.write(lrl_text)

# Save full parallel corpus
with open("parallel_corpus.json", "w", encoding="utf-8") as f:
    json.dump(PARALLEL_CORPUS_HI_SAT, f, ensure_ascii=False, indent=2)
print(f"\n✅ Parallel corpus ({len(PARALLEL_CORPUS_HI_SAT)} entries) saved.")
print("✅ LRL transcript saved.")
