# ============================================================
# SPEECH UNDERSTANDING - PA2
# PART I: Robust Code-Switched Transcription (STT)
# ============================================================
# Run each cell in Google Colab (GPU recommended: T4 or A100)
# ============================================================

# ─────────────────────────────────────────────
# CELL 1: Install Dependencies
# ─────────────────────────────────────────────
"""
!pip install torch torchaudio transformers datasets
!pip install openai-whisper
!pip install phonemizer epitran
!pip install nltk scikit-learn
!pip install deepfilternet
!pip install librosa soundfile
!pip install jiwer  # for WER computation
!pip install accelerate
"""

# ─────────────────────────────────────────────
# CELL 2: Imports
# ─────────────────────────────────────────────
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json, re, os
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# CELL 3: Upload your .wav files
# ─────────────────────────────────────────────
"""
from google.colab import files
uploaded = files.upload()
# Upload: original_segment.wav, student_voice_ref.wav
"""

AUDIO_PATH = "original_segment.wav"      # Your Hindi lecture .wav
REF_VOICE  = "student_voice_ref.wav"     # Your 60-second reference voice

# ─────────────────────────────────────────────
# CELL 4: Task 1.3 — Denoising & Normalization
# (DeepFilterNet or Spectral Subtraction fallback)
# ─────────────────────────────────────────────
def spectral_subtraction(audio_np, sr, noise_frames=20, alpha=2.0):
    """
    Spectral Subtraction for noise reduction.
    Estimates noise PSD from the first `noise_frames` frames
    and subtracts it from the full spectrogram.
    """
    n_fft   = 1024
    hop_len = 256
    # STFT
    stft = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_len)
    mag, phase = np.abs(stft), np.angle(stft)

    # Estimate noise from first N frames
    noise_est = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)

    # Subtract
    mag_clean = np.maximum(mag - alpha * noise_est, 0.0)

    # Reconstruct
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_len)
    return audio_clean


def denoise_audio(input_path: str, output_path: str = "denoised.wav") -> str:
    """
    Tries DeepFilterNet first; falls back to spectral subtraction.
    Returns path to denoised file.
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        model, df_state, _ = init_df()
        audio, _ = load_audio(input_path, sr=df_state.sr())
        enhanced  = enhance(model, df_state, audio)
        save_audio(output_path, enhanced, df_state.sr())
        print("✅ DeepFilterNet denoising complete.")
    except Exception as e:
        print(f"DeepFilterNet unavailable ({e}). Using Spectral Subtraction.")
        audio_np, sr = librosa.load(input_path, sr=16000, mono=True)
        audio_clean  = spectral_subtraction(audio_np, sr)
        sf.write(output_path, audio_clean, sr)
        print("✅ Spectral Subtraction complete.")
    return output_path


denoised_path = denoise_audio(AUDIO_PATH, "denoised_segment.wav")


# ─────────────────────────────────────────────
# CELL 5: Task 1.1 — Multi-Head LID at Frame Level
# ─────────────────────────────────────────────
import torch.nn as nn

class MultiHeadLID(nn.Module):
    """
    Frame-level Language Identification Model.
    Input : MFCC features [B, T, n_mfcc]
    Output: logits for [English, Hindi] per frame
    Uses multi-head attention to capture temporal context,
    then a per-frame classifier.
    """
    def __init__(self, n_mfcc=40, d_model=128, n_heads=4, n_layers=2, n_langs=2):
        super().__init__()
        self.proj = nn.Linear(n_mfcc, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier  = nn.Linear(d_model, n_langs)  # per-frame

    def forward(self, x):
        # x: [B, T, n_mfcc]
        x = self.proj(x)                    # [B, T, d_model]
        x = self.transformer(x)             # [B, T, d_model]
        logits = self.classifier(x)         # [B, T, n_langs]
        return logits


def extract_mfcc_frames(audio_path: str, sr=16000, n_mfcc=40,
                         frame_len=0.025, hop_len=0.010):
    """Extract MFCC feature matrix from an audio file."""
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        n_fft=int(frame_len * sr),
        hop_length=int(hop_len * sr)
    )
    # delta features for better discrimination
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats  = np.concatenate([mfcc, delta, delta2], axis=0)  # [3*n_mfcc, T]
    return feats.T  # [T, features]


def generate_pseudo_labels(audio_path: str, sr=16000, hop_len=0.010):
    """
    Pseudo-label frames using energy + zero-crossing heuristic.
    Hindi frames tend to have higher ZCR in the retroflex region.
    This is a placeholder — replace with actual annotated data if available.
    Returns: np.array of shape [T] with values 0=English, 1=Hindi
    """
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    hop = int(hop_len * sr)
    zcr    = librosa.feature.zero_crossing_rate(audio, hop_length=hop)[0]
    energy = librosa.feature.rms(y=audio, hop_length=hop)[0]
    # Simple heuristic: high ZCR + moderate energy → Hindi-like
    labels = ((zcr > np.median(zcr)) & (energy > np.percentile(energy, 20))).astype(int)
    return labels


# ── Training the LID ──
def train_lid_model(audio_path: str, epochs=10, lr=1e-3):
    n_mfcc   = 40
    n_input  = n_mfcc * 3   # mfcc + delta + delta2

    model = MultiHeadLID(n_mfcc=n_input).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    feats  = extract_mfcc_frames(audio_path)        # [T, 120]
    labels = generate_pseudo_labels(audio_path)      # [T]

    # Align lengths
    T = min(len(feats), len(labels))
    feats, labels = feats[:T], labels[:T]

    X = torch.tensor(feats,  dtype=torch.float32).unsqueeze(0).to(device)  # [1,T,120]
    Y = torch.tensor(labels, dtype=torch.long).unsqueeze(0).to(device)     # [1,T]

    model.train()
    for epoch in range(epochs):
        optim.zero_grad()
        logits = model(X)                   # [1, T, 2]
        loss   = loss_fn(logits.view(-1,2), Y.view(-1))
        loss.backward()
        optim.step()
        if (epoch+1) % 2 == 0:
            preds = logits.argmax(-1).view(-1).cpu().numpy()
            f1    = f1_score(Y.view(-1).cpu().numpy(), preds, average='macro')
            print(f"  Epoch {epoch+1:2d} | Loss: {loss.item():.4f} | F1: {f1:.4f}")

    model.eval()
    return model


print("\n🔵 Training Multi-Head LID Model...")
lid_model = train_lid_model(denoised_path, epochs=20)
torch.save(lid_model.state_dict(), "lid_model.pth")
print("✅ LID model saved.")


# ── LID Inference with timestamps ──
def run_lid_inference(audio_path: str, model, hop_len=0.010):
    """
    Returns a list of (start_sec, end_sec, lang) segments.
    lang: 0=English, 1=Hindi
    """
    n_mfcc = 40
    feats  = extract_mfcc_frames(audio_path)       # [T, 120]
    X      = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(X)
    preds = logits.argmax(-1).squeeze(0).cpu().numpy()  # [T]

    # Group consecutive same-language frames into segments
    segments = []
    start_frame, cur_lang = 0, preds[0]
    for i, p in enumerate(preds[1:], 1):
        if p != cur_lang:
            segments.append({
                "start": round(start_frame * hop_len, 3),
                "end":   round(i * hop_len, 3),
                "lang":  "EN" if cur_lang == 0 else "HI"
            })
            start_frame, cur_lang = i, p
    segments.append({
        "start": round(start_frame * hop_len, 3),
        "end":   round(len(preds) * hop_len, 3),
        "lang":  "EN" if cur_lang == 0 else "HI"
    })
    return segments


lid_segments = run_lid_inference(denoised_path, lid_model)
print(f"\n📋 LID detected {len(lid_segments)} language segments:")
for s in lid_segments[:10]:
    print(f"  [{s['start']:.2f}s – {s['end']:.2f}s] → {s['lang']}")


# ─────────────────────────────────────────────
# CELL 6: Task 1.2 — N-gram LM + Constrained Whisper Decoding
# ─────────────────────────────────────────────
import whisper
from whisper.tokenizer import get_tokenizer

# ── Build N-gram LM from syllabus / technical corpus ──
TECH_TERMS = [
    "stochastic", "cepstrum", "mel-frequency", "cepstral", "spectrogram",
    "formant", "phoneme", "grapheme", "allophone", "prosody", "fricative",
    "plosive", "voiced", "unvoiced", "fundamental frequency", "pitch",
    "dynamic time warping", "hidden markov model", "gaussian mixture model",
    "connectionist temporal classification", "language model", "acoustic model",
    "beam search", "viterbi", "backpropagation", "gradient descent",
    "attention mechanism", "transformer", "wav2vec", "whisper", "speaker diarization",
    "zero crossing rate", "short time fourier transform", "filter bank",
    "inverse fourier transform", "linear predictive coding", "perceptual linear prediction"
]

def build_ngram_lm(word_list, n=2):
    """Build a simple N-gram language model from a list of technical terms."""
    ngrams = defaultdict(lambda: defaultdict(int))
    unigrams = defaultdict(int)
    for phrase in word_list:
        tokens = phrase.lower().split()
        for tok in tokens:
            unigrams[tok] += 1
        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i:i+n-1])
            next_w  = tokens[i+n-1]
            ngrams[context][next_w] += 1
    return ngrams, unigrams


ngram_lm, unigram_lm = build_ngram_lm(TECH_TERMS, n=2)
print(f"✅ N-gram LM built with {len(unigram_lm)} unique technical tokens.")


def compute_logit_bias(tokenizer, unigram_lm, boost=5.0):
    """
    Returns a dict {token_id: bias_score} for technical terms.
    These will be ADDED to Whisper logits before sampling.
    """
    bias = {}
    vocab = tokenizer.encoding._mergeable_ranks  # tiktoken vocab
    for word, count in unigram_lm.items():
        token_ids = tokenizer.encoding.encode(" " + word)
        for tid in token_ids:
            # Scale bias by log frequency of the term
            bias[tid] = bias.get(tid, 0.0) + boost * np.log1p(count)
    return bias


# ── Load Whisper with Logit Bias Hook ──
print("\n🔵 Loading Whisper-large-v3 ...")
whisper_model = whisper.load_model("large-v3", device=device)
tokenizer = get_tokenizer(multilingual=True, language="hi", task="transcribe")

logit_bias = compute_logit_bias(tokenizer, unigram_lm, boost=3.0)
print(f"✅ Logit bias computed for {len(logit_bias)} token IDs.")


class LogitBiasHook:
    """
    Registers a forward hook on Whisper's final linear layer
    to inject N-gram logit biases for technical terms.
    """
    def __init__(self, bias_dict: dict, device):
        self.bias_tensor = torch.zeros(
            whisper_model.dims.n_vocab, device=device
        )
        for tid, val in bias_dict.items():
            if tid < self.bias_tensor.shape[0]:
                self.bias_tensor[tid] = val

    def __call__(self, module, input, output):
        return output + self.bias_tensor


hook_obj    = LogitBiasHook(logit_bias, device)
hook_handle = whisper_model.decoder.ln.register_forward_hook(hook_obj)


def transcribe_with_bias(audio_path: str, language: str = "hi") -> dict:
    """Transcribe with logit bias active."""
    result = whisper_model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        word_timestamps=True,
        verbose=False
    )
    return result


print("\n🔵 Transcribing with Constrained Decoding...")
transcript_result = transcribe_with_bias(denoised_path, language="hi")
hook_handle.remove()  # clean up hook

full_transcript = transcript_result["text"]
print(f"\n📝 Transcript (first 500 chars):\n{full_transcript[:500]}")

# Save transcript
with open("transcript.json", "w", encoding="utf-8") as f:
    json.dump(transcript_result, f, ensure_ascii=False, indent=2)
print("\n✅ Transcript saved to transcript.json")


# ─────────────────────────────────────────────
# CELL 7: WER Computation (Task 5 Metric)
# ─────────────────────────────────────────────
from jiwer import wer

def compute_wer(hypothesis: str, reference: str = None) -> float:
    """
    If no reference provided, prints WER = N/A.
    Replace `reference` with your ground-truth transcript.
    """
    if reference is None:
        print("⚠️  No reference transcript provided — WER cannot be computed.")
        print("    Add your ground-truth text as `reference` string.")
        return None
    error = wer(reference, hypothesis)
    print(f"WER: {error*100:.2f}%")
    return error

# Replace with actual ground truth if available
# reference_text = "your ground truth here"
compute_wer(full_transcript)
