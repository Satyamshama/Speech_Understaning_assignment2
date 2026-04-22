# ============================================================
# SPEECH UNDERSTANDING - PA2
# PART III: Zero-Shot Cross-Lingual Voice Cloning (TTS)
# ============================================================

# ─────────────────────────────────────────────
# CELL 10: Install TTS dependencies
# ─────────────────────────────────────────────
"""
!pip install TTS   # Coqui TTS (includes VITS, YourTTS)
!pip install resemblyzer  # for d-vector/x-vector
!pip install fastdtw scipy
"""

import numpy as np
import librosa
import soundfile as sf
import torch
import json
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REF_VOICE   = "student_voice_ref.wav"
AUDIO_PATH  = "denoised_segment.wav"

# ─────────────────────────────────────────────
# CELL 11: Task 3.1 — Speaker Embedding Extraction
# ─────────────────────────────────────────────
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

def extract_speaker_embedding(audio_path: str) -> np.ndarray:
    """
    Extracts a 256-dim d-vector (speaker embedding) using GE2E-trained
    VoiceEncoder from Resemblyzer.
    Your 60-second recording should ideally be clean, mono, 16kHz.
    """
    encoder  = VoiceEncoder(device="cpu")
    wav      = preprocess_wav(Path(audio_path))
    embedding = encoder.embed_utterance(wav)
    print(f"✅ Speaker embedding extracted: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
    return embedding   # [256]


print("🔵 Extracting your speaker embedding from 60s reference...")
speaker_emb = extract_speaker_embedding(REF_VOICE)
np.save("speaker_embedding.npy", speaker_emb)
print("✅ Speaker embedding saved → speaker_embedding.npy")


# ─────────────────────────────────────────────
# CELL 12: Task 3.2 — Prosody Extraction + DTW Warping
# ─────────────────────────────────────────────

def extract_prosody(audio_path: str, sr: int = 16000):
    """
    Extracts F0 (fundamental frequency) and RMS energy contours
    from an audio file.
    Returns:
        f0     : np.array [T] — pitch in Hz (0 = unvoiced)
        energy : np.array [T] — RMS energy
        times  : np.array [T] — timestamps in seconds
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    hop_length = 256

    # F0 via pyin (probabilistic YIN)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        hop_length=hop_length, sr=sr
    )
    f0 = np.nan_to_num(f0, nan=0.0)

    # RMS Energy
    energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    T = min(len(f0), len(energy))
    times = librosa.times_like(f0[:T], sr=sr, hop_length=hop_length)

    return f0[:T], energy[:T], times


print("\n🔵 Extracting prosody from professor's lecture...")
prof_f0, prof_energy, prof_times = extract_prosody(AUDIO_PATH)
print(f"   F0 range: {prof_f0[prof_f0>0].min():.1f} – {prof_f0.max():.1f} Hz")
print(f"   Energy range: {prof_energy.min():.4f} – {prof_energy.max():.4f}")

print("\n🔵 Extracting prosody from your reference voice...")
ref_f0, ref_energy, ref_times = extract_prosody(REF_VOICE)


def dtw_warp_prosody(source_f0, source_energy, target_f0, target_energy):
    """
    Apply Dynamic Time Warping (DTW) to warp the professor's (source) prosody
    onto the synthesized reference (target) sequence.

    DTW finds an optimal alignment path between source and target,
    then resamples the source prosody to match the target length.

    Returns:
        warped_f0     : np.array [len(target_f0)]
        warped_energy : np.array [len(target_energy)]
    """
    # Feature vectors for DTW: [f0_norm, energy_norm]
    def normalize(arr):
        a = arr - arr.mean()
        std = arr.std()
        return a / (std + 1e-8)

    src_feat = np.stack([normalize(source_f0), normalize(source_energy)], axis=1)
    tgt_feat = np.stack([normalize(target_f0), normalize(target_energy)], axis=1)

    # DTW alignment
    distance, path = fastdtw(src_feat, tgt_feat, dist=euclidean)
    path = np.array(path)  # [(src_idx, tgt_idx), ...]

    # Remap source prosody to target indices
    T_tgt = len(target_f0)
    warped_f0     = np.zeros(T_tgt)
    warped_energy = np.zeros(T_tgt)
    count         = np.zeros(T_tgt)

    for src_i, tgt_i in path:
        warped_f0[tgt_i]     += source_f0[src_i]
        warped_energy[tgt_i] += source_energy[src_i]
        count[tgt_i]         += 1

    count = np.maximum(count, 1)
    warped_f0     /= count
    warped_energy /= count

    print(f"✅ DTW warping complete. Path length: {len(path)}")
    print(f"   DTW distance: {distance:.4f}")
    return warped_f0, warped_energy


print("\n🔵 Applying DTW prosody warping...")
warped_f0, warped_energy = dtw_warp_prosody(prof_f0, prof_energy, ref_f0, ref_energy)

np.save("warped_f0.npy", warped_f0)
np.save("warped_energy.npy", warped_energy)
print("✅ Warped prosody saved.")


# ─────────────────────────────────────────────
# CELL 13: Task 3.3 — Zero-Shot TTS Synthesis
# Using YourTTS (supports zero-shot cloning)
# ─────────────────────────────────────────────

def synthesize_lrl_audio(
    lrl_text_path:    str = "lrl_transcript.txt",
    speaker_wav_path: str = "student_voice_ref.wav",
    output_path:      str = "output_LRL_cloned.wav",
    target_sr:        int = 22050
):
    """
    Synthesize LRL speech using YourTTS with zero-shot voice cloning.
    YourTTS conditions on a reference speaker wav (your 60s recording)
    to clone your voice in the target language.
    """
    from TTS.api import TTS as CoquiTTS

    # Load LRL text
    with open(lrl_text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # YourTTS: zero-shot multilingual TTS with speaker embedding
    print("\n🔵 Loading YourTTS model...")
    tts = CoquiTTS(model_name="tts_models/multilingual/multi-dataset/your_tts",
                   progress_bar=False, gpu=torch.cuda.is_available())

    # Synthesize with your voice reference (zero-shot cloning)
    # Split into chunks if text > 500 chars for stability
    MAX_CHUNK = 400
    chunks    = [text[i:i+MAX_CHUNK] for i in range(0, len(text), MAX_CHUNK)]

    all_audio = []
    for idx, chunk in enumerate(chunks):
        tmp_path = f"tmp_chunk_{idx}.wav"
        tts.tts_to_file(
            text=chunk,
            speaker_wav=speaker_wav_path,
            language="hi",          # closest supported; use "en" if needed
            file_path=tmp_path
        )
        audio, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        all_audio.append(audio)
        print(f"   Synthesized chunk {idx+1}/{len(chunks)}")

    # Concatenate all chunks
    final_audio = np.concatenate(all_audio)
    sf.write(output_path, final_audio, target_sr)
    print(f"\n✅ Synthesized audio saved → {output_path}")
    print(f"   Duration: {len(final_audio)/target_sr:.1f}s | Sample rate: {target_sr} Hz")
    return output_path


synth_path = synthesize_lrl_audio()


# ─────────────────────────────────────────────
# CELL 14: MCD Computation (Task 5 Metric)
# ─────────────────────────────────────────────

def compute_mcd(ref_path: str, syn_path: str, sr: int = 22050, n_mfcc: int = 13) -> float:
    """
    Mel-Cepstral Distortion (MCD) between reference and synthesized audio.
    Lower is better. Passing criterion: MCD < 8.0
    MCD = (10/ln10) * sqrt(2) * mean(||mc_ref - mc_syn||_2)
    """
    ref, _ = librosa.load(ref_path, sr=sr, mono=True)
    syn, _ = librosa.load(syn_path, sr=sr, mono=True)

    def get_mcep(y):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc+1)[1:]  # skip C0
        return mfcc.T  # [T, n_mfcc]

    mc_ref = get_mcep(ref)
    mc_syn = get_mcep(syn)

    # Align via DTW for fair comparison
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    _, path = fastdtw(mc_ref, mc_syn, dist=euclidean)
    path    = np.array(path)

    diff = mc_ref[path[:,0]] - mc_syn[path[:,1]]
    mcd  = (10.0 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff**2, axis=1)))
    print(f"📊 MCD = {mcd:.4f} dB  (threshold: < 8.0 dB)")
    return mcd


mcd_score = compute_mcd(REF_VOICE, synth_path)
