# Speech Assignment 2: Code-Switched Transcription & Zero-Shot Voice Cloning

## Overview
This project implements a complete pipeline for transcribing code-switched (English-Hindi) lectures and re-synthesizing them in a Low-Resource Language (LRL) using zero-shot voice cloning.

## Project Structure
```
.
├── pipeline.py                    # Main integrated pipeline
├── pipeline.ipynb                 # Jupyter notebook version
├── Scripts/
│   ├── PA2_Part1_STT.py          # Multi-Head LID + Constrained Decoding
│   ├── PA2_Part2_Phonetic.py     # IPA Conversion & Translation
│   ├── PA2_Part3_TTS.py          # Zero-Shot Voice Cloning with Prosody Warping
│   └── PA2_Part4_Adversarial.py  # Anti-Spoofing & Adversarial Robustness
├── Data/
│   ├── original_segment.wav      # Input lecture snippet (10 min)
│   └── student_voice_ref.wav     # Reference voice for cloning (60 sec)
├── Results/
│   ├── output_LRL_cloned.wav     # Final synthesized lecture
│   ├── denoised_segment.wav      # Preprocessed audio
│   ├── lid_segments.json         # Language ID predictions with timestamps
│   ├── ipa_transcript.txt        # IPA representation
│   ├── lrl_transcript.txt        # Low-Resource Language transcript
│   ├── parallel_corpus.json      # Technical term translation dictionary
│   ├── adversarial_results.json  # Robustness evaluation metrics
│   ├── speaker_embedding.npy     # Voice embeddings
│   ├── warped_f0.npy             # Prosodic F0 contours
│   └── warped_energy.npy         # Energy contours
├── Models/
│   ├── lid_model.pth             # Trained Language ID classifier
│   └── cm_model.pth              # Anti-Spoofing countermeasure model
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment definition
├── setup.bat / setup.sh          # Setup scripts
└── README.md                      # This file
```

## Installation

### Option 1: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate speech-assignment2
```

### Option 2: Using pip
```bash
pip install -r requirements.txt
```

### Option 3: Using Setup Script
**Windows:**
```cmd
setup.bat
```

**Linux/macOS:**
```bash
bash setup.sh
chmod +x setup.sh
```

## Pipeline Components

### Part 1: Robust Code-Switched Transcription (STT)
**Task 1.1: Multi-Head Language Identification (LID)**
- Frame-level language detection between English and Hindi
- Custom CNN architecture for binary classification
- Target F1-score: ≥0.85

**Task 1.2: Constrained Decoding**
- Uses OpenAI Whisper v3 as base model
- Implements Logit Bias for N-gram language model
- Prioritizes technical terms from speech course syllabus
- Custom beam search decoding

**Task 1.3: Denoising & Normalization**
- Spectral subtraction for noise removal
- Classroom reverb reduction
- Multi-band energy normalization

### Part 2: Phonetic Mapping & Translation
**Task 2.1: IPA Unified Representation**
- Grapheme-to-Phoneme conversion for Hinglish
- Custom phonological mapping layer
- Handles code-switching phonology

**Task 2.2: Semantic Translation**
- Code-switched text to target LRL translation
- 500+ technical term parallel corpus
- Domain-specific translation for speech concepts

### Part 3: Zero-Shot Cross-Lingual Voice Cloning (TTS)
**Task 3.1: Voice Embedding Extraction**
- D-vector/X-vector speaker embeddings
- 60-second reference voice processing
- Normalization and quality checks

**Task 3.2: Prosody Warping**
- F0 (Fundamental Frequency) extraction via YIN algorithm
- Energy contour extraction from log-magnitude spectrogram
- Dynamic Time Warping (DTW) for prosodic alignment
- Preserves teaching style and intonation patterns

**Task 3.3: Synthesis**
- VITS-based generative model
- Output: 22.05kHz+ WAV format
- 10-minute synthesized lecture

### Part 4: Adversarial Robustness & Spoofing Detection
**Task 4.1: Anti-Spoofing Classifier**
- LFCC/CQCC-based countermeasure system
- Binary classification: Bona Fide vs. Spoof
- Equal Error Rate (EER) metric tracking

**Task 4.2: Adversarial Noise Injection**
- FGSM-based adversarial perturbations
- Language ID robustness testing
- SNR > 40dB for imperceptibility

## Usage

### Run Full Pipeline
```python
python pipeline.py
```

### Run Individual Components
```python
# Part 1: Speech-to-Text with Language ID
python Scripts/PA2_Part1_STT.py

# Part 2: Phonetic Mapping
python Scripts/PA2_Part2_Phonetic.py

# Part 3: Text-to-Speech with Voice Cloning
python Scripts/PA2_Part3_TTS.py

# Part 4: Adversarial Robustness & Spoofing
python Scripts/PA2_Part4_Adversarial.py
```

### Using Jupyter Notebook
```bash
jupyter notebook pipeline.ipynb
```

## Evaluation Metrics

| Metric | Target | Status |
|--------|--------|--------|
| WER (English) | <15% | ✓ Achieved |
| WER (Hindi) | <25% | ✓ Achieved |
| MCD (Mel-Cepstral Distortion) | <8.0 | ✓ Achieved |
| LID Switching Accuracy | ±200ms | ✓ Achieved |
| Anti-Spoofing EER | <10% | ✓ Achieved |
| Adversarial Robustness (ε) | Reported | ✓ Reported |

## Key Features

✅ **Frame-level Language Identification** - Precise language switching detection  
✅ **Custom N-gram Language Model** - Technical term prioritization  
✅ **Hinglish Phonology Support** - Custom G2P mapping for code-switching  
✅ **Prosody Warping with DTW** - Preserves speaker characteristics  
✅ **Zero-Shot Voice Cloning** - Speaker adaptation without fine-tuning  
✅ **Anti-Spoofing Classifier** - Synthetic speech detection  
✅ **Adversarial Robustness Analysis** - Perturbation sensitivity testing  

## System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.10+
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional but recommended)
- **Storage**: 20GB free space (for models and audio files)

## Dependencies

See `requirements.txt` for Python package dependencies.

**Key Libraries:**
- PyTorch 2.0.1 + TorchAudio
- OpenAI Whisper (Speech-to-Text)
- Transformers (VITS model)
- Librosa (Audio processing)
- Scikit-Learn (ML utilities)
- FastDTW (Dynamic Time Warping)
- Resemblyzer (Voice embeddings)

## Configuration

Edit `pipeline.py` to customize:
- Audio paths and sample rates
- Model hyperparameters
- LID threshold values
- Beam search parameters
- Voice cloning settings
- Adversarial perturbation epsilon

## Output Files

- **output_LRL_cloned.wav** - Synthesized lecture (22.05kHz, 10 minutes)
- **lid_segments.json** - Language predictions with timestamps (±200ms precision)
- **ipa_transcript.txt** - International Phonetic Alphabet representation
- **lrl_transcript.txt** - Target language transcript
- **adversarial_results.json** - Robustness metrics (ε, perturbation analysis)
- **Models/** - Trained LID and CM (countermeasure) models

## Troubleshooting

**CUDA/GPU Issues:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Memory Issues:**
- Reduce batch size in configuration
- Enable audio chunking (automatic for >10min)
- Use CPU inference if GPU memory insufficient

**Missing Audio Files:**
- Ensure `Data/original_segment.wav` and `student_voice_ref.wav` are present
- Check audio formats (WAV recommended)

## Performance Benchmarks

Tested on:
- NVIDIA RTX 3090 GPU
- 32GB RAM
- Python 3.10

Processing times:
- LID + Transcription: ~8min for 10min audio
- Phonetic mapping: ~2min
- Voice cloning & synthesis: ~12min
- Total pipeline: ~25min

## References & Citations

1. **Whisper**: Radford et al., 2022 - "Robust Speech Recognition via Large-Scale Weak Supervision"
2. **VITS**: Kim et al., 2021 - "Conditional Variational Autoencoder with Adversarial Learning"
3. **FastDTW**: Salvador & Chan, 2007 - "FastDTW: Toward Accurate Dynamic Time Warping"
4. **Resemblyzer**: 2023 - Voice Encoder for speaker embeddings
5. **Hinglish Phonology**: Sharma et al., 2020 - Linguistic analysis of code-switching



## Author Notes

This implementation prioritizes architectural correctness over performance optimization. Custom components include:
- Frame-level bidirectional LID model
- Hinglish-specific G2P mapping layer
- Prosody warping module with DTW
- Adversarial robustness evaluation framework

**Design Decisions:**
See the supplementary 1-page implementation notes document for non-obvious architectural choices in each component.

---

**Last Updated:** 2026
**Status:** Complete & Tested
