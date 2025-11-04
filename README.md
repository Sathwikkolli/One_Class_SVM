# Mono-Speaker Deepfake Guard

A one-class support vector machine (SVM) trained on HuBERT embeddings to detect audio deepfakes using only genuine speech samples from a single speaker.

## Overview

This project implements a deepfake detection system that learns from positive examples only (real audio) using One-Class SVM. It's designed for single-speaker scenarios but can be extended to multiple voices. The system extracts speaker-specific data from Common Voice, applies data augmentation, and trains a robust anomaly detector.

## Features

- **One-Class Learning**: Trains on real audio only—no fake samples required
- **HuBERT Embeddings**: Leverages facebook/hubert-base-ls960 via s3prl for robust feature extraction
- **Data Augmentation**: Speed perturbation (0.9×, 1.1×) and noise injection (25 dB) to expand training data
- **Speaker-Specific Models**: Extract and train on individual Common Voice speakers
- **Validation Pipeline**: Comprehensive evaluation with per-file confidence scores

## Project Structure

```
.
├── audio_svm.py                    # MP3 → WAV conversion (mono, 16 kHz)
├── train_svm.py                    # One-Class SVM training pipeline
├── test_audio.py                   # Single audio file inference
├── validate_model.py               # Batch validation with metrics
├── scripts/
│   ├── extract_speaker.py          # Filter Common Voice by speaker ID
│   └── augment_wavs.py             # Speed + noise augmentation
├── data/
│   ├── real/                       # Raw Common Voice MP3s
│   └── speaker_<id>/
│       ├── raw/                    # Extracted MP3s (train/test)
│       ├── wav/                    # Normalized WAVs
│       └── wav_aug/                # Augmented training data
├── saved_model/                    # Trained SVM checkpoints
└── requirements.txt
```

## Requirements

- Python 3.11+
- PyTorch
- torchaudio
- s3prl
- librosa
- scikit-learn
- soundfile

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: HuBERT feature extraction is CPU-intensive. GPU recommended for faster processing.

## Quick Start

### 1. Extract Speaker Data

Filter Common Voice manifests by `client_id` and create train/test splits:

```bash
python scripts/extract_speaker.py --client-id <speaker_hash>
```

**Output**: 
- `data/speaker_<id>/raw/train/` – Training MP3s
- `data/speaker_<id>/raw/test/` – Test MP3s
- `data/speaker_<id>/missing_files.txt` – Log of unavailable clips

### 2. Convert to WAV

Normalize audio to mono 16 kHz WAV:

```bash
python audio_svm.py --src data/speaker_a/raw/train --dst data/speaker_a/wav/train
python audio_svm.py --src data/speaker_a/raw/test --dst data/speaker_a/wav/test
```

### 3. Augment Training Data

Expand dataset with speed perturbations and noise:

```bash
python scripts/augment_wavs.py \
  --input-dir data/speaker_a/wav/train \
  --output-dir data/speaker_a/wav_aug/train
```

**Result**: 394 clips → 1,576 augmented samples (original + 0.9× + 1.1× + noisy variants)

### 4. Train One-Class SVM

Extract HuBERT embeddings and fit the model:

```bash
python train_svm.py \
  --real-dir data/speaker_a/wav_aug/train \
  --output-dir saved_model/speaker_a_aug \
  --nu 0.01 \
  --gamma 0.001
```

**Hyperparameters**:
- `--nu`: Upper bound on fraction of outliers (default: 0.01)
- `--gamma`: RBF kernel coefficient (default: 0.001)

### 5. Validate Model

Evaluate on test set:

```bash
python validate_model.py \
  --real-dir data/speaker_a/wav/test \
  --fake-dir data/fake_placeholder \
  --model-dir saved_model/speaker_a_aug \
  --results-path validation_results.csv
```

Test a single audio file:

```bash
python test_audio.py --audio-path path/to/clip.wav --model-dir saved_model/speaker_a_aug
```

## Dataset Details

### Common Voice Corpus
- **Total Pool**: 2,892 MP3s across multiple speakers
- **Speaker A Subset**: 394 train / 42 test clips
- **Missing Data**: 1,299 referenced files not found locally (logged in `missing_files.txt`)

### Augmentation Strategy
Each training WAV generates:
- Original (1×)
- Slow speed (0.9×)
- Fast speed (1.1×)
- Noise-injected variants (25 dB SNR)

## Performance

### Speaker A Results (Augmented Model)
- **Accuracy**: 85.71% on 42-clip test set
- **Confusion**: 36 correctly classified as REAL, 6 false positives (flagged as FAKE)
- **Decision Scores**: False negatives cluster near –0.12

### Threshold Tuning
Adjust classification threshold in `validate_model.py`:
```python
# Default: score > 0 → REAL, else FAKE
# Relaxed: score > -0.15 → REAL
```

## Pipeline Walkthrough

1. **Speaker Extraction** → `extract_speaker.py` filters manifests and copies MP3s
2. **Audio Normalization** → `audio_svm.py` standardizes format
3. **Data Augmentation** → `augment_wavs.py` creates variants
4. **Feature Extraction** → `train_svm.py` loads HuBERT, mean-pools 768-dim embeddings
5. **Model Training** → Fits RBF One-Class SVM with StandardScaler
6. **Inference** → `test_audio.py` / `validate_model.py` score new audio

## Extending to New Speakers

Repeat the pipeline with a different `--client-id`:

```bash
python scripts/extract_speaker.py --client-id <new_speaker_hash>
# ... follow steps 2-5 with new paths
```

## Future Work

- [ ] Recover missing Common Voice clips
- [ ] Hyperparameter grid search (nu, gamma)
- [ ] Calibrated probability scores (Platt scaling)
- [ ] Multi-speaker ensemble model
- [ ] Real-time streaming inference

## Dependencies

Key packages (see `requirements.txt` for full list):
```
torch>=2.0.0
torchaudio>=2.0.0
s3prl>=0.4.0
librosa>=0.10.0
scikit-learn>=1.3.0
soundfile>=0.12.0
```

## Notes

- **Manifests Required**: Keep `validated.tsv` / `other.tsv` alongside audio files
- **GPU Recommended**: CPU feature extraction takes ~2-3 seconds per clip
- **Reproducibility**: Speaker extraction uses deterministic train/test splits (80/20)

## License

MIT License - See LICENSE file for details

## Citation

If you use this code, please cite:
```bibtex
@software{mono_speaker_deepfake_guard,
  title={Mono-Speaker Deepfake Guard: One-Class SVM Audio Verification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/mono-speaker-deepfake-guard}
}
```

## Contact

For questions or contributions, open an issue or submit a pull request.

---

**Built with HuBERT embeddings and One-Class SVM for robust single-speaker deepfake detection.**
