import os
import argparse

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


class Settings:
    sample_rate = 16000
    duration = 4
    n_mels = 128
    n_fft = 1024
    hop_length = 256
    f_min = 20
    f_max = 8000

    max_samples = sample_rate * duration
    time_frames = max_samples // hop_length + 1

    cnn_channels = [1, 32, 64, 128]
    cnn_dropout = 0.2

    d_model = 128
    n_heads = 8
    n_layers = 4
    ff_dim = 512
    tf_dropout = 0.1


S = Settings()


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(pool),
            nn.Dropout2d(S.cnn_dropout),
        )

    def forward(self, x):
        return self.layers(x)


class SpectrogramEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = S.cnn_channels
        self.block1 = ConvBlock(ch[0], ch[1], pool=(2, 2))
        self.block2 = ConvBlock(ch[1], ch[2], pool=(2, 2))
        self.block3 = ConvBlock(ch[2], ch[3], pool=(2, 1))
        self.proj = nn.Linear(ch[3] * (S.n_mels // 8), S.d_model)
        self.norm = nn.LayerNorm(S.d_model)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        batch, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, width, channels * height)
        return self.norm(self.proj(x))


class TemporalEncoder(nn.Module):
    def __init__(self, max_len=500):
        super().__init__()
        self.position_embed = nn.Embedding(max_len, S.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=S.d_model,
            nhead=S.n_heads,
            dim_feedforward=S.ff_dim,
            dropout=S.tf_dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=S.n_layers, norm=nn.LayerNorm(S.d_model)
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.encoder(x + self.position_embed(positions))


class AttentionClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.attn_score = nn.Linear(S.d_model, 1)
        self.classifier = nn.Sequential(
            nn.Linear(S.d_model, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        weights = torch.softmax(self.attn_score(x), dim=1)
        pooled = (weights * x).sum(dim=1)
        return self.classifier(pooled)


class AudioAuthenticityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpectrogramEncoder()
        self.temporal = TemporalEncoder()
        self.head = AttentionClassifier()

    def forward(self, x):
        return self.head(self.temporal(self.encoder(x)))


def audio_to_melspec(path: str) -> torch.Tensor:
    mel_transform = T.MelSpectrogram(
        sample_rate=S.sample_rate,
        n_fft=S.n_fft,
        hop_length=S.hop_length,
        n_mels=S.n_mels,
        f_min=S.f_min,
        f_max=S.f_max,
    )
    db_transform = T.AmplitudeToDB(top_db=80)

    samples, sr = librosa.load(path, sr=S.sample_rate, mono=True)
    waveform = torch.FloatTensor(samples).unsqueeze(0)

    length = waveform.shape[1]
    if length < S.max_samples:
        waveform = F.pad(waveform, (0, S.max_samples - length))
    else:
        waveform = waveform[:, :S.max_samples]

    mel = db_transform(mel_transform(waveform))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0)


def run_inference(audio_path: str, model_path: str, device: torch.device) -> dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = AudioAuthenticityModel().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    mel = audio_to_melspec(audio_path).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(mel), dim=1)[0].cpu().numpy()

    decision_threshold = 0.000018
    label = "Deepfake" if probs[1] >= decision_threshold else "Genuine"

    return {
        "label": label,
        "confidence": float(probs.max()),
        "genuine_prob": float(probs[0]),
        "deepfake_prob": float(probs[1]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify a speech recording as Genuine (Human) or Deepfake (AI-Generated)."
    )
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav / .flac / .mp3 / .ogg)")
    parser.add_argument("--model", default="best_model.pt", help="Path to trained model weights")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nDevice : {device}")
    print(f"Model  : {args.model}")
    print(f"Audio  : {args.audio}")

    result = run_inference(args.audio, args.model, device)

    divider = "=" * 50
    label_text = (
        "GENUINE (Human Speech)" if result["label"] == "Genuine" else "DEEPFAKE (AI-Generated)"
    )

    print(f"\n{divider}")
    print(f"  Result     : {label_text}")
    print(f"  Confidence : {result['confidence'] * 100:.1f}%")
    print(f"  Genuine    : {result['genuine_prob'] * 100:.1f}%")
    print(f"  Deepfake   : {result['deepfake_prob'] * 100:.1f}%")
    print(f"{divider}\n")


if __name__ == "__main__":
    main()
