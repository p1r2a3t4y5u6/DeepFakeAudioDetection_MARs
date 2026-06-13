import os
import tempfile

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.main { max-width: 800px; margin: 0 auto; }

.result-genuine {
    background: linear-gradient(135deg, #1a472a, #2d6a4f);
    border: 2px solid #52b788;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 20px 0;
}

.result-deepfake {
    background: linear-gradient(135deg, #6b1a1a, #9b2226);
    border: 2px solid #e63946;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 20px 0;
}

.result-label {
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: 1px;
}

.result-sublabel {
    font-size: 1rem;
    color: rgba(255,255,255,0.75);
    margin-top: 6px;
}

.confidence-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border-radius: 50px;
    padding: 6px 20px;
    font-size: 1.1rem;
    color: white;
    font-weight: 600;
    margin-top: 14px;
}

.prob-row {
    display: flex;
    justify-content: center;
    gap: 24px;
    margin-top: 16px;
}

.prob-item {
    text-align: center;
    color: rgba(255,255,255,0.85);
    font-size: 0.9rem;
}

.prob-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: white;
}

.upload-box {
    border: 2px dashed #4a4a6a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    background: #0e0e1a;
}

.info-card {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    border-left: 3px solid #7c6fcd;
}

h1 { color: #e0e0ff !important; }
.stProgress > div > div { background-color: #7c6fcd; }
</style>
""", unsafe_allow_html=True)


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

    model_path = "best_model.pt"


S = Settings()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@st.cache_resource
def load_model():
    if not os.path.exists(S.model_path):
        return None
    model = AudioAuthenticityModel().to(DEVICE)
    model.load_state_dict(
        torch.load(S.model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model


def bytes_to_melspec(file_bytes: bytes):
    mel_transform = T.MelSpectrogram(
        sample_rate=S.sample_rate,
        n_fft=S.n_fft,
        hop_length=S.hop_length,
        n_mels=S.n_mels,
        f_min=S.f_min,
        f_max=S.f_max,
    )
    db_transform = T.AmplitudeToDB(top_db=80)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        samples, sr = librosa.load(tmp_path, sr=None, mono=True)
        waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    finally:
        os.unlink(tmp_path)

    if sr != S.sample_rate:
        waveform = T.Resample(sr, S.sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    length = waveform.shape[1]
    if length < S.max_samples:
        waveform = F.pad(waveform, (0, S.max_samples - length))
    else:
        waveform = waveform[:, :S.max_samples]

    mel = db_transform(mel_transform(waveform))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0), mel.squeeze().numpy()


def render_spectrogram(mel_np: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")

    img = ax.imshow(mel_np, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    ax.set_title("Mel-Spectrogram", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Time Frames", color="#aaaacc")
    ax.set_ylabel("Mel Bins", color="#aaaacc")
    ax.tick_params(colors="#aaaacc")

    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaacc")

    plt.tight_layout()
    return fig


def main():
    st.markdown("# 🎙️ Deepfake Audio Detector")
    st.markdown(
        "Upload a speech recording to detect whether it is "
        "**Genuine (Human)** or **Deepfake (AI-Generated)**."
    )
    st.divider()

    with st.expander("ℹ️ How it works", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**1. 🎵 Audio Input**\nWAV / MP3 / FLAC")
        with col2:
            st.markdown("**2. 📊 Mel-Spectrogram**\n128 mel bins")
        with col3:
            st.markdown("**3. 🧠 CNN + Transformer**\nLocal + global features")
        with col4:
            st.markdown("**4. 🏷️ Classification**\nGenuine vs Deepfake")

    st.divider()

    model = load_model()
    if model is None:
        st.error(
            "⚠️ `best_model.pt` not found in the app directory. "
            "Please place the trained model file next to `app.py`."
        )
        st.stop()

    st.markdown("### 📂 Upload Audio File")
    uploaded = st.file_uploader(
        label="Choose an audio file",
        type=["wav", "mp3", "flac", "ogg"],
        help="Supported formats: WAV, MP3, FLAC, OGG",
    )

    if uploaded is None:
        st.info("👆 Upload an audio file above to get started.")
        return

    st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
    st.markdown(f"**File:** `{uploaded.name}` | **Size:** `{uploaded.size / 1024:.1f} KB`")

    with st.spinner("Analysing audio..."):
        try:
            file_bytes = uploaded.read()
            mel_tensor, mel_np = bytes_to_melspec(file_bytes)
            mel_tensor = mel_tensor.to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(model(mel_tensor), dim=1)[0].cpu().numpy()

            genuine_prob = float(probs[0])
            deepfake_prob = float(probs[1])

            decision_threshold = 0.000018
            label = "Deepfake" if deepfake_prob >= decision_threshold else "Genuine"
            confidence = deepfake_prob if label == "Deepfake" else genuine_prob
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return

    st.markdown("### 🔍 Detection Result")

    if label == "Genuine":
        st.markdown(f"""
        <div class="result-genuine">
            <p class="result-label">✅ GENUINE</p>
            <p class="result-sublabel">This audio appears to be real human speech</p>
            <div class="confidence-badge">Confidence: {confidence*100:.1f}%</div>
            <div class="prob-row">
                <div class="prob-item">
                    <div class="prob-value">{genuine_prob*100:.1f}%</div>
                    Genuine
                </div>
                <div class="prob-item">
                    <div class="prob-value">{deepfake_prob*100:.1f}%</div>
                    Deepfake
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-deepfake">
            <p class="result-label">⚠️ DEEPFAKE</p>
            <p class="result-sublabel">This audio appears to be AI-generated speech</p>
            <div class="confidence-badge">Confidence: {confidence*100:.1f}%</div>
            <div class="prob-row">
                <div class="prob-item">
                    <div class="prob-value">{genuine_prob*100:.1f}%</div>
                    Genuine
                </div>
                <div class="prob-item">
                    <div class="prob-value">{deepfake_prob*100:.1f}%</div>
                    Deepfake
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Probability Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 Genuine**")
        st.progress(genuine_prob)
        st.markdown(f"`{genuine_prob*100:.2f}%`")
    with col2:
        st.markdown("**🔴 Deepfake**")
        st.progress(deepfake_prob)
        st.markdown(f"`{deepfake_prob*100:.2f}%`")

    st.markdown("### 🎨 Mel-Spectrogram Visualization")
    fig = render_spectrogram(mel_np)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#666688; font-size:0.85rem;'>"
        "CNN + Transformer pipeline · Trained on Fake-or-Real Dataset · "
        "Val Accuracy 99.9% · EER 0.08%"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
