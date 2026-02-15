import streamlit as st
import torch
from model import LSTMModel
from generate import generate_text
from utils import build_vocab
from audio import abc_to_wav
import time
import base64
import streamlit.components.v1 as components

# ---------- Animated Visualizers ----------

def animated_visualizer(audio_bytes, mode="bars"):
    b64 = base64.b64encode(audio_bytes).decode()

    script_mode = {
        "bars": "drawBars()",
        "wave": "drawWave()",
        "circle": "drawCircle()"
    }[mode]

    html = f"""
    <audio id="audio" controls src="data:audio/wav;base64,{b64}"></audio>
    <canvas id="canvas" width="800" height="350"></canvas>

    <script>
    const audio = document.getElementById("audio");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;

    const source = audioCtx.createMediaElementSource(audio);
    source.connect(analyser);
    analyser.connect(audioCtx.destination);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function drawBars() {{
        requestAnimationFrame(drawBars);
        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {{
            const h = dataArray[i];
            ctx.fillStyle = `hsl(${{h*2}}, 100%, 50%)`;
            ctx.fillRect(x, canvas.height - h, barWidth, h);
            x += barWidth + 1;
        }}
    }}

    function drawWave() {{
        requestAnimationFrame(drawWave);
        analyser.getByteTimeDomainData(dataArray);

        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2;
        ctx.strokeStyle = "cyan";
        ctx.beginPath();

        let slice = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {{
            let v = dataArray[i] / 128.0;
            let y = v * canvas.height/2;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);

            x += slice;
        }}

        ctx.stroke();
    }}

    function drawCircle() {{
        requestAnimationFrame(drawCircle);
        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        let cx = canvas.width/2;
        let cy = canvas.height/2;

        for (let i = 0; i < bufferLength; i++) {{
            let r = dataArray[i] + 50;
            let angle = (i / bufferLength) * Math.PI * 2;

            let x = cx + r * Math.cos(angle);
            let y = cy + r * Math.sin(angle);

            ctx.fillStyle = `hsl(${{i*3}}, 100%, 50%)`;
            ctx.fillRect(x, y, 4, 4);
        }}
    }}

    audio.onplay = () => audioCtx.resume();
    {script_mode};
    </script>
    """

    components.html(html, height=380)

# ---------- Page Config ----------

st.set_page_config(page_title="AI Music Composer", page_icon="🎵", layout="wide")

# ---------- Load Dataset ----------

with open("songs.txt", "r") as f:
    songs_joined = f.read()

vocab, char2idx, idx2char = build_vocab(songs_joined)
device = torch.device("cpu")

# ---------- Load Model ----------

@st.cache_resource
def load_model():
    checkpoint = torch.load("music_model.pt", map_location=device)
    model = LSTMModel(
        checkpoint["vocab_size"],
        checkpoint["embedding_dim"],
        checkpoint["hidden_size"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

model = load_model()

# ---------- Sidebar ----------

st.sidebar.title("🎛 Control Panel")

seed = st.sidebar.text_input("Seed text", "X")
temperature = st.sidebar.slider("Creativity 🎨", 0.2, 1.5, 0.8)
length = st.sidebar.slider("Song length 🎼", 100, 2000, 500)

viz_mode = st.sidebar.selectbox(
    "Visualizer Style",
    ["🪩 Spectrum Bars", "🌊 Wave Neon", "🔥 Radial Pulse"]
)

mode_map = {
    "🪩 Spectrum Bars": "bars",
    "🌊 Wave Neon": "wave",
    "🔥 Radial Pulse": "circle"
}

compose = st.sidebar.button("🎶 Compose Music")

# ---------- Main UI ----------

st.title("🎵 AI Music Composer")
st.subheader("Neural network generating music in real-time")

col1, col2 = st.columns([1,1])

if compose:

    progress = st.progress(0)
    eta = st.empty()

    start = time.time()

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i+1)

        elapsed = time.time() - start
        remaining = elapsed * (100/(i+1) - 1)
        eta.text(f"⏳ ETA: {remaining:.1f}s")

    with st.spinner("Finalizing composition..."):
        text = generate_text(
            model,
            seed,
            char2idx,
            idx2char,
            device,
            generation_length=length,
            temperature=temperature
        )

    st.balloons()

    with col1:
        st.subheader("Generated ABC Score")
        st.code(text)

    audio_bytes = abc_to_wav(text)

    with col2:
        st.subheader("Audio + Live Visualizer")
        st.caption("🎧 Tip: press play to activate the live visualizer")

        animated_visualizer(audio_bytes, mode_map[viz_mode])

        st.download_button(
            label="⬇ Download Audio",
            data=audio_bytes,
            file_name="ai_composition.wav",
            mime="audio/wav"
        )

st.markdown("---")
st.caption("Built with PyTorch + Streamlit • Neural Music Generation Demo")
