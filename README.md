# AI Music Composer

An interactive neural network that composes original music in real time using a character-level LSTM trained on symbolic music notation.

Live demo: https://ai-music-lab.streamlit.app

This project explores sequence modeling and generative AI by treating music as a language modeling problem. The system learns the grammar of ABC music notation and generates new compositions token-by-token.

---

## Features

- Real-time AI music composition
- Creativity control via temperature scaling
- Adjustable composition length
- Built-in audio playback
- Live animated audio visualizer
- Download generated music
- Interactive Streamlit web interface

---

## Model Overview

The model is a character-level LSTM trained on ~800 folk songs in ABC notation.

Architecture:

```
Input tokens
→ Embedding (256)
→ LSTM (hidden size 1024)
→ Linear projection
→ Softmax sampling
```

Key ideas:

- Music treated as next-token prediction
- Sliding window batching
- Cross-entropy loss
- Multinomial sampling for generation
- Temperature scaling to control creativity

Conceptually similar to language modeling, but applied to musical structure.

---

## Tech Stack

- PyTorch — neural network training
- NumPy — data processing
- Streamlit — interactive UI
- Web Audio API — live visualizers
- Comet ML — experiment tracking

---

## Run Locally

Clone the repository:

```
git clone https://github.com/yourusername/ai-music-composer
cd ai-music-composer
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
python -m streamlit run app.py
```

Open your browser:

```
http://localhost:8501
```

---

## Run in Browser

No installation required:

https://ai-music-lab.streamlit.app

---

## Usage

1. Enter a seed string to influence style  
2. Adjust creativity (temperature)  
3. Set song length  
4. Press Compose Music  
5. Listen, visualize, and download

Each composition is unique.

---

## Project Structure

```
app.py          → Streamlit interface
model.py        → LSTM architecture
generate.py     → music generation logic
audio.py        → audio synthesis
utils.py        → preprocessing utilities
songs.txt       → training dataset
```

---

## What This Project Demonstrates

- Sequence modeling with LSTMs
- Character-level generative modeling
- Training + inference pipeline design
- Interactive ML deployment
- Audio synthesis + visualization
- ML to product integration

Full lifecycle:

data → modeling → training → inference → UI → deployment

---

## Author

Built by Vasanth Bhaskara

Exploring generative AI, ML systems, and interactive neural applications.
