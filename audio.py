import numpy as np
import io
from scipy.io.wavfile import write

# simple note frequency mapping (subset)
NOTE_FREQ = {
    "C": 261.63,
    "D": 293.66,
    "E": 329.63,
    "F": 349.23,
    "G": 392.00,
    "A": 440.00,
    "B": 493.88,
}

def synth_note(freq, duration=0.25, rate=44100):
    t = np.linspace(0, duration, int(rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return wave

def abc_to_wav(abc_text):
    notes = []

    for char in abc_text:
        if char.upper() in NOTE_FREQ:
            freq = NOTE_FREQ[char.upper()]
            notes.append(synth_note(freq))

    if not notes:
        notes.append(synth_note(440))

    audio = np.concatenate(notes)
    audio = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    write(buffer, 44100, audio)
    buffer.seek(0)

    return buffer.read()
