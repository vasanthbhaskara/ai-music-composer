import subprocess
import tempfile
import os

def abc_to_wav(abc_text):
    with tempfile.TemporaryDirectory() as tmpdir:
        abc_file = os.path.join(tmpdir, "song.abc")
        midi_file = os.path.join(tmpdir, "song.mid")
        wav_file = os.path.join(tmpdir, "song.wav")

        with open(abc_file, "w") as f:
            f.write(abc_text)

        subprocess.run(["abc2midi", abc_file, "-o", midi_file])
        subprocess.run(["timidity", midi_file, "-Ow", "-o", wav_file])

        with open(wav_file, "rb") as f:
            audio_bytes = f.read()

    return audio_bytes