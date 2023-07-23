import librosa
import numpy as np
from PIL import Image
from pydub import AudioSegment

# Load image and convert to grayscale
img = Image.open('image.jpg').convert('L')

# Resize image and convert to spectrogram
img = img.resize((256, 256))
data = np.array(img)
spec = librosa.feature.melspectrogram(S=data)

# Synthesize audio from spectrogram
audio = librosa.feature.inverse.mel_to_audio(spec)
audio = (audio * 32767).astype(np.int16)
sound = AudioSegment(audio.tobytes(), frame_rate=22050, sample_width=2, channels=1)

# Export as MP3 file
sound.export('output.mp3', format='mp3')