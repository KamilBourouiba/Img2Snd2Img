import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file and extract features
y, sr = librosa.load('output.mp3', sr=22050, res_type='kaiser_fast')

spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

# Convert to image and save
spectrogram_image = np.uint8((log_spectrogram - log_spectrogram.min()) / (log_spectrogram.max() - log_spectrogram.min()) * 255)
spectrogram_image = 255 - spectrogram_image  
# Invert colors to make the image more visually appealing

# Display image using Matplotlib
plt.imshow(spectrogram_image, cmap='gray', interpolation='nearest')
plt.show()

# Save image
plt.imsave('output.png', spectrogram_image, cmap='gray')