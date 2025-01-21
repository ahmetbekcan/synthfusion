import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath("")))
import torch
import urllib
import requests
from IPython.display import Audio
from audiodiffusion import AudioDiffusion
from audiodiffusion.audio_encoder import AudioEncoder
import matplotlib.pyplot as plt
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device)

audio_diffusion = AudioDiffusion(model_id="ahmetbekcan/synthfusion-256")
audio_encoder = AudioEncoder.from_pretrained("teticio/audio-encoder")

file_path = os.path.abspath("audio-condition-input/dx7-synth.mp3")
encoding = torch.unsqueeze(audio_encoder.encode([file_path]),
                           axis=1).to(device)
output_dir = "outputs"

for _ in range(3):
    seed = generator.seed()
    print(f'Seed = {seed}')
    generator.manual_seed(seed)
    image, (sample_rate,
            audio) = audio_diffusion.generate_spectrogram_and_audio(
                generator=generator, encoding=encoding)
    spectrogram_file = os.path.join(output_dir, f"output_spectrogram_seed_{seed}.png")
    plt.imsave(spectrogram_file, image)
    print(f"Saved spectrogram: {spectrogram_file}")
    
    # Save audio as a .wav file
    audio_file = os.path.join(output_dir, f"output_audio_seed_{seed}.wav")
    sf.write(audio_file, audio, sample_rate)
    print(f"Saved audio: {audio_file}")