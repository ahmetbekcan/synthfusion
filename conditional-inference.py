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
condition_path = "audio-condition-input/moog-synth.wav"

output_dir = "outputs"
number_of_outputs = 1

condition_path = "audio-condition-input/"
for file_name in os.listdir(condition_path):
    if not file_name.endswith('.wav'):
        continue
    
    audio_file = os.path.join(condition_path, file_name)
    encoding = torch.unsqueeze(audio_encoder.encode([audio_file]),
                           axis=1).to(device)
    for _ in range(number_of_outputs):
        os.path.splitext(os.path.basename(audio_file))[0]
        seed = generator.seed()
        print(f'Seed = {seed}')
        generator.manual_seed(seed)
        image, (sample_rate,
                audio) = audio_diffusion.generate_spectrogram_and_audio(
                    generator=generator, encoding=encoding)
        audio_file_name = os.path.splitext(os.path.basename(audio_file))[0]
        spectrogram_file = os.path.join(output_dir, f"{audio_file_name}_output_spectrogram_seed_{seed}.png")
        plt.imsave(spectrogram_file, image)
        print(f"Saved spectrogram: {spectrogram_file}")
        
        # Save audio as a .wav file
        audio_file = os.path.join(output_dir, f"{audio_file_name}_output_audio_seed_{seed}.wav")
        sf.write(audio_file, audio, sample_rate)
        print(f"Saved audio: {audio_file}")