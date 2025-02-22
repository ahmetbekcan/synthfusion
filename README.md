# Synthfusion
(Model repository : https://huggingface.co/ahmetbekcan/synthfusion-256)

(This repository is based on the https://github.com/teticio/audio-diffusion)

Synthesizer sound generation using latent diffusion conditioned on audio encodings. Output quality is low since it was trained on a relatively small dataset. However, it would be a nice idea to reproduce some synthesizer sounds given example audio samples. Some output examples can be seen in the "outputs" folder.

## How to run

### Prepare running environment
  conda create -n synthfusion-env python=3.10.15
  
  conda activate synthfusion
  
  pip install -r requirements.txt

### Prepare inputs
  In addition to given example audio inputs, .wav files can be added into "audio-condition-input" folder.

### Run the code
  python conditional-inference.py
  
  This will save the outputs into "outputs" folder.
