# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from wavegrad.params import params


Fs = params.sample_rate

def load_wav(filename):
  audio, sr = T.load_wav(filename)

  if Fs != sr:
    audio = T.transforms.Resample(orig_freq=sr, new_freq=Fs)(audio)

  audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)
  return audio, Fs

def extract_melspectrogram(audio):
  """ 
  :param audio: [T] with value in [-1, 1]
  """
  
  hop = params.hop_samples
  win = hop * 4
  n_fft = 2**((win-1).bit_length())
  f_max = Fs / 2.0
  mel_spec_transform = TT.MelSpectrogram(
    sample_rate=Fs, 
    n_fft=n_fft, 
    win_length=win, 
    hop_length=hop, 
    f_min=20.0, 
    f_max=f_max, 
    # power=1.0,    # FIXME disabled for experimental purpose
    # normalized=True,  # FIXME disabled for experimental purpose
  )

  # with torch.no_grad():
  spectrogram = mel_spec_transform(audio)
  spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
  spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    # if save:
    #   np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())
  return spectrogram

def transform(filename, save=False):
  # audio, sr = T.load_wav(filename)

  # if Fs != sr:
  #   audio = T.transforms.Resample(orig_freq=sr, new_freq=Fs)(audio)
  #   # raise ValueError(f'Invalid sample rate {sr}.')
  # audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)
  audio, Fs = load_wav(filename)

  # hop = params.hop_samples
  # win = hop * 4
  # n_fft = 2**((win-1).bit_length())
  # f_max = Fs / 2.0
  # mel_spec_transform = TT.MelSpectrogram(sample_rate=Fs, n_fft=n_fft, win_length=win, hop_length=hop, f_min=20.0, f_max=f_max, power=1.0, normalized=True)

  with torch.no_grad():
  #   spectrogram = mel_spec_transform(audio)
  #   spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
  #   spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    spectrogram = extract_melspectrogram(audio)
    
    if save:
      np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())

  return spectrogram, audio


def main(args):
  """
  Specify `extension` in `params.py`
  """
  filenames = glob(
    f'{args.dir}/**/*.{params.extension}', 
    recursive=True
  )
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train WaveGrad')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())
