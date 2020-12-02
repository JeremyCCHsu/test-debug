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
import os
import random
import torch
import torchaudio

from glob import glob
from pathlib import Path


def short2float(x):
  return x / 32767.5


speaker_map = {
  "bdl": 1,
  "slt": 2,
  "jmk": 3,
  "clb": 4,
}

class ArcticSpeakerResolver():
  def __init__(self):
    self.speaker_map = speaker_map

  def __call__(self, filename):
    speaker = Path(filename).parent.parent.stem.split("_")[2]
    return self.speaker_map[speaker]

class SupervisedDataset(torch.utils.data.Dataset):
  """
  Arctic Podcast:
    supervised: long, speaker-heterogeneous audio recordings
    unsupervised: short, speaker-homogeneous clips
    test: short, speaker-homogeneous clips (make mixture from this set for mixture testing)
  """
  def __init__(self, paths, extension="wav"):
    super().__init__()
  
    if isinstance(paths, str) or isinstance(paths, Path):
      paths = [paths]

    self.filenames = []
    for path in paths:
      self.filenames += glob(f"{path}/**/*.{extension}", recursive=True)

    if len(self.filenames) < 1:
      raise

    self.resolve_speaker = ArcticSpeakerResolver()

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    signal, _ = torchaudio.load_wav(audio_filename)
    return {
      "audio": short2float(signal[0]),
      "speaker": speaker_map[Path(audio_filename).parent.stem]
      # "speaker": self.resolve_speaker(audio_filename)
    }


class MixtureDataset(torch.utils.data.Dataset):
  def __init__(self, paths, extension="wav"):
    super().__init__()

    if isinstance(paths, str) or isinstance(paths, Path):
      paths = [paths]

    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.{extension}', recursive=True)

    if len(self.filenames) < 1:
      raise

    self.wavs = []
    for f in self.filenames:
      wav, _ = torchaudio.load_wav(f)
      self.wavs.append(short2float(wav[0]))

    # self.resolve_speaker = ArcticSpeakerResolver()

  def __len__(self):
    return sum(len(wav) for wav in self.wavs)

  def __getitem__(self, idx):
    file_idx = idx % len(self.wavs)
    speakers = Path(self.filenames[file_idx]).stem.split("-")
    speakers = torch.LongTensor([speaker_map[s] for s in speakers])
    return {
      "audio": self.wavs[file_idx],
      "speaker": speakers,
    }


# import soundfile as sf
from shutil import copy

def make_dataset(root, output_dir, num_unsupervised=1000, num_supervised=120):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  resolver = ArcticSpeakerResolver()
  sl_train = dict()
  ul_train = dict()
  test = dict()
  Fs = 16000
  for speaker in resolver.speaker_map.keys():
    pattern = str(Path(root) / f"cmu_us_{speaker}_arctic" / "**/*.wav")
    print(pattern)
    files = glob(pattern, recursive=True)
    files.sort()

    ul_train[speaker] = files[:num_unsupervised]
    sl_train[speaker] = files[num_unsupervised: num_unsupervised + num_supervised]
    test[speaker] = files[num_unsupervised + num_supervised:]

  # ========== Mixed (unsupervised) signal ==========
  unsupervised = output_dir / "unsupervised"
  unsupervised.mkdir(exist_ok=True)
  with torch.no_grad():
    for male in ["bdl", "jmk"]:
      # male_audios = ul_train[male]
      male_audio = []
      for i, filename in enumerate(ul_train[male], 1):
        print(f"{male}: [{i}/{len(ul_train[male])}]", end="\r")
        wav, fs = torchaudio.load_wav(filename)
        assert Fs == fs
        male_audio.append(wav[0])
      male_audio = torch.cat(male_audio)

      print()
      for female in ["slt", "clb"]:
        # female_audio = ul_train[female]
        female_audio = []
        for j, filename in enumerate(ul_train[female], 1):
          print(f"{female}: [{i}/{len(ul_train[female])}]", end="\r")
          wav, fs = torchaudio.load_wav(filename)
          assert Fs == fs
          female_audio.append(wav[0])
        np.random.shuffle(female_audio)
        female_audio = torch.cat(female_audio)

        L = min(len(female_audio), len(male_audio))
        mixture = male_audio[:L] + female_audio[:L]

        # import pdb; pdb.set_trace()
        mixture = mixture / (mixture.abs().max() + 1e-6)
        torchaudio.save(unsupervised / f"{male}-{female}.wav", mixture, Fs)
        print()

  # ===== supervised subset ===============
  supervised = output_dir / "supervised"
  supervised.mkdir(exist_ok=True)
  for speaker, files in sl_train.items():
    speaker_output = supervised / speaker
    speaker_output.mkdir(exist_ok=True)
    for filename in files:
      filename = Path(filename)
      copy(filename, speaker_output / (filename.stem + ".wav") )

  # ============= test set ==============
  test_dir = output_dir / "test"
  test_dir.mkdir(exist_ok=True)
  for speaker, files in test.items():
    speaker_output = test_dir / speaker
    speaker_output.mkdir(exist_ok=True)
    for filename in files:
      filename = Path(filename)
      copy(filename, speaker_output / (filename.stem + ".wav") )

  return sl_train, ul_train, test


class SemisupervisedDataset():
  def __init__(self, path, extension="wav"):
    path = Path(path)
    self.supervised_set = SupervisedDataset(path / "supervised", extension)
    self.unsupervised_set = MixtureDataset(path / "unsupervised", extension)
  
  def __len__(self):
    return len(self.unsupervised_set)
  
  def __getitem__(self, idx):
    # i = (idx + ) % len(self.unsupervised_set)
    i = np.random.randint(0, len(self.supervised_set))
    supervised = self.supervised_set[i]
    unsupervised = self.unsupervised_set[idx]
    return {
      "audio_u": unsupervised["audio"],
      "audio_s": supervised["audio"],
      "speaker_u": unsupervised["speaker"],
      "speaker_s": supervised["speaker"]
    }


class BidatasetCollator():
  def __init__(self, params):
    self.params = params

  def __call__(self, minibatch):
    samples_per_frame = self.params.hop_samples
    audio_length = samples_per_frame * self.params.crop_mel_frames
    for record in minibatch:
      for suffix in ["_u", "_s"]:
        start = random.randint(0, len(record['audio' + suffix]) - audio_length)
        end = start + audio_length
        record['audio' + suffix] = record['audio' + suffix][start:end]
        record['audio' + suffix] = np.pad(
            record['audio' + suffix], 
            (0, audio_length - len(record['audio' + suffix])),
            mode='constant')
  
    audio_u = np.stack(
      [record['audio_u']
        for record in minibatch])
    
    audio_s = np.stack(
      [record['audio_s']
        for record in minibatch])
    
    speaker_u = np.stack(
      [record["speaker_u"]
        for record in minibatch]
    )
    
    speaker_s = np.stack(
      [record["speaker_s"]
        for record in minibatch]
    )

    return {
        'audio_u': torch.from_numpy(audio_u),
        "speaker_u": torch.from_numpy(speaker_u).long(),
        'audio_s': torch.from_numpy(audio_s),
        "speaker_s": torch.from_numpy(speaker_s).long(),
    }


class Collator2:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    audio_length = samples_per_frame * self.params.crop_mel_frames
    for record in minibatch:
      start = random.randint(0, len(record['audio']) - audio_length)
      end = start + audio_length
      record['audio'] = record['audio'][start:end]
      record['audio'] = np.pad(
          record['audio'], 
          (0, audio_length - len(record['audio'])),
          mode='constant')

    audio = np.stack(
      [record['audio']
        for record in minibatch if 'audio' in record])
    speaker = np.stack(
      [record["speaker"]
        for record in minibatch]
    )

    # speaker = torhc.from_numpy(minibatch)
    # spectrogram = np.stack([record['spectrogram']
    #                         for record in minibatch if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio),
        "speaker": torch.from_numpy(speaker).long(),
        # 'spectrogram': torch.from_numpy(spectrogram),
    }


def get_arctic(data_dirs, params):
  return torch.utils.data.DataLoader(
      SupervisedDataset(data_dirs, extension=params.extension),
      batch_size=params.batch_size,
      collate_fn=Collator2(params).collate,
      shuffle=True,
      num_workers=4)  # TODO


def get_ssl_arctic(data_dirs, params):
  # data_dirs = "/home/chincheh/datasets/ArcticPodcast"
  # ssl_ds = dataset.SemisupervisedDataset(data_dirs)+
  # collator = dataset.BidatasetCollator(params)
  return torch.utils.data.DataLoader(
      SemisupervisedDataset(data_dirs),
      batch_size=params.batch_size,
      # collate_fn=collator,
      collate_fn=BidatasetCollator(params),
      shuffle=True,
      num_workers=4)  # TODO



# ============
class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, paths, extension="wav"):
    super().__init__()
    
    if isinstance(paths, str) or isinstance(paths, Path):
      paths = [paths]

    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.{extension}', recursive=True)

    if not self.filenames:
      raise RuntimeError(f"Empty list loaded from {path}/**/*.{extension}")

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, _ = torchaudio.load_wav(audio_filename)
    spectrogram = np.load(spec_filename)
    return {
        'audio': short2float(signal[0]),  # / 32767.5,
        'spectrogram': spectrogram.T
    }



class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      # Filter out records that aren't long enough.
      if len(record['spectrogram']) < self.params.crop_mel_frames:
        del record['spectrogram']
        del record['audio']
        continue

      start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
      end = start + self.params.crop_mel_frames
      record['spectrogram'] = record['spectrogram'][start:end].T

      start *= samples_per_frame
      end *= samples_per_frame
      record['audio'] = record['audio'][start:end]
      record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio),
        'spectrogram': torch.from_numpy(spectrogram),
    }


def from_path(data_dirs, params):
  return torch.utils.data.DataLoader(
      NumpyDataset(data_dirs, extension=params.extension),
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=True,
      num_workers=os.cpu_count())
