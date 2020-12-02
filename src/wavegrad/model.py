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
import torch.nn as nn
import torch.nn.functional as F

from math import log as ln


CONTENT_EMB_DIM = 32
SPEAKER_EMB_DIM = 96
N_SPEAKER = 5

class Conv1d(nn.Conv1d):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.weight)
    nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x, noise_level):
    """
    Arguments:
      x:
          (shape: [N,C,T], dtype: float32)
      noise_level:
          (shape: [N], dtype: float32)

    Returns:
      noise_level:
          (shape: [N,C,T], dtype: float32)
    """
    N = x.shape[0]
    T = x.shape[2]
    return (x + self._build_encoding(noise_level)[:, :, None])

  def _build_encoding(self, noise_level):
    count = self.dim // 2
    step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
    encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
    encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
    return encoding


class FiLM(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.encoding = PositionalEncoding(input_size)
    self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
    self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.input_conv.weight)
    nn.init.xavier_uniform_(self.output_conv.weight)
    nn.init.zeros_(self.input_conv.bias)
    nn.init.zeros_(self.output_conv.bias)

  def forward(self, x, noise_scale):
    x = self.input_conv(x)
    x = F.leaky_relu(x, 0.2)
    x = self.encoding(x, noise_scale)
    shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
    return shift, scale


class UBlock(nn.Module):
  def __init__(self, input_size, hidden_size, factor, dilation):
    super().__init__()
    assert isinstance(dilation, (list, tuple))
    assert len(dilation) == 4

    self.factor = factor
    self.block1 = Conv1d(input_size, hidden_size, 1)
    self.block2 = nn.ModuleList([
        Conv1d(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
        Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1])
    ])
    self.block3 = nn.ModuleList([
        Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
        Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3])
    ])

  def forward(self, x, film_shift, film_scale):
    block1 = F.interpolate(x, size=x.shape[-1] * self.factor)
    block1 = self.block1(block1)

    block2 = F.leaky_relu(x, 0.2)
    block2 = F.interpolate(block2, size=x.shape[-1] * self.factor)
    block2 = self.block2[0](block2)
    block2 = film_shift + film_scale * block2
    block2 = F.leaky_relu(block2, 0.2)
    block2 = self.block2[1](block2)

    x = block1 + block2

    block3 = film_shift + film_scale * x
    block3 = F.leaky_relu(block3, 0.2)
    block3 = self.block3[0](block3)
    block3 = film_shift + film_scale * block3
    block3 = F.leaky_relu(block3, 0.2)
    block3 = self.block3[1](block3)

    x = x + block3
    return x


class DBlock(nn.Module):
  def __init__(self, input_size, hidden_size, factor):
    super().__init__()
    self.factor = factor
    self.residual_dense = Conv1d(input_size, hidden_size, 1)
    self.conv = nn.ModuleList([
        Conv1d(input_size, hidden_size, 3, dilation=1, padding=1),
        Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
        Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
    ])

  def forward(self, x):
    size = x.shape[-1] // self.factor

    residual = self.residual_dense(x)
    residual = F.interpolate(residual, size=size)

    x = F.interpolate(x, size=size)
    for layer in self.conv:
      x = F.leaky_relu(x, 0.2)
      x = layer(x)

    return x + residual


class WaveGrad(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    self.downsample = nn.ModuleList([
        Conv1d(1, 32, 5, padding=2),
        DBlock(32, 128, 2),
        DBlock(128, 128, 2),
        DBlock(128, 256, 3),
        DBlock(256, 512, 5),
    ])
    self.film = nn.ModuleList([
        FiLM(32, 128),
        FiLM(128, 128),
        FiLM(128, 256),
        FiLM(256, 512),
        FiLM(512, 512),
    ])
    self.upsample = nn.ModuleList([
        UBlock(768, 512, 5, [1, 2, 1, 2]),
        UBlock(512, 512, 5, [1, 2, 1, 2]),
        UBlock(512, 256, 3, [1, 2, 4, 8]),
        UBlock(256, 128, 2, [1, 2, 4, 8]),
        UBlock(128, 128, 2, [1, 2, 4, 8]),
    ])
    self.first_conv = Conv1d(128, 768, 3, padding=1)
    self.last_conv = Conv1d(128, 1, 3, padding=1)

  def forward(self, audio, spectrogram, noise_scale):
    # FIXME: EXPERIMENTAL: UNCONDITIONAL ===============
    spectrogram = 0 * spectrogram
    # ==================================================

    x = audio.unsqueeze(1)
    downsampled = []
    for film, layer in zip(self.film, self.downsample):
      x = layer(x)
      downsampled.append(film(x, noise_scale))

    x = self.first_conv(spectrogram)
    for layer, (film_shift, film_scale) in zip(self.upsample, reversed(downsampled)):
      x = layer(x, film_shift, film_scale)
    x = self.last_conv(x)
    return x


# Jadd =======================================
class SpeakerFiLM(nn.Module):
  def __init__(self, output_size, n_speakers=N_SPEAKER):
    super().__init__()
    self.shift = nn.Embedding(n_speakers, output_size)
    self.scale = nn.Embedding(n_speakers, output_size)

  def forward(self, x, index):
    """
    :param x: [B, ]
    """
    shift = self.shift(index).unsqueeze(-1)
    scale = self.scale(index).unsqueeze(-1)
    return scale * x + shift


class DownCleansingBlock(nn.Module):
  def __init__(self, input_size, hidden_size, factor):
    super().__init__()
    self.factor = factor
    self.residual_dense = Conv1d(input_size, hidden_size, 1)
    self.conv = nn.ModuleList([
        Conv1d(input_size, hidden_size, 3, dilation=1, padding=1),
        Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
        Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
    ])
    self.instance_norm = nn.ModuleList([
        nn.InstanceNorm1d(hidden_size, affine=False),
        nn.InstanceNorm1d(hidden_size, affine=False),
        nn.InstanceNorm1d(hidden_size, affine=False),
    ])

  def forward(self, x):
    size = x.shape[-1] // self.factor

    residual = self.residual_dense(x)
    residual = F.interpolate(residual, size=size)

    x = F.interpolate(x, size=size)
    for layer, normalize in zip(self.conv, self.instance_norm):
      x = normalize(x)
      x = F.leaky_relu(x, 0.2)
      x = layer(x)

    return x + residual

class ContentEncoder(nn.Module):
  """
  Enc(waveform, speaker) --> phone embedding
  """
  def __init__(self):  #, params):
    super().__init__()
    # self.params = params
    self.downsample = nn.ModuleList([
        Conv1d(1, 32, 5, padding=2),
        DownCleansingBlock(32, 128, 2),
        DownCleansingBlock(128, 128, 2),
        DownCleansingBlock(128, 256, 3),
        DownCleansingBlock(256, 512, 5),
        DownCleansingBlock(512, CONTENT_EMB_DIM, 5),
    ])
    self.film = nn.ModuleList([
        SpeakerFiLM(32,),
        SpeakerFiLM(128),
        SpeakerFiLM(128),
        SpeakerFiLM(256),
        SpeakerFiLM(512),
        SpeakerFiLM(CONTENT_EMB_DIM),
    ])
    # self.first_conv = Conv1d(128, 768, 3, padding=1)
    # self.last_conv = Conv1d(128, 1, 3, padding=1)

  def forward(self, audio, speaker):
    """
    :param audio: [B, T]
    :param speaker: [B,]
    """
    x = audio.unsqueeze(1)
    # downsampled = []
    for film, layer in zip(self.film, self.downsample):
      x = layer(x)
      
      # shift, scale = film(x)
      # x = shift + scale * x
      x = film(x, speaker)

      # downsampled.append(x)

    # x = self.first_conv(spectrogram)
    # for layer, (film_shift, film_scale) in zip(self.upsample, reversed(downsampled)):
    #   x = layer(x, film_shift, film_scale)
    # x = self.last_conv(x)
    return x


class LatentEmbedding(nn.Module):
  """
  Enc(waveform, speaker) --> phone embedding
  """
  def __init__(self):  #, params):
    super().__init__()
    self.speaker_emb = nn.Embedding(N_SPEAKER, SPEAKER_EMB_DIM)

    # self.conv = nn.ModuleList([
    #   nn.Conv1d()
    # ])
    dim = SPEAKER_EMB_DIM + CONTENT_EMB_DIM

    self.conv = nn.Sequential(
      nn.Conv1d(dim, 256, 3, padding=1),
      nn.LeakyReLU(0.2),
      nn.Conv1d(256, dim, 3, padding=1)
    )

    final_layer = nn.Conv1d(256, dim, 3, padding=1)
    final_layer.bias.data.fill_(2.)
    # torch.nn.init.xavier_uniform(conv1.weight)
    self.gate = nn.Sequential(
      nn.Conv1d(dim, 256, 3, padding=1),
      nn.LeakyReLU(0.2),
      # nn.Conv1d(256, dim, 3, padding=1),
      final_layer,
      nn.Sigmoid()
    )

    # augment  -> conv -> ReLU -> conv -> linear  = conv
    #         \-> conv -> ReLU -> conv -> sigmoid = gate
    # gate * conv

  def forward(self, content_emb, speaker):
    """
    :param content_emb: [B, c, L]
    :param speaker: [B]
    """
    L = content_emb.shape[2]
    
    speaker_emb = self.speaker_emb(speaker).unsqueeze(-1).repeat([1, 1, L])
    embeddings = torch.cat([content_emb, speaker_emb], 1)
    conv = self.conv(embeddings)
    gate = self.gate(embeddings)
    return gate * conv


class WaveVCTraining(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    self.wavegrad = WaveGrad(params)
    self.fuse_latent = LatentEmbedding()
    self.encoder = ContentEncoder()

  def encode(self, audio, source_speakers):
    """
    :param audio: [B, T]
    :param speaker: [B, (S)]

    :return pseudospecs: [S, B, c, T]
    """
    if len(source_speakers.shape) == 1:
      source_speakers = source_speakers.unsqueeze(-1)
      # target_speakers = target_speakers.unsqueeze(-1)

    # pseudospecs = []
    content_emb = []
    n_speaker = source_speakers.shape[1]
    for i in range(n_speaker):
      content = self.encoder(audio, source_speakers[:, i])
      content_emb.append(content)
      # pseudospec = self.fuse_latent(content, target_speakers[:, i])
      # pseudospecs.append(pseudospec)

    # return pseudospecs
    return content_emb

  def sample(self, content):
    """
    :param content: [b, 2c, T] c for mean and log var, respectively.
    """

    torch.split()

  def blende(self, content, speaker):
    # equiv to `fuse_latent`
    pass

  def forward(self, audio, speaker, noise_scale):
    if len(speaker.shape) == 1:
      speaker = speaker.unsqueeze(-1)

    embeddings = []
    n_speaker = speaker.shape[1]
    for i in range(n_speaker):
      content = self.encoder(audio, speaker[:, i])
      pseudospec = self.fuse_latent(content, speaker[:, i])
      embeddings.append(pseudospec)

    pseudospec = sum(embeddings)

    # NOTE: the output is grad log p(x|z, y)
    output = self.wavegrad(audio, pseudospec, noise_scale)
    return output


class WaveConvert(WaveVCTraining):
  def __init__(self, params):
    super().__init__(params)
  
  def decode(self, audio, pseudospecs, noise_scale):  # speaker,
    """
    :param audio: [B, T]
    :param pseudospecs: [S, B, c, T]
        # :param speaker: [B, (S)]

    :return gradient: [B, c=1, T]
    """
    # if len(speaker.shape) == 1:
    #   speaker = speaker.unsqueeze(-1)
    #   assert speaker.shape[1] == len(pseudospecs)

    # TODO: maybe change its format
    pseudospecs = torch.stack(pseudospecs)
    n_speakers, batch_size, c, n_frames = pseudospecs.shape
    pseudospecs = pseudospecs.view(-1, c, n_frames)

    _, time_length = audio.shape
    audio = audio.unsqueeze(1).repeat(1, n_speakers, 1).view(-1, time_length)

    # speaker = speaker.repeat(batch_size, 1).view(-1)

    return self.wavegrad(audio, pseudospecs, noise_scale)

    # output = []
    # n_speaker = speaker.shape[1]
    # for i in range(n_speaker):
    #   output = self.wavegrad(audio, pseudospecs[i], noise_scale)


  def encode(self, audio, source_speakers, target_speakers):
    """
    :param audio: [B, T]
    :param speaker: [B, (S)]

    :return pseudospecs: [S, B, c, T]
    """
    if len(source_speakers.shape) == 1:
      source_speakers = source_speakers.unsqueeze(-1)
      target_speakers = target_speakers.unsqueeze(-1)

    pseudospecs = []
    n_speaker = source_speakers.shape[1]
    for i in range(n_speaker):
      content = self.encoder(audio, source_speakers[:, i])
      pseudospec = self.fuse_latent(content, target_speakers[:, i])
      pseudospecs.append(pseudospec)

    return pseudospecs
