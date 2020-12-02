
import numpy as np
import matplotlib.pyplot as plt
from wavegrad.params import params
from wavegrad.preprocess import transform
from wavegrad.inference import predict
import soundfile as sf
import torchaudio
# params = { 'noise_schedule': np.load('/path/to/noise_schedule.npy') }
from pathlib import Path


Fs = params.sample_rate
# model_dir = '../wavegrad-24kHz.pt'
model_dir = "wavegrad-16khz-libri.pt"
# device = "cuda" if torch.cuda.is_available() else "cpu"

filename = Path(
    "/data/speech/LibriSpeech/train-clean-100/103/1240/103-1240-0005.flac"
    # "/home/chincheh/datasets/BilingualNews/valid/bilin-both-01.wav"
    # "/home/chincheh/datasets/VCTK-Corpus/wav48/p225/p225_002.wav"
    # "/home/chincheh/datasets/VCTK-Corpus/wav48/p360/p360_057.wav"
    # "/data/speech/LJSpeech-1.1/wavs/LJ050-0271.wav"
    # "/home/chincheh/test/wav22k/p360_057.wav"
    # "/home/chincheh/test/wavs/p360_057.wav"
    # "/home/chincheh/datasets/Conan20/valid/Conan-Chelsea-Hilary.wav"
)

    # p292/p292_079.wav")
# get your hands on a spectrogram in [N,C,W] format
spectrogram, wav = transform(filename)





# spectrogram
# sf.write(f"input-{Fs}.wav", wav[0].detach().cpu().numpy(), Fs)
stem = filename.stem

torchaudio.save(f"input-resample-{stem}-{Fs}.wav", wav, Fs)

# audio, fs = predict(spectrogram, model_dir, params=params)

severity = 1000
# audio, fs = predict(spectrogram, model_dir, params=params, audio=wav, severity=severity)
audio, fs = predict(spectrogram, model_dir, params=params)
torchaudio.save(f"output-resample-{stem}-{Fs}-denoised-{severity}.wav", audio[0].detach().cpu(), Fs)


# # ======================
# from_sox = np.load("/home/chincheh/test/wav22k/p360_057.wav.spec.npy")

# fig, ax = plt.subplots(3, 1)

# im = ax[0].imshow(from_sox, origin="lower")
# fig.colorbar(im, ax=ax[0])

# im = ax[1].imshow(spectrogram.numpy(), origin="lower")
# fig.colorbar(im, ax=ax[1])

# im = ax[2].imshow(from_sox - spectrogram.numpy(), origin="lower")
# fig.colorbar(im, ax=ax[2])

# fig.savefig("test.png")
# plt.close()
# # ======================
