import os
import pickle

import librosa
import soundfile as sf


def load_pickle(file_path):
    """Load a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, file_path, info_show=True):
    """Save a pickle file, creating parent directories when necessary."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    if info_show:
        print(f'pickle保存路径{file_path}')


def load(path, return_sr=False, resample=48000):
    """Load an audio file as a numpy array; channel dimension is preserved."""
    sr = sf.SoundFile(path).samplerate
    audio = librosa.load(path, sr=sr, mono=False)[0]
    if resample and sr != resample:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=resample)
        sr = resample
    if return_sr:
        return audio, sr
    return audio
