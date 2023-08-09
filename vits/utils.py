import torch
import numpy as np
from scipy.io.wavfile import read


def load_class(full_class_name):
    cls = None
    if full_class_name in globals():
        cls = globals()[full_class_name]
    else:
        if "." in full_class_name:
            import importlib
            module_name, cls_name = full_class_name.rsplit('.', 1)
            mod = importlib.import_module(module_name)
            cls = (getattr(mod, cls_name))
    return cls


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * \
        np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * \
        (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (
        f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min(
    ) >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse