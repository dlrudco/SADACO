import os
from typing import List, Optional, Tuple

import cv2
import librosa
import numpy as np
import torch
import torchaudio.transforms as AT
import torchvision.transforms as VT
from torch.utils.data import Dataset

cv2.setNumThreads(0)


class FraiwanDataset(Dataset):
    r"""A `torch` dataset class for Fraiwan dataset that loads audio file and returns a tuple of sample (spectrogram, label) when called. 

    Args:
        data_list: List of audio file names.
        label_list: List of labels corresponding to each audio file.
        data_dir: Directory where audio files are stored.
        train: Flag to indicate if the dataset is for training or testing.
        output_dim: Output dimension of the spectrogram. `(num_mel, num_time_frames)` (default: (32, 70))
        get_weights: Flag to indicate if weights are needed for weighted loss function. (default: False)
        sample_rate: Target sample rate of the audio files. (default: 16000)
        nfft: Target number of FFT points. (default: 401)
        hop_length: Target hop length. (default: 160)
        time_mask: Time masking parameter. (default: 24)
        freq_mask: Frequency masking parameter. (default: 24)
        mixup_alpha: Mix Up alpha parameter. (default: 0.0)
    """
    def __init__(
        self,
        data_list: List[str],
        label_list: List[int],
        data_dir: str,
        train: bool,
        output_dim: Tuple[int] = (32, 70),
        get_weights: bool = False,
        sample_rate: int = 16000,
        nfft: int = 401,
        hop_length: int = 160,
        time_mask: int = 24,
        freq_mask: int = 24,
        mixup_alpha: float = 0.0,
    ) -> None:
        super().__init__()

        assert len(data_list) > 0, "Data list is empty."
        assert len(data_list) == len(label_list), "Data list and label list must have the same length."

        self.data_list = data_list
        self.label_list = label_list
        self.data_dir = data_dir
        self.train = train
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.hop_length = hop_length
        self.augmentation = self._get_augmentation(time_mask=time_mask, freq_mask=freq_mask)
        self.mixup_alpha = mixup_alpha
        self.num_classes = len(set(self.label_list))
        
        self.normed_weights = self._get_weights() if get_weights else None

    def _get_augmentation(self, time_mask: Optional[int], freq_mask: Optional[int]) -> VT.Compose:
        r"""Compile both vision and audio transformations/ augmentations into `VT.Compose`.
        
        Flag:
            - `self.train`: If True, apply time masking, frequency masking, and random crop. Otherwise, apply only center crop.
        """
        augmentation =[VT.ToTensor()]
        if self.train:
            augmentation += [
                AT.TimeMasking(time_mask),
                AT.FrequencyMasking(freq_mask),
                VT.RandomCrop(
                    size=self.output_dim,
                    pad_if_needed=True,
                    padding_mode="constant"
                ),
            ]
        else:
            augmentation += [
                VT.CenterCrop(
                    size=self.output_dim,
                ),
            ]
        return VT.Compose(augmentation)

    def _get_weights(self) -> torch.Tensor:
        r"""Get normalised weights for weighted loss function if needed."""
        weights = [0.0] * self.num_classes
        for label in self.label_list:
            weights[label] += 1.0
        normed_weights = [1.0 - (w / sum(weights)) for w in weights]
        return torch.as_tensor(normed_weights)

    def _right_pad_if_necessary(self, mel_spec: np.ndarray) -> np.ndarray:
        r"""Pad a spectrogram with zeros at time axis (using `cv2`) if it is shorter than the output dimension. Otherwise, returns the spectrogram as is.
        
        Args:
            mel_spec: (Mel) Spectrogram to be padded.
        """
        _, w = mel_spec.shape
        if w < self.output_dim[1]:
            w_to_pad = self.output_dim[1] - w
            padded_mel_spec = cv2.copyMakeBorder(
                mel_spec,
                0,
                0,
                0,
                w_to_pad,
                cv2.BORDER_CONSTANT,
                (0, 0),
            )
            return padded_mel_spec
        return mel_spec

    def _align_time_axis(self, sig_1: np.ndarray, sig_2: np.ndarray) -> np.ndarray:
        r"""Align the time axis of two signals.
        
        Args:
            sig_1: First signal.
            sig_2: Second signal.
        """
        sig_1_length = sig_1.shape[0]
        sig_2_length = sig_2.shape[0]
        if sig_1_length > sig_2_length:
            pad_length = (sig_1_length // sig_2_length) + 1
            sig_2 = sig_2.repeat(pad_length, axis=0)
            sig_2 = sig_2[:sig_1_length]
        elif sig_1_length < sig_2_length:
            start_time = np.random.randint(0, sig_2_length - sig_1_length)
            sig_2 = sig_2[start_time : start_time + sig_1_length]
        return sig_2

    def _get_mixup_sample(self, data_path_1: str, data_path_2: str) -> Tuple[torch.Tensor, float]:
        r"""Performs mixup on two audio files.
        
        Args: 
            data_path_1: First audio datapath. 
            data_path_2: Second audio datapath. 

        Returns:
            Tuple of `(mixup_signal, mixup_lambda)`
        """
        sig_1, _ = librosa.load(data_path_1, sr=self.sample_rate)
        if np.random.random() < 0.5:
            sig_2, _ = librosa.load(data_path_2, sr=self.sample_rate)
            sig_2 = self._align_time_axis(sig_1, sig_2)
            
            mixup_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            mixup_signal = mixup_lambda * sig_1 + (1 - mixup_lambda) * sig_2
        else:
            mixup_signal = sig_1
            mixup_lambda = 1.0
        return mixup_signal, mixup_lambda


    def sig2spec(self, signal: np.ndarray) -> torch.Tensor:
        r"""Convert a signal to a spectrogram using `librosa` with target prameterss such as `n_mels`, `nfft`, `hop_length`, and `sample_rate`.

        Args:
            signal: Signal to be converted into spectrogram.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_mels=self.output_dim[0],
            n_fft=self.nfft,
            hop_length=self.hop_length,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        norm_mel_spec = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        if not self.train:
            norm_mel_spec = self._right_pad_if_necessary(norm_mel_spec)
        final_mel_spec = self.augmentation(norm_mel_spec)
        return final_mel_spec

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        data_path = os.path.join(self.data_dir, self.data_list[idx])
        label = self.label_list[idx]
        if self.train and self.mixup_alpha > 0.0:
            mix_sample_idx = np.random.randint(0, len(self.data_list) - 1)
            data_path_2 = os.path.join(self.data_dir, self.data_list[mix_sample_idx])
            label_2 = self.label_list[mix_sample_idx]
            mixup_signal, mixup_lambda = self._get_mixup_sample(data_path, data_path_2)
            mixup_spectrogram = self.sig2spec(mixup_signal)
            return mixup_spectrogram, torch.as_tensor(label), torch.as_tensor(label_2), torch.as_tensor(mixup_lambda)
        else:
            signal, _ = librosa.load(data_path, sr=self.sample_rate)
            spectrogram = self.sig2spec(signal)
            return spectrogram, torch.as_tensor(label)
        