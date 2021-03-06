# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Compute input examples for VGGish from audio waveform."""

from math import ceil

import numpy as np
import resampy
from scipy.io import wavfile

from .mel_features import log_mel_spectrogram, frame
from .vggish_params import *


def waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.

    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    if data.shape[0] == 0:
        data = np.zeros(ceil(sample_rate * EXAMPLE_WINDOW_SECONDS))
    elif data.shape[0] < sample_rate * EXAMPLE_WINDOW_SECONDS:
        data_length = data.shape[0]
        multiplier = ceil(sample_rate * EXAMPLE_WINDOW_SECONDS / data_length)
        for i in range(multiplier):
            data = np.concatenate((data, data[:data_length]))

    # print(sample_rate * EXAMPLE_WINDOW_SECONDS)
    # print(data.shape)
    # data = data[:int(sample_rate * EXAMPLE_WINDOW_SECONDS)]
    # Resample to the rate assumed by VGGish.
    if sample_rate != SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = log_mel_spectrogram(
        data,
        audio_sample_rate=SAMPLE_RATE,
        log_offset=LOG_OFFSET,
        window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=NUM_MEL_BINS,
        lower_edge_hertz=MEL_MIN_HZ,
        upper_edge_hertz=MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)
    return log_mel_examples


def wavfile_to_examples(wav_file):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.

    Returns:
      See waveform_to_examples.
    """
    sr, wav_data = wavfile.read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr)
