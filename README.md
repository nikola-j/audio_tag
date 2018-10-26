# Audiotag

Automatic .wav file audio tagging using VGGish.


## Installation

Install requirements using:
```
pip install -r requiremnts.txt
```

Download VGGish model:
```
# Download data files into the audioset directory
cd audioset
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
```
Test if VGGish is working:
```
# Installation ready, let's test it (from audioset).
python vggish_smoke_test.py
# If we see "Looks Good To Me", then we're all set.
```

## Dataset

Download dataset from this competition: https://www.kaggle.com/c/freesound-audio-tagging


## External code

Audioset used from here: https://github.com/tensorflow/models

Batched vggish inference used from here: https://github.com/knstmrd/vggish-batch
