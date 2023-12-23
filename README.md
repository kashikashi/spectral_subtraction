Spectral Subtraction Noise Reduction
====================================

This repository contains a Python script for noise reduction in WAV files using the spectral subtraction method. It reduces noise from audio signals. This implementation is inspired by the research in the paper "A study of musical tone reduction on iterative spectral subtraction based on higher flooring parameters" by Takahiro Fukumori.

Features
--------

- Supports multi-channel WAV files
- Noise reduction using spectral subtraction
- Easy to use from the command line

Prerequisites
-------------

Before running the script, ensure the following libraries are installed:

- numpy
- scipy
- librosa
- soundfile

These can be installed using the `requirements.txt` file.

Installation
------------

After cloning or downloading this repository, install the required libraries with the following command:

    pip install -r requirements.txt

Usage
-----

The script can be executed from the command line in the following format:

    python spectral_subtraction.py input.wav output.wav --noise_estimation_duration 1

Where:
- `input.wav` is the path to the WAV file to be processed.
- `output.wav` is the path where the noise-reduced audio will be saved.
- `--noise_reduction` is the noise reduction coefficient (optional, default is 2.0).
- `--n_iterations` is the number of times to apply spectral subtraction (optional, default is 10).
- `--noise_estimation_duration` is the duration in seconds for noise estimation (default: 1).
- `--flooring_factor` is the flooring factor for spectral subtraction (default: 0.7).

Reference
---------

This implementation is based on the techniques discussed in the paper "A study of musical tone reduction on iterative spectral subtraction based on higher flooring parameters" by Takahiro Fukumori. The paper can be accessed at [this link](https://www.ieice.org/publications/conference-FIT-DVDs/FIT2010/pdf/E/E_002.pdf).

License
-------

This project is published under the [MIT License](LICENSE).
