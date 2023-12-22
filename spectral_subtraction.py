import numpy as np
import librosa
import soundfile as sf
import argparse
from scipy.signal import butter, filtfilt

def low_pass_filter(signal, sr, cutoff=8000, order=5):
    """
    Apply a low-pass filter to the signal.

    :param signal: Input signal.
    :param sr: Sampling rate of the signal.
    :param cutoff: Cutoff frequency of the low-pass filter.
    :param order: Order of the filter.
    :return: Filtered signal.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def spectral_subtraction(wav_path, output_path, noise_reduction=2.0, n_iterations=10, noise_estimation_duration=1, flooring_factor=0.7):
    """
    Apply spectral subtraction to each channel of a WAV file.

    :param wav_path: Path to the input WAV file.
    :param output_path: Path to save the output WAV file.
    :param noise_reduction: Noise reduction factor.
    :param n_iterations: Number of iterations to apply spectral subtraction.
    :param noise_estimation_duration: Duration in seconds for noise estimation.
    :param flooring_factor: Flooring factor for spectral subtraction.
    """
    # Load the WAV file
    y, sr = librosa.load(wav_path, sr=None, mono=False)

    # Process each channel separately
    if y.ndim > 1:
        y_denoised = [None] * y.shape[0]
        for i in range(y.shape[0]):
            y_denoised[i] = _denoise_channel(y[i], sr, noise_reduction, n_iterations, noise_estimation_duration, flooring_factor)
        y_denoised = np.vstack(y_denoised)
    else:
        y_denoised = _denoise_channel(y, sr, noise_reduction, n_iterations, noise_estimation_duration, flooring_factor)

    # Save the output
    sf.write(output_path, y_denoised.T, sr)

def _denoise_channel(channel, sr, noise_reduction, n_iterations, noise_estimation_duration, flooring_factor):
    """
    Denoise a single channel of audio using spectral subtraction, applied multiple times.
    Noise is estimated from the first specified duration of the audio.

    :param channel: Single-channel audio data.
    :param sr: Sampling rate of the audio.
    :param noise_reduction: Noise reduction factor.
    :param n_iterations: Number of iterations to apply spectral subtraction.
    :param noise_estimation_duration: Duration in seconds for noise estimation.
    :param flooring_factor: Flooring factor for spectral subtraction.
    :return: Denoised audio data for the channel.
    """

    channel = low_pass_filter(channel, sr)

    for _ in range(n_iterations):
        # Compute the short-time Fourier transform (STFT)
        stft = librosa.stft(channel)

        # Calculate the number of frames for the noise estimation duration
        noise_frames = int(noise_estimation_duration * sr / stft.shape[1])

        # Use the first few frames for noise estimation
        noise_est = np.mean(np.abs(stft[:, :noise_frames]), axis=1)

        # Subtract the estimated noise power spectrum with flooring
        denoised_stft = np.maximum(np.abs(stft) - noise_reduction * noise_est[:, np.newaxis], np.abs(stft) * flooring_factor) * np.exp(1j * np.angle(stft))

        # Compute the inverse STFT
        channel = librosa.istft(denoised_stft)

    return channel

def main():
    parser = argparse.ArgumentParser(description="Noise reduction using spectral subtraction.")
    parser.add_argument("input_file", type=str, help="Input WAV file path.")
    parser.add_argument("output_file", type=str, help="Output WAV file path.")
    parser.add_argument("-nr", "--noise_reduction", type=float, default=2.0, help="Noise reduction factor. Default is 2.0.")
    parser.add_argument("-ni", "--n_iterations", type=int, default=10, help="Number of iterations to apply spectral subtraction. Default is 10.")
    parser.add_argument("-ned", "--noise_estimation_duration", type=float, default=1, help="Duration in seconds for noise estimation. Default is 1.")
    parser.add_argument("-ff", "--flooring_factor", type=float, default=0.7, help="Flooring factor for spectral subtraction. Default is 0.7.")
    args = parser.parse_args()

    spectral_subtraction(args.input_file, args.output_file, args.noise_reduction, args.n_iterations, args.noise_estimation_duration, args.flooring_factor)

if __name__ == "__main__":
    main()
