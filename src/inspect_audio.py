from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MANIFEST_PATH = Path("data/sample_manifest.csv")
OUTPUT_PATH = Path("outputs/first_sample.png")

SAMPLE_RATE = 32000
DURATION_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * DURATION_SECONDS


def crop_or_pad(audio):
    if len(audio) >= NUM_SAMPLES:
        return audio[:NUM_SAMPLES]

    return np.pad(audio, (0, NUM_SAMPLES - len(audio)))


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    row = manifest.iloc[0]

    audio_path = Path(row["local_path"])
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    original_duration = len(audio) / SAMPLE_RATE
    audio = crop_or_pad(audio)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=20,
        fmax=16000,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7))

    librosa.display.waveshow(audio, sr=SAMPLE_RATE, ax=axes[0])
    axes[0].set_title(
        f"Waveform | label={row['primary_label']} | "
        f"original duration={original_duration:.2f}s"
    )

    image = librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmin=20,
        fmax=16000,
        ax=axes[1],
    )
    axes[1].set_title("Log-Mel Spectrogram")
    figure.colorbar(image, ax=axes[1], format="%+2.0f dB")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)

    print(f"音频文件: {audio_path}")
    print(f"标签: {row['primary_label']}")
    print(f"原始时长: {original_duration:.2f} 秒")
    print(f"模型输入: {len(audio)} samples @ {SAMPLE_RATE} Hz")
    print(f"Mel形状: {mel_db.shape}")
    print(f"图片保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

