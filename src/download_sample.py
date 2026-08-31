from pathlib import Path

import pandas as pd
import requests


CSV_PATH = Path("data/raw/train.csv")
OUTPUT_DIR = Path("data/sample_audio")
MANIFEST_PATH = Path("data/sample_manifest.csv")

NUM_CLASSES = 5
SAMPLES_PER_CLASS = 2


def main():
    df = pd.read_csv(CSV_PATH)

    # 首先只使用可直接下载的 iNaturalist MP3 链接。
    direct = df[
        df["url"]
        .fillna("")
        .str.contains("static.inaturalist.org/sounds/", regex=False)
    ].copy()

    selected_labels = direct["primary_label"].drop_duplicates().head(NUM_CLASSES)
    selected = (
        direct[direct["primary_label"].isin(selected_labels)]
        .groupby("primary_label", group_keys=False)
        .head(SAMPLES_PER_CLASS)
        .copy()
    )

    downloaded_rows = []

    for _, row in selected.iterrows():
        label = str(row["primary_label"])
        original_stem = Path(row["filename"]).stem
        output_path = OUTPUT_DIR / label / f"{original_stem}.mp3"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"下载: {label}/{output_path.name}")

        try:
            response = requests.get(row["url"], timeout=120)
            response.raise_for_status()
            output_path.write_bytes(response.content)

            saved_row = row.to_dict()
            saved_row["local_path"] = str(output_path)
            downloaded_rows.append(saved_row)

            print(f"  成功: {len(response.content) / 1024:.1f} KB")
        except requests.RequestException as error:
            print(f"  失败: {error}")

    manifest = pd.DataFrame(downloaded_rows)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print()
    print(f"成功下载 {len(manifest)} 条音频")
    print(f"清单保存到: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

