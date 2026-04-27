import os
import pandas as pd
from datasets import load_dataset
import re

# ==============================
# 1. CẤU HÌNH
# ==============================
OUTPUT_DIR = "sample_data"
TEST_SIZE = 0.2
SAMPLES_PER_LABEL = 100
RANDOM_STATE = 42


# ==============================
# 2. CLEAN TEXT
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================
# 3. MAIN
# ==============================
def main():
    print("Bắt đầu tiền xử lý dữ liệu...")

    # Load dataset
    dataset = load_dataset("banking77")
    full_data = list(dataset["train"]) + list(dataset["test"])

    print(f"Tổng số mẫu: {len(full_data)}")

    # Convert to DataFrame
    df = pd.DataFrame(full_data)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    print(f"Số label: {df['label'].nunique()}")

    # ==============================
    # SAMPLE THEO TỪNG LABEL (AN TOÀN)
    # ==============================
    print("Đang lấy mẫu đều từng label...")

    balanced_list = []

    for label in df["label"].unique():
        df_label = df[df["label"] == label]

        df_sampled = df_label.sample(
            n=min(len(df_label), SAMPLES_PER_LABEL),
            random_state=RANDOM_STATE
        )

        balanced_list.append(df_sampled)

    # Gộp lại
    df_balanced = pd.concat(balanced_list).reset_index(drop=True)

    print(f"Tổng sau cân bằng: {len(df_balanced)}")

    # ==============================
    # SHUFFLE
    # ==============================
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # ==============================
    # SPLIT
    # ==============================
    split_idx = int(len(df_balanced) * (1 - TEST_SIZE))

    train_df = df_balanced[:split_idx]
    test_df = df_balanced[split_idx:]

    print(f"Train: {len(train_df)}")
    print(f"Test : {len(test_df)}")

    # ==============================
    # SAVE
    # ==============================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    print("Hoàn thành!")


if __name__ == "__main__":
    main()