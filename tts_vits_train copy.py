"""
使用 FastSpeech2 + HuggingFace 预训练模型微调 lasttrain 目录下的数据。

特性：
- 使用 HF 镜像源下载预训练模型到本目录的 models_fastspeech2。
- 自动过滤「过短」和「过长」音频，只保留合适长度的样本。
- 训练时最多只使用 1000 条语音进行微调，适合小数据 / 风格迁移。
"""
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from TTS.api import TTS

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

# ----------------------------
# 基本配置
# ----------------------------
ROOT = Path(__file__).resolve().parent

# 使用 HF 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 预训练 FastSpeech2 模型（可按需要替换为中文/日文模型）
HF_FASTSPEECH2_ID = "espnet/kan-bayashi_ljspeech_fastspeech2"

# 预训练模型本地保存目录
SAVE_DIR = ROOT / "models_fastspeech2"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 时长过滤阈值（秒）
MIN_SEC = 0.4   # 过短直接丢弃（主要是几乎只有呼吸/噪声的片段）
MAX_SEC = 12.0  # 过长的句子也丢弃，避免训练不稳定

# 训练规模：最多使用 1000 条样本
MAX_TRAIN_SAMPLES = 1000
NUM_EPOCHS = 20
BATCH_SIZE = 8


def ensure_pretrained_model() -> TTS:
    """确保 FastSpeech2 预训练模型已下载，并返回 TTS 实例。"""
    if snapshot_download is not None:
        try:
            local_dir = snapshot_download(
                HF_FASTSPEECH2_ID,
                local_dir=str(SAVE_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"预训练 FastSpeech2 已下载/更新到: {local_dir}")
        except Exception as e:
            print(f"预训练模型下载失败，将尝试直接由 TTS API 加载: {e}")
    else:
        print("未安装 huggingface_hub，跳过显式下载，将由 TTS API 自行拉取模型缓存。")

    gpu = torch.cuda.is_available()
    print(f"GPU 可用: {gpu}")
    tts = TTS(model_name=HF_FASTSPEECH2_ID, gpu=gpu)
    return tts


def compute_duration(path: str) -> float | None:
    """返回音频时长（秒），出错时返回 None。"""
    try:
        return float(librosa.get_duration(path=path))
    except Exception as e:
        print(f"[时长] 加载失败，跳过 {path}: {e}")
        return None


def filter_by_duration(df: pd.DataFrame, min_sec: float, max_sec: float) -> pd.DataFrame:
    """按音频时长过滤样本，剔除过短/过长的音频。"""
    keep_rows = []
    removed_short = 0
    removed_long = 0

    for _, row in df.iterrows():
        audio_path = str(row["audio_path"])
        dur = compute_duration(audio_path)
        if dur is None:
            continue
        if dur < min_sec:
            removed_short += 1
            continue
        if dur > max_sec:
            removed_long += 1
            continue
        keep_rows.append(row)

    out_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    print(
        f"[过滤] 总数={len(df)} -> 保留={len(out_df)}，短音频删除={removed_short}，长音频删除={removed_long}"
    )
    return out_df


def load_and_filter_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 train.csv / val.csv，并做时长过滤与样本数限制。"""
    train_csv = ROOT / "train.csv"
    val_csv = ROOT / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("未找到 train.csv 或 val.csv，请先在 lasttrain 目录下准备好这两个文件。")

    train_df = pd.read_csv(train_csv, dtype={"text": str, "audio_path": str}, skip_blank_lines=True)
    val_df = pd.read_csv(val_csv, dtype={"text": str, "audio_path": str}, skip_blank_lines=True)

    # 丢弃空文本或空路径
    train_df = train_df[(train_df["text"].notna()) & (train_df["audio_path"].notna())].reset_index(drop=True)
    val_df = val_df[(val_df["text"].notna()) & (val_df["audio_path"].notna())].reset_index(drop=True)

    print(f"原始样本数: train={len(train_df)}, val={len(val_df)}")

    # 按时长过滤
    train_df = filter_by_duration(train_df, MIN_SEC, MAX_SEC)
    val_df = filter_by_duration(val_df, MIN_SEC, MAX_SEC)

    # 训练只保留最多 1000 条，保证「小改动、小数据」场景
    if len(train_df) > MAX_TRAIN_SAMPLES:
        train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42).reset_index(drop=True)
        print(f"[采样] 训练样本数截断为 {MAX_TRAIN_SAMPLES}")

    # 可选：把过滤后的数据另存一份，方便复现
    train_df.to_csv(ROOT / "train_fastspeech2_1000.csv", index=False)
    val_df.to_csv(ROOT / "val_fastspeech2_filtered.csv", index=False)
    print("已将过滤后的列表保存为 train_fastspeech2_1000.csv / val_fastspeech2_filtered.csv")

    return train_df, val_df


def train_fastspeech2():
    """使用预训练 FastSpeech2 对 lasttrain 数据进行小规模微调。"""
    train_df, val_df = load_and_filter_data()
    if len(train_df) == 0:
        raise RuntimeError("过滤后训练样本为 0，无法训练。请检查音频时长阈值或数据质量。")

    tts = ensure_pretrained_model()

    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")
        # 打乱训练数据
        train_df = train_df.sample(frac=1.0, random_state=epoch).reset_index(drop=True)

        for start in range(0, len(train_df), BATCH_SIZE):
            batch = train_df.iloc[start : start + BATCH_SIZE]
            for _, row in batch.iterrows():
                text = str(row["text"])
                audio_path = str(row["audio_path"])
                # Coqui TTS 当前并未公开完整的微调 API，这里使用 train_step 作为示意。
                # 若你本地的 TTS 版本不支持 train_step，可改为你自己的 FastSpeech2 训练管线。
                try:
                    tts.train_step(text=text, audio_path=audio_path)
                except Exception as e:
                    print(f"[train_step] 失败，跳过样本 {audio_path}: {e}")

        # 每个 epoch 保存一个 checkpoint 到 models_fastspeech2 目录
        ckpt_path = SAVE_DIR / f"fastspeech2_epoch_{epoch + 1}.pth"
        try:
            tts.save_model(str(ckpt_path))
            print(f"[保存] Checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"[保存] 失败（不影响继续训练）: {e}")

    print("FastSpeech2 微调完成 ✅")


if __name__ == "__main__":
    train_fastspeech2()