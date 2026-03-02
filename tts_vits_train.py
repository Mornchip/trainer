#!/usr/bin/env python3
"""
FastSpeech2 训练脚本 - 修复版
"""
import os
import sys
import torch
import pandas as pd
from pathlib import Path

# 配置
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models_fastspeech2"
OUTPUT_DIR = ROOT / "tts_train_output"

# 确保目录存在
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU 设置
NUM_GPUS = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu = torch.cuda.is_available()
print(f"GPU可用: {gpu}")

# 训练参数
EPOCHS = 1000
BATCH_SIZE = 32
SAVE_STEP = 500
PRINT_STEP = 50

def download_model():
    """下载 FastSpeech2 预训练模型"""
    from TTS.utils.manage import ModelManager
    
    print("正在下载 FastSpeech2 模型...")
    manager = ModelManager()
    
    # 下载 FastSpeech2
    model_path, config_path, _ = manager.download_model("tts_models/zh-CN/baker/fastspeech2")
    print(f"模型下载完成: {model_path}")
    
    return model_path, config_path

def prepare_data():
    """准备数据"""
    # 读取 CSV
    train_csv = ROOT / "train.csv"
    val_csv = ROOT / "val.csv"
    
    if not train_csv.exists():
        print(f"错误: 找不到 {train_csv}")
        sys.exit(1)
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    print(f"训练样本: {len(train_df)}, 验证样本: {len(val_df)}")
    
    # 生成 Coqui 格式
    meta_train = ROOT / "meta_train.txt"
    meta_val = ROOT / "meta_val.txt"
    
    def to_meta(df, out_path):
        lines = ["audio_file|text|speaker_name"]
        for _, row in df.iterrows():
            ap = Path(row["audio_path"])
            text = str(row["text"]).replace("|", " ")
            lines.append(f"{ap.name}|{text}|female")
        out_path.write_text("\n".join(lines), encoding="utf-8")
    
    to_meta(train_df, meta_train)
    to_meta(val_df, meta_val)
    
    return str(meta_train), str(meta_val)

def train():
    """训练 FastSpeech2"""
    from TTS.config import load_config
    from TTS.tts.models import setup_model
    from trainer import Trainer, TrainerArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.config.shared_configs import BaseDatasetConfig
    
    # 下载或加载模型
    model_path = MODEL_DIR / "model_file.pth"
    config_path = MODEL_DIR / "config.json"
    
    if not model_path.exists():
        download_model()
    
    # 加载配置
    config = load_config(str(config_path))
    
    # 准备数据
    meta_train, meta_val = prepare_data()
    
    # 配置数据集
    config.datasets = [
        BaseDatasetConfig(
            formatter="ljspeech",
            dataset_name="adult_voice",
            path=str(ROOT),
            meta_file_train=Path(meta_train).name,
            meta_file_val=Path(meta_val).name,
        )
    ]
    
    # 关键配置
    config.model = "fast_speech2"
    config.output_path = str(OUTPUT_DIR)
    config.epochs = EPOCHS
    config.batch_size = BATCH_SIZE
    config.save_step = SAVE_STEP
    config.print_step = PRINT_STEP
    
    # 数据过滤（删除过短/过长音频）
    config.min_text_len = 5
    config.max_text_len = 100
    config.min_audio_len = 22050      # 1秒
    config.max_audio_len = 220500     # 10秒
    
    # Duration 约束
    if not hasattr(config, 'duration_predictor'):
        config.duration_predictor = {}
    config.duration_predictor['min_duration'] = 3
    config.duration_predictor['max_duration'] = 50
    
    # 加载数据
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_size=0.1,
    )
    
    print(f"加载样本: train={len(train_samples)}, eval={len(eval_samples)}")
    
    # 初始化模型
    model = setup_model(config, train_samples + eval_samples)
    
    # 加载预训练权重
    if model_path.exists():
        model.load_checkpoint(config, str(model_path), eval=False)
        print(f"已加载预训练: {model_path}")
    
    # 移动到 GPU
    if gpu:
        model = model.cuda()
    
    # 训练参数
    train_args = TrainerArgs(
        restore_path="",
        continue_path="",
        gpu=0 if gpu else None,
    )
    
    # 初始化训练器
    trainer = Trainer(
        train_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    
    # 开始训练
    print(f"开始训练 FastSpeech2...")
    trainer.fit()
    print("训练完成!")

if __name__ == "__main__":
    train()
