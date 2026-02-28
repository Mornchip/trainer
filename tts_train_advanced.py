import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from TTS.api import TTS
import torch

# ----------------------------
# 配置
# ----------------------------
audio_dir = "./audio"
label_dir = "./labels"
pretrained_model = "espnet/kan-bayashi_ljspeech_vits"  # 可换中文女性模型
gpu = True
num_epochs = 20
batch_size = 8

# ----------------------------
# 1. 解析标注文件 -> DataFrame
# ----------------------------
rows = []

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue
    audio_file = label_file.replace(".txt", ".wav")
    audio_path = os.path.join(audio_dir, audio_file)
    
    with open(os.path.join(label_dir, label_file), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                start, end, speaker, text = line.split(" ", 3)
            except:
                continue
            if len(text) < 2:
                continue
            rows.append([audio_path, text, speaker])

df = pd.DataFrame(rows, columns=["audio_path", "text", "speaker"])

# ----------------------------
# 2. 划分训练/验证集
# ----------------------------
train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

# ----------------------------
# 3. 加载 TTS 模型
# ----------------------------
tts = TTS(model_name=pretrained_model, gpu=gpu)

# ----------------------------
# 4. 辅助函数：长度分桶 + pitch/energy 归一化
# ----------------------------
def bucket_data(df, max_len):
    return df[df["text"].str.len() <= max_len].reset_index(drop=True)

def normalize_f0_energy(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    f0 = librosa.yin(y, fmin=50, fmax=500)
    f0 = np.nan_to_num(f0)
    f0_norm = (f0 - np.min(f0)) / (np.max(f0) - np.min(f0) + 1e-6)
    
    energy = np.abs(y)
    energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-6)
    
    return f0_norm, energy_norm

# ----------------------------
# 5. 分阶段训练
# ----------------------------
buckets = [
    {"name": "short", "max_len": 30},
    {"name": "medium", "max_len": 60},
    {"name": "long", "max_len": 200}
]

for bucket in buckets:
    print(f"\n=== 训练 {bucket['name']} 句子 ===")
    bucket_train = bucket_data(train_df, bucket["max_len"])
    bucket_val = bucket_data(val_df, bucket["max_len"])
    
    if len(bucket_train) == 0:
        print("该阶段无训练数据，跳过")
        continue
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # 打乱训练数据
        bucket_train = bucket_train.sample(frac=1).reset_index(drop=True)
        
        for i in range(0, len(bucket_train), batch_size):
            batch = bucket_train.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                text = row["text"]
                audio_path = row["audio_path"]
                speaker_id = row["speaker"]
                
                # pitch / energy 归一化
                f0_norm, energy_norm = normalize_f0_energy(audio_path)
                
                # 微调接口（示意）
                tts.train_step(text=text, audio_path=audio_path, speaker=speaker_id,
                               f0=f0_norm, energy=energy_norm)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(tts.model.parameters(), max_norm=1.0)
        
        # 保存 checkpoint
        ckpt_path = f"tts_{bucket['name']}_epoch_{epoch+1}.pth"
        tts.save_model(ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

print("训练完成 ✅")