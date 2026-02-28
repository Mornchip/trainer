import os
import pandas as pd
from sklearn.model_selection import train_test_split
from TTS.api import TTS
import random

# ----------------------------
# 1. 配置
# ----------------------------
audio_dir = "./audio"
label_dir = "./labels"
pretrained_model = "espnet/kan-bayashi_ljspeech_vits"  # 可换中文女性模型
gpu = True

num_epochs = 20
batch_size = 8

# ----------------------------
# 2. 解析标注文件 -> CSV
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
            rows.append([audio_path, text])

df = pd.DataFrame(rows, columns=["audio_path", "text"])

# 划分训练/验证集
train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

# ----------------------------
# 3. 加载 TTS 预训练模型
# ----------------------------
tts = TTS(model_name=pretrained_model, gpu=gpu)

# ----------------------------
# 4. 辅助函数：分桶
# ----------------------------
def bucket_data(df, max_len):
    return df[df["text"].str.len() <= max_len].reset_index(drop=True)

# ----------------------------
# 5. 分阶段训练
# ----------------------------
buckets = [
    {"name": "short", "max_len": 30},
    {"name": "medium", "max_len": 60},
    {"name": "long", "max_len": 200}  # 超长句
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
        # 打乱数据
        bucket_train = bucket_train.sample(frac=1).reset_index(drop=True)
        
        # batch 训练
        for i in range(0, len(bucket_train), batch_size):
            batch = bucket_train.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                text = row["text"]
                audio_path = row["audio_path"]
                # 微调训练接口
                tts.train_step(text=text, audio_path=audio_path)
        
        # 每个 epoch 保存 checkpoint
        ckpt_path = f"tts_{bucket['name']}_epoch_{epoch+1}.pth"
        tts.save_model(ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

print("训练完成 ✅")