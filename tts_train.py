# tts_train.py
import pandas as pd
from TTS.api import TTS

# 加载数据 CSV
train_csv = "train.csv"
val_csv = "val.csv"

train_data = pd.read_csv(train_csv)
val_data = pd.read_csv(val_csv)

# 加载预训练中文女性 TTS v2 模型
tts = TTS(
    model_name="espnet/kan-bayashi_ljspeech_vits",  # 可换成中文女性预训练模型
    gpu=True
)

# 训练参数
num_epochs = 20
batch_size = 8

# 简单微调循环示例
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    # Shuffle train_data
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    for idx in range(0, len(train_data), batch_size):
        batch = train_data[idx:idx+batch_size]
        for _, row in batch.iterrows():
            audio_path = row['audio_path']
            text = row['text']
            
            # v2 框架训练接口（示意）
            tts.train_step(text=text, audio_path=audio_path)
    
    # 每个 epoch 可保存 checkpoint
    tts.save_model(f"tts_epoch_{epoch+1}.pth")