"""使用训练产出的 checkpoint 做 TTS 推理（Tacotron2 + GST）。"""
from pathlib import Path
from TTS.api import TTS

# 训练产出目录（本次 run）
RUN_DIR = Path(__file__).resolve().parent / "tts_train_output" / "mandarin_dca_attn_gst_dcc-February-25-2026_02+04PM-1c93a8e"
CHECKPOINT = RUN_DIR / "checkpoint_30.pth"
CONFIG = RUN_DIR / "config.json"

if not CHECKPOINT.exists() or not CONFIG.exists():
    raise FileNotFoundError(f"未找到模型: {CHECKPOINT} 或 {CONFIG}")

# 加载微调后的模型（需同时指定 config）
tts = TTS(model_path=str(CHECKPOINT), config_path=str(CONFIG), gpu=True)

# 输入文本
text = "这个家伙。难道就是最近才从魔王城回来？"

# 单说话人模型，直接生成
output_wav = Path(__file__).resolve().parent / "output.wav"
tts.tts_to_file(text=text, file_path=str(output_wav))
print(f"生成语音保存至 {output_wav}")
