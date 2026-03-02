# tts_infer.py - 使用训练产出的 checkpoint 推理（支持单条与批量呻吟词生成）
from pathlib import Path
from TTS.api import TTS

# 训练产出目录（10kep 当前最佳：mandarin_dca_attn_gst_dcc-February-26-2026_11+03AM）
RUN_DIR = Path(__file__).resolve().parent / "tts_train_output" / "mandarin_dca_attn_gst_dcc-February-26-2026_11+03AM-0000000"
CHECKPOINT = RUN_DIR / "best_model.pth"  # 当前验证集最佳（与 best_model_51523.pth 等价或更新）
CONFIG = RUN_DIR / "config.json"

if not CHECKPOINT.exists() or not CONFIG.exists():
    raise FileNotFoundError(f"未找到模型: {CHECKPOINT} 或 {CONFIG}。请先完成训练或修改 RUN_DIR。")

# 本地模型必须用 model_path + config_path，不能用 model_name
tts = TTS(model_path=str(CHECKPOINT), config_path=str(CONFIG), gpu=True)

# 呻吟类词 / 语气词：用于批量生成 10 个音频
SHENYIN_TEXTS = [
    "啊,好爽啊，来操我啊",
    "啊啊",
    "嗯",
    "嗯嗯",
    "啊，嗯",
    "呵",
    "哦",
    "呀",
    "唔",
    "哈啊",
]
SHENYIN_TEXTS = [
    "啊,好爽啊，来操我啊",
    "啊，啊。。。,好爽啊，我要你干个够，你好厉害啊，你的每一次撞击，让我很爽",
 
   
]

def main():
    out_dir = Path(__file__).resolve().parent / "infer"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(SHENYIN_TEXTS, start=1):
        output_wav = out_dir / f"output_shenyin_{i}.wav"
        tts.tts_to_file(text=text, file_path=str(output_wav))
        print(f"生成 [{i}/10] 文本「{text}」 -> {output_wav}")
    print("10 个呻吟词音频已生成完毕，保存至 openaiclaud/infer/")

if __name__ == "__main__":
    main()
