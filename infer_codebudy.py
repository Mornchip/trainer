"""
VITS 最佳模型推理脚本
使用 best_model（或步数最大的 best_model_*.pth）生成 10 个测试音频。
加载时兼容 checkpoint 与 config 的 num_chars 不一致（1282 vs 1283）。
"""
import os
import sys
from pathlib import Path
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRAINER_TELEMETRY"] = "0"

import numpy as np
import torch
from TTS.config import load_config
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.synthesis import synthesis

# ==================== 配置路径 ====================
BASE_DIR = Path("/root/autodl-tmp/zlynew/womenvoice/lasttraincodebuddy")
MODEL_DIR = BASE_DIR / "vits_moaning_output/vits_moaning_voice-February-28-2026_03+49PM-1c93a8e/vits_moaning_voice-February-28-2026_07+15PM-1c93a8e/vits_moaning_voice-February-28-2026_07+20PM-0000000/vits_moaning_voice-February-28-2026_07+26PM-1c93a8e/vits_moaning_voice-February-28-2026_07+30PM-1c93a8e"
CONFIG_PATH = MODEL_DIR / "config.json"
OUTPUT_DIR = BASE_DIR / "inference_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 测试文本（10条）====================
TEST_SENTENCES = [
    "你好，这是一个语音合成测试。",
    "今天的温度是二十五度，明天可能会下雨。",
    "太棒了！我们成功了，这真是一个令人激动的时刻。",
    "人工智能技术的发展正在改变我们的生活方式，从智能手机到自动驾驶汽车，科技让一切变得更加便捷。",
    "你觉得这个声音听起来自然吗？",
    "多么美丽的风景啊，真让人流连忘返！",
    "快过来。",
    "虽然他平时很忙，但是周末的时候总会抽出时间陪伴家人，这让大家都感到很温暖。",
    "早上好，吃过早餐了吗？今天的工作安排是什么？",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
]


def _resolve_best_checkpoint(run_dir: Path) -> Path:
    """优先 best_model.pth，否则取步数最大的 best_model_*.pth。"""
    best = run_dir / "best_model.pth"
    if best.exists():
        return best
    import re
    candidates = list(run_dir.glob("best_model_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"未找到 best_model.pth 或 best_model_*.pth in {run_dir}")
    def step(p):
        m = re.search(r"best_model_(\d+)\.pth", p.name)
        return int(m.group(1)) if m else 0
    return max(candidates, key=step)


def check_model_exists():
    if not MODEL_DIR.is_dir():
        print(f"❌ 错误：模型目录不存在: {MODEL_DIR}")
        sys.exit(1)
    if not CONFIG_PATH.exists():
        print(f"❌ 错误：配置文件不存在: {CONFIG_PATH}")
        sys.exit(1)
    try:
        ckpt = _resolve_best_checkpoint(MODEL_DIR)
    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")
        sys.exit(1)
    print(f"✅ 模型目录: {MODEL_DIR}")
    print(f"✅ 配置文件: {CONFIG_PATH}")
    print(f"✅ 使用权重: {ckpt.name}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("-" * 60)


def load_model():
    """加载 TTS 模型（与 vits_moaning_infer 一致，兼容 num_chars 1282/1283）。"""
    print("🚀 正在加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用设备: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    checkpoint_path = _resolve_best_checkpoint(MODEL_DIR)
    config = load_config(CONFIG_PATH)
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    n_chars = tokenizer.characters.num_chars
    required_num_chars = n_chars + 1 if tokenizer.add_blank else n_chars
    config.model_args.num_chars = required_num_chars

    _saved_characters = getattr(config, "characters", None)
    if _saved_characters is not None and required_num_chars != n_chars:
        config.characters = None
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    if _saved_characters is not None:
        config.characters = _saved_characters

    # checkpoint 可能是 1283（add_blank），当前 model 可能是 1282，需扩展 embedding 再 load
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else getattr(ckpt, "state_dict", lambda: ckpt)()
    if isinstance(state_dict, dict) and any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    ckpt_num_chars = state_dict["text_encoder.emb.weight"].shape[0]
    if model.args.num_chars < ckpt_num_chars:
        emb = model.text_encoder.emb
        new_emb = torch.nn.Embedding(
            ckpt_num_chars,
            emb.embedding_dim,
            padding_idx=emb.padding_idx if hasattr(emb, "padding_idx") else None,
        )
        with torch.no_grad():
            new_emb.weight[: model.args.num_chars].copy_(emb.weight)
        model.text_encoder.emb = new_emb
        model.args.num_chars = ckpt_num_chars

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    model.inference_noise_scale = 0.5
    model.inference_noise_scale_dp = 0.8
    model.length_scale = 1.0
    print("✅ 模型加载完成！\n")
    return model, config, device


def generate_audio(model, config, device, text, output_file, length_scale=1.0):
    """生成单个音频（与 vits_moaning_infer 一致）。"""
    try:
        use_cuda = device.type == "cuda"
        model.length_scale = length_scale
        result = synthesis(
            model, text, config, use_cuda,
            speaker_id=None, do_trim_silence=False, use_griffin_lim=False,
        )
        wav_np = result["wav"]
        if wav_np is None or (isinstance(wav_np, np.ndarray) and wav_np.size == 0):
            return False
        wav_np = np.asarray(wav_np, dtype=np.float64)
        wav_np = np.clip(wav_np, -1.0, 1.0)
        peak = np.abs(wav_np).max()
        if peak > 1e-6:
            wav_np = wav_np * (0.9 / peak)
        save_wav(wav=wav_np, path=str(output_file), sample_rate=config.audio.sample_rate)
        return True
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return False


def main():
    print("=" * 60)
    print("🎙️  VITS 语音合成推理脚本")
    print(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    check_model_exists()
    model, config, device = load_model()
    print(f"📝 将生成 {len(TEST_SENTENCES)} 个音频文件\n")
    success_count = 0
    for idx, text in enumerate(TEST_SENTENCES, 1):
        text_short = text[:8].replace("，", "").replace("。", "").replace("？", "")
        filename = f"{idx:02d}_{text_short}.wav"
        output_file = OUTPUT_DIR / filename
        speed_configs = [1.0, 1.2, 0.8]
        speed = speed_configs[(idx - 1) % 3]
        speed_name = ["标准", "慢速", "快速"][(idx - 1) % 3]
        print(f"[{idx:02d}/{len(TEST_SENTENCES)}] 生成中...")
        print(f"    文本: {text[:35]}{'...' if len(text) > 35 else ''}")
        print(f"    语速: {speed_name} ({speed}x)")
        if generate_audio(model, config, device, text, output_file, speed):
            file_size = output_file.stat().st_size / 1024
            print(f"    ✅ 完成: {filename} ({file_size:.1f} KB)\n")
            success_count += 1
        else:
            print(f"    ❌ 失败\n")
    print("=" * 60)
    print(f"✅ 成功生成: {success_count}/{len(TEST_SENTENCES)} 个音频")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
