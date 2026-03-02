#!/usr/bin/env python3
"""
VITS 呻吟语音模型 - 推理脚本
使用训练产出的最佳模型生成音频，不做 ASR 验证。
用法: python vits_moaning_infer.py           # 生成 TEXTS 中全部到 infer_moaning/output_01.wav ...
       python vits_moaning_infer.py --text "啊，嗯"  # 只生成一条到 infer_moaning/ah_en.wav
"""
import argparse
import re
import numpy as np
import torch
from pathlib import Path
from TTS.config import load_config
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.synthesis import synthesis

# 训练 run 目录（含 config.json 与 best_model_*.pth）
# train_codebuddy_20260228193028.log 最佳：best_model_34720.pth（07+30PM run，loss_mel ~19.5）
RUN_DIR = Path(__file__).resolve().parent / "vits_moaning_output" / "vits_moaning_voice-February-28-2026_03+49PM-1c93a8e" / "vits_moaning_voice-February-28-2026_07+15PM-1c93a8e" / "vits_moaning_voice-February-28-2026_07+20PM-0000000" / "vits_moaning_voice-February-28-2026_07+26PM-1c93a8e" / "vits_moaning_voice-February-28-2026_07+30PM-1c93a8e"
CONFIG = RUN_DIR / "config.json"


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

# 10 条：各式各样文本，含几句呻吟声（仅用训练字表中出现的字符，如无 ~ 则不用）
TEXTS = [
    "啊",                 # 1 呻吟
    "嗯",                 # 2 呻吟（与训练字表一致，未用 ~）
    "啊，嗯，不要",       # 3 呻吟
    "哦，好爽啊",         # 4 呻吟
    "嗯嗯，再快一点",     # 5 呻吟
    "你好，在吗？",       # 6 正常对话
    "今天天气不错。",     # 7 正常对话
    "我马上过来。",       # 8 正常对话
    "等一下，我看看。",   # 9 正常对话
    "好的，知道了。",     # 10 正常对话
]


def main():
    parser = argparse.ArgumentParser(description="VITS 呻吟语音推理")
    parser.add_argument("--text", type=str, default="", help="只生成这一条文本")
    parser.add_argument("--output", type=str, default="", help="单条时的输出文件名，如 ah.wav（保存在 infer_moaning/ 下）")
    parser.add_argument("--length_scale", type=float, default=1.0, help="语速/时长因子，>1 更慢更饱满，单字建议 1.5~2.0，默认 1.0")
    parser.add_argument("--noise_scale", type=float, default=0.5, help="推理时潜在空间采样噪声，越小越干净、越大越易杂音，默认 0.5（原 0.667）")
    parser.add_argument("--noise_scale_dp", type=float, default=0.8, help="时长预测器噪声，默认 0.8，可略降以稳定节奏")
    parser.add_argument("--normalize", type=float, default=0.9, help="输出响度归一化，0 不归一化，0.9 表示峰值缩放到 0.9（推荐，减轻轰鸣/失真）")
    parser.add_argument("--check-chars", action="store_true", help="仅检测 TEXTS 中是否有字符被丢弃（Character not found），不生成音频")
    args = parser.parse_args()

    if not CONFIG.exists():
        raise FileNotFoundError(f"未找到 {CONFIG}")
    try:
        CHECKPOINT = _resolve_best_checkpoint(RUN_DIR)
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    print(f"使用最优模型: {CHECKPOINT.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(CONFIG)
    # 推理与训练共用同一 config → 字母表/词表（characters、num_chars）一致
    use_ph = getattr(config, "use_phonemes", False)
    print(f"与训练一致: config 来自 {RUN_DIR.name}，use_phonemes={use_ph}，字表来自 config.characters")

    # 与训练一致：用该 run 的 config 初始化（字母表/num_chars 与训练时一致）
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

    # 若 checkpoint 的 embedding 维数大于当前（如旧 run 多一字），则扩一维再加载
    if model.args.num_chars < required_num_chars:
        emb = model.text_encoder.emb
        new_emb = torch.nn.Embedding(
            required_num_chars,
            emb.embedding_dim,
            padding_idx=emb.padding_idx if hasattr(emb, "padding_idx") else None,
        )
        with torch.no_grad():
            new_emb.weight[: model.args.num_chars].copy_(emb.weight)
        model.text_encoder.emb = new_emb
        model.args.num_chars = required_num_chars

    # 加载 checkpoint（兼容带 _orig_mod. 前缀的 state_dict，如 torch.compile 保存的）
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else getattr(ckpt, "state_dict", lambda: ckpt)()
    if isinstance(state_dict, dict) and any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    # 推理噪声：TTS 默认 inference_noise_scale=0.667，偏大易导致杂音；降低可明显改善清晰度（loss_mel 只约束频谱，波形由 flow+decoder 采样）
    model.inference_noise_scale = float(args.noise_scale)
    model.inference_noise_scale_dp = float(args.noise_scale_dp)
    if args.noise_scale != 0.667 or args.noise_scale_dp != 1.0:
        print(f"推理噪声: noise_scale={model.inference_noise_scale}, noise_scale_dp={model.inference_noise_scale_dp}（降低可减杂音）")
    model.length_scale = float(args.length_scale)
    if args.length_scale != 1.0:
        print(f"使用 length_scale={model.length_scale}（拉长单字/短句，减轻断断续续）")

    # 【关键】中文 phonemizer 的 PINYIN_DICT 里部分音节用 ASCII（如 "guo"->"guo", "ga"->"ga"），
    # 词表是 IPA（ɡ U+0261 等），encode 时 ASCII 被丢弃导致序列错误、轰鸣。此处统一 ASCII→IPA。
    # 根据 TTS 中文 pinyinToPhonemes 中可能出现的 ASCII 与 IPA 对照补全（可随 --check-chars 发现新缺失再补）。
    _ASCII_TO_IPA = str.maketrans({
        "g": "ɡ",   # ɡ U+0261  voiced velar stop，PINYIN_DICT 里 ga/gu/guo 等用 ASCII g
        "r": "ɹ",   # ɹ U+0279  alveolar approximant，"er" 等可能出 ASCII r
    })
    if getattr(config, "use_phonemes", False) and getattr(model.tokenizer, "phonemizer", None):
        _orig_phonemize = model.tokenizer.phonemizer.phonemize
        def _patched_phonemize(text, separator="", language=None):
            out = _orig_phonemize(text, separator=separator, language=language)
            return out.translate(_ASCII_TO_IPA) if isinstance(out, str) else out
        model.tokenizer.phonemizer.phonemize = _patched_phonemize
        print("已启用音素输出 ASCII→IPA 校正（g→ɡ, r→ɹ）")

    if args.check_chars:
        # 仅检测：对 TEXTS 跑一遍 text_to_ids，收集并打印被丢弃的字符
        model.tokenizer.not_found_characters = []
        for t in TEXTS:
            model.tokenizer.text_to_ids(t)
        missing = getattr(model.tokenizer, "not_found_characters", [])
        if missing:
            print(f"⚠️ 以下字符不在词表中（会被丢弃）: {sorted(set(missing))}")
            print("   请确保输入仅含训练数据中出现过的字符（use_phonemes 时可在 _ASCII_TO_IPA 中补 IPA 映射）。")
        else:
            print("✓ 未发现被丢弃字符，与训练字母表一致。")
        return

    sample_rate = config.audio.sample_rate
    out_dir = Path(__file__).resolve().parent / "vits_infer_moaning_voice"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.text.strip():
        out_name = (args.output.strip() or "ah_en.wav").replace(".wav", "") + ".wav"
        texts_to_gen = [(args.text.strip(), out_dir / out_name)]
    else:
        texts_to_gen = [(t, out_dir / f"output_{i:02d}.wav") for i, t in enumerate(TEXTS, start=1)]

    # 使用 TTS 官方的 synthesis()：与训练一致的 text→音素→ids→VITS（含内置 waveform_decoder）→波形，避免轰鸣/机械声
    use_cuda = device.type == "cuda"
    for text, out_wav in texts_to_gen:
        try:
            result = synthesis(
                model, text, config, use_cuda,
                speaker_id=None, do_trim_silence=False, use_griffin_lim=False,
            )
            wav_np = result["wav"]
            if wav_np is None or (isinstance(wav_np, np.ndarray) and wav_np.size == 0):
                print(f"跳过 空输出 「{text}」")
                continue
            if isinstance(wav_np, np.ndarray):
                wav_np = wav_np.astype(np.float64)
            else:
                wav_np = np.asarray(wav_np, dtype=np.float64)
            # 裁剪到 [-1, 1] 并可选峰值归一化，避免数值溢出导致轰鸣/失真
            wav_np = np.clip(wav_np, -1.0, 1.0)
            if args.normalize > 0:
                peak = np.abs(wav_np).max()
                if peak > 1e-6:
                    wav_np = wav_np * (float(args.normalize) / peak)
            save_wav(wav=wav_np, path=str(out_wav), sample_rate=sample_rate)
            print(f"生成 「{text}」 -> {out_wav}")
        except Exception as e:
            print(f"生成失败 「{text}」: {e}")

    print(f"音频已保存: {out_dir}")


if __name__ == "__main__":
    main()
