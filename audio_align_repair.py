#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频对齐修复模块：ASR 检查与重标注、静音切除、批量处理、Excel 报告。
输出修复后的 JSON/CSV 可直接用于 VITS 训练。

依赖: pip install openai-whisper pydub pandas openpyxl
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# 可选依赖
try:
    import whisper
except ImportError:
    whisper = None
try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
except ImportError:
    AudioSegment = None
    detect_nonsilent = None


# ---------- 重合度与规则 ----------
def normalize_text(s: str) -> str:
    """统一空白与标点便于比较。"""
    if not s or not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", "", s.strip())
    return s


def char_overlap_ratio(orig: str, asr: str) -> float:
    """计算原标注与 ASR 的字符重合度，返回 0~1。"""
    a, b = normalize_text(orig), normalize_text(asr)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def asr_action(ratio: float) -> str:
    """根据重合度返回执行动作。"""
    if ratio > 0.90:
        return "保留原标注"
    if 0.60 <= ratio <= 0.90:
        return "ASR替换"
    if ratio < 0.40:
        return "需人工审核"
    return "保留原标注"  # 40% <= ratio <= 90% 时保守保留


# ---------- ASR ----------
def run_whisper_asr(audio_path: str | Path, model=None, language: str = "zh") -> str:
    """对单条音频做中文 ASR，返回识别文本。"""
    if whisper is None:
        raise RuntimeError("请安装 openai-whisper: pip install openai-whisper")
    path = Path(audio_path)
    if not path.exists():
        return ""
    result = model.transcribe(str(path), language=language, fp16=False)
    text = (result.get("text") or "").strip()
    return text


# ---------- 静音切除 ----------
SILENCE_THRESH_MS = 300  # 头尾静音超过 300ms 则切除
MIN_SILENCE_LEN_MS = 300
SILENCE_THRESH_DBFS = -40  # 低于此视为静音
SEC_PER_CHAR_SLOW = 1.0   # 语速过慢：>1 秒/字


def is_slow_speech(duration_sec: float, text: str) -> bool:
    """是否语速过慢（>1 秒/字）。"""
    n = len(normalize_text(text))
    if n <= 0:
        return False
    return duration_sec / n > SEC_PER_CHAR_SLOW


def trim_silence_head_tail(
    audio_path: str | Path,
    out_path: str | Path,
    head_tail_silence_ms: int = SILENCE_THRESH_MS,
    min_silence_len_ms: int = MIN_SILENCE_LEN_MS,
    thresh_dbfs: int = SILENCE_THRESH_DBFS,
) -> bool:
    """
    仅切除头部和尾部超过 head_tail_silence_ms 的静音，保留中间有声内容不切割。
    若无需修剪或失败则复制原文件并返回 False；否则写出到 out_path 并返回 True。
    """
    if AudioSegment is None or detect_nonsilent is None:
        return False
    path = Path(audio_path)
    out_path = Path(out_path)
    if not path.exists():
        return False
    try:
        seg = AudioSegment.from_file(str(path))
        duration_ms = len(seg)
        if duration_ms < 2 * head_tail_silence_ms:
            shutil.copy2(path, out_path)
            return False
        nonsilent = detect_nonsilent(
            seg,
            min_silence_len=min_silence_len_ms,
            silence_thresh=thresh_dbfs,
            seek_step=10,
        )
        if not nonsilent:
            shutil.copy2(path, out_path)
            return False
        start_ms, end_ms = nonsilent[0][0], nonsilent[-1][1]
        trim_start = 0
        trim_end = duration_ms
        if start_ms >= head_tail_silence_ms:
            trim_start = start_ms
        if duration_ms - end_ms >= head_tail_silence_ms:
            trim_end = end_ms
        if trim_start <= 0 and trim_end >= duration_ms:
            shutil.copy2(path, out_path)
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seg[trim_start:trim_end].export(str(out_path), format=path.suffix.lstrip(".") or "wav")
        return True
    except Exception:
        try:
            shutil.copy2(path, out_path)
        except Exception:
            pass
        return False


# ---------- 批量处理 ----------
def load_input_json(path: str | Path) -> list[dict]:
    """读取含 audio_path 与 text 的 JSON（数组或带 list 键的对象）。"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "samples", "list"):
            if isinstance(data.get(key), list):
                return data[key]
        if "audio_path" in data and "text" in data:
            return [data]
    return []


def run_repair(
    input_json: str | Path,
    output_dir: str | Path,
    whisper_model_size: str = "base",
    language: str = "zh",
    trim_silence: bool = True,
    copy_audio: bool = True,
    report_excel: str | Path | None = None,
) -> dict:
    """
    批量处理：ASR 检查与重标注、可选静音切除，输出修复 JSON 与 Excel 报告。

    - 原始文件保持不变；修复后音频与 JSON 写入 output_dir。
    - report_excel: 报告路径，默认 output_dir/align_repair_report.xlsx
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_input_json(input_json)
    if not samples:
        return {"error": "无有效数据", "kept": 0, "replaced": 0, "review": 0}

    # 解析基础目录：优先 JSON 所在目录，便于 audio_path 为相对路径
    base_dir = input_json.parent

    def resolve_audio(p: str):
        path = Path(p)
        if path.is_absolute() and path.exists():
            return path
        for root in (base_dir, Path.cwd()):
            candidate = root / p
            if candidate.exists():
                return candidate
        return base_dir / p

    if whisper is None:
        raise RuntimeError("请安装 openai-whisper: pip install openai-whisper")
    print("加载 Whisper 模型...")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    model = whisper.load_model(whisper_model_size, device=device)

    rows = []
    repaired = []
    stats = {"kept": 0, "replaced": 0, "review": 0}

    for i, item in enumerate(samples):
        audio_path = item.get("audio_path") or item.get("audio") or item.get("path") or ""
        orig_text = (item.get("text") or item.get("caption") or "").strip()
        if not audio_path:
            rows.append({
                "文件名": "",
                "原文本": orig_text,
                "ASR结果": "",
                "重合度": 0.0,
                "执行动作": "跳过(无音频路径)",
            })
            continue

        src_audio = resolve_audio(audio_path)
        if not src_audio.exists():
            rows.append({
                "文件名": audio_path,
                "原文本": orig_text,
                "ASR结果": "",
                "重合度": 0.0,
                "执行动作": "跳过(文件不存在)",
            })
            continue

        # ASR
        try:
            asr_text = run_whisper_asr(src_audio, model=model, language=language)
        except Exception as e:
            asr_text = ""
            rows.append({
                "文件名": Path(audio_path).name,
                "原文本": orig_text,
                "ASR结果": f"[ASR异常: {e}]",
                "重合度": 0.0,
                "执行动作": "需人工审核",
            })
            stats["review"] += 1
            continue

        ratio = char_overlap_ratio(orig_text, asr_text)
        action = asr_action(ratio)

        if action == "保留原标注":
            final_text = orig_text
            stats["kept"] += 1
        elif action == "ASR替换":
            final_text = asr_text
            stats["replaced"] += 1
        else:
            final_text = orig_text
            stats["review"] += 1

        # 静音切除：仅对语速过慢样本
        try:
            if AudioSegment:
                seg = AudioSegment.from_file(str(src_audio))
                duration_sec = len(seg) / 1000.0
            else:
                duration_sec = 0.0
        except Exception:
            duration_sec = 0.0

        out_filename = Path(audio_path).name
        out_audio_path = audio_out_dir / out_filename
        if copy_audio:
            if trim_silence and is_slow_speech(duration_sec, final_text):
                trim_silence_head_tail(src_audio, out_audio_path)
            else:
                shutil.copy2(src_audio, out_audio_path)
        # 写入 repaired 时使用相对 output_dir 的路径，便于 VITS 直接读
        rel_audio = str(Path("audio") / out_filename)
        repaired.append({
            "audio_path": rel_audio,
            "text": final_text,
        })
        rows.append({
            "文件名": out_filename,
            "原文本": orig_text,
            "ASR结果": asr_text,
            "重合度": round(ratio * 100, 2),
            "执行动作": action,
        })

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{len(samples)} 条...")

    # 写出修复后的 JSON（仅含保留/替换条目，可直接用于 VITS 训练；待审核条目仅在报告中）
    out_json = output_dir / "repaired_data.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(repaired, f, ensure_ascii=False, indent=2)

    # train.csv / val.csv：按 95/5 划分，与现有训练流程无缝对接
    df_out = pd.DataFrame(repaired)
    n = len(df_out)
    n_val = max(1, min(n // 20, n - 1)) if n > 1 else 0
    n_train = n - n_val
    train_df = df_out.iloc[:n_train]
    val_df = df_out.iloc[n_train:] if n_val else pd.DataFrame(columns=df_out.columns)
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")

    # Excel 报告
    report_path = Path(report_excel) if report_excel else output_dir / "align_repair_report.xlsx"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df_report = pd.DataFrame(rows)
    df_report.to_excel(report_path, index=False, engine="openpyxl")

    summary = {
        "kept": stats["kept"],
        "replaced": stats["replaced"],
        "review": stats["review"],
        "output_dir": str(output_dir),
        "repaired_json": str(out_json),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "report_excel": str(report_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="音频对齐修复：ASR 重标注 + 静音切除，输出 VITS 可用数据")
    parser.add_argument("input_json", type=str, help="输入 JSON（含 audio_path、text）")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="输出目录（原始文件不变）")
    parser.add_argument("--whisper", type=str, default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper 模型")
    parser.add_argument("--language", type=str, default="zh", help="ASR 语言")
    parser.add_argument("--no_trim", action="store_true", help="不做静音切除")
    parser.add_argument("--report", type=str, default="", help="Excel 报告路径（默认 output_dir/align_repair_report.xlsx）")
    args = parser.parse_args()

    report_path = args.report or None
    summary = run_repair(
        args.input_json,
        args.output_dir,
        whisper_model_size=args.whisper,
        language=args.language,
        trim_silence=not args.no_trim,
        report_excel=report_path,
    )
    if "error" in summary:
        print("错误:", summary["error"])
        return
    print("修复完成:")
    print("  保留条数:", summary["kept"])
    print("  替换条数:", summary["replaced"])
    print("  待审核条数:", summary["review"])
    print("  修复 JSON:", summary["repaired_json"])
    print("  train.csv:", summary["train_csv"])
    print("  val.csv:", summary["val_csv"])
    print("  报告 Excel:", summary["report_excel"])


if __name__ == "__main__":
    main()
