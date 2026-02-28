#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本-音频一致性校验：可仅用启发式（快），或用 ASR 精确判断（慢但准）。

用法：
  # 仅启发式（字数/时长）：不跑 ASR，秒级完成，适合先筛一批
  python check_text_audio_consistency.py --heuristic-only --output-dirty dirty_heuristic.csv

  # 启发式 + 对「可疑样本」跑 ASR 校验（只对字数/时长异常的跑 ASR，省时）
  python check_text_audio_consistency.py --asr-suspicious-only --min-similarity 0.85 --output-dirty dirty_asr.csv

  # 全量 ASR 校验（慢，适合数据量不大或离线跑一次）
  python check_text_audio_consistency.py --run-asr --min-similarity 0.85 --output-dirty dirty.csv --output-clean train_clean.csv

  # 只检查前 500 条（调试）
  python check_text_audio_consistency.py --run-asr --max-samples 500
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    import librosa
except ImportError:
    librosa = None

try:
    import pandas as pd
except ImportError:
    pd = None

# 与 tts_process 保持一致（可再收紧：如 MAX 改为 6～7 更严格）
MIN_CHARS_PER_SEC = 1.5
MAX_CHARS_PER_SEC = 7.0


def _normalize_text(s: str) -> str:
    """去空格、统一空白，便于和 ASR 结果对比。"""
    if not s:
        return ""
    s = re.sub(r"\s+", "", str(s).strip())
    return s


def _similarity(a: str, b: str) -> float:
    """0~1，1 为完全一致。"""
    a, b = _normalize_text(a), _normalize_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def get_duration(audio_path: str) -> float | None:
    if not librosa:
        return None
    try:
        return float(librosa.get_duration(path=audio_path))
    except Exception:
        return None


def heuristic_suspicious(audio_path: str, text: str) -> tuple[bool, str]:
    """仅凭时长和字数判断是否可疑。返回 (是否可疑, 原因)。"""
    dur = get_duration(audio_path)
    if dur is None or dur <= 0:
        return True, "无法获取时长"
    n_chars = len(_normalize_text(text))
    if n_chars == 0:
        return True, "文本为空"
    cps = n_chars / dur
    if cps < MIN_CHARS_PER_SEC:
        return True, f"字数/时长过低({cps:.2f}<{MIN_CHARS_PER_SEC})"
    if cps > MAX_CHARS_PER_SEC:
        return True, f"字数/时长过高({cps:.2f}>{MAX_CHARS_PER_SEC})"
    return False, ""


def load_csv_paths(csv_paths: list[str]) -> list[tuple[str, str]]:
    """返回 [(audio_path, text), ...]。"""
    rows = []
    for path in csv_paths:
        p = Path(path)
        if not p.exists():
            print(f"警告: 文件不存在 {path}", file=sys.stderr)
            continue
        if pd is not None:
            df = pd.read_csv(p, dtype={"text": str, "audio_path": str})
            df = df.dropna(subset=["audio_path", "text"])
            for _, r in df.iterrows():
                rows.append((str(r["audio_path"]).strip(), str(r["text"]).strip()))
        else:
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ap = (r.get("audio_path") or "").strip()
                    tx = (r.get("text") or "").strip()
                    if ap and tx:
                        rows.append((ap, tx))
    return rows


def _parse_funasr_result(result) -> str:
    """解析 FunASR generate() 返回的多种格式。"""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, dict):
                t = item.get("text") or item.get("pred") or item.get("value")
                if t:
                    return (t if isinstance(t, str) else str(t)).strip()
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def run_asr_funasr(audio_path: str) -> str:
    """FunASR 中文识别，返回识别文本。"""
    try:
        from funasr import AutoModel
        if not hasattr(run_asr_funasr, "_model"):
            run_asr_funasr._model = AutoModel(
                model="iic/Paraformer-large-v2",
                device="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
                disable_update=True,
            )
        model = run_asr_funasr._model
        raw = model.generate(input=audio_path, batch_size_s=300)
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            return _parse_funasr_result(raw[0])
        return _parse_funasr_result(raw)
    except Exception as e:
        return f"__ASR_ERROR__:{e!s}"


def run_asr_whisper(audio_path: str) -> str:
    """Whisper / faster_whisper 识别。"""
    try:
        import torch
        # 优先 faster_whisper（更快）
        try:
            from faster_whisper import WhisperModel
            if not hasattr(run_asr_whisper, "_model"):
                run_asr_whisper._model = WhisperModel(
                    "large-v3" if torch.cuda.is_available() else "base",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    compute_type="float16" if torch.cuda.is_available() else "int8",
                )
            model = run_asr_whisper._model
            segments, _ = model.transcribe(audio_path, language="zh")
            return "".join(s.text for s in segments).strip()
        except ImportError:
            pass
        import whisper
        if not hasattr(run_asr_whisper, "_model"):
            run_asr_whisper._model = whisper.load_model("base")
        out = run_asr_whisper._model.transcribe(audio_path, language="zh")
        return (out.get("text") or "").strip()
    except Exception as e:
        return f"__ASR_ERROR__:{e!s}"


def main():
    ap = argparse.ArgumentParser(description="文本-音频一致性：启发式 + 可选 ASR")
    ap.add_argument("csv", nargs="*", default=None, help="train.csv val.csv 等，默认用同目录 train.csv + val.csv")
    ap.add_argument("--heuristic-only", action="store_true", help="只做字数/时长启发式，不跑 ASR")
    ap.add_argument("--asr-suspicious-only", action="store_true", help="只对启发式判为可疑的样本跑 ASR")
    ap.add_argument("--run-asr", action="store_true", help="全量跑 ASR 做一致性校验")
    ap.add_argument("--asr-engine", choices=["funasr", "whisper"], default="funasr", help="ASR 引擎")
    ap.add_argument("--min-similarity", type=float, default=0.80, help="ASR 与标注文本相似度低于此视为脏数据")
    ap.add_argument("--max-samples", type=int, default=None, help="最多检查条数（调试用）")
    ap.add_argument("--output-dirty", type=str, default=None, help="脏数据列表 CSV：audio_path,text,asr_text,similarity,reason")
    ap.add_argument("--output-clean", type=str, default=None, help="清理后的 CSV（仅保留通过校验的样本）")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent
    if args.csv:
        csv_paths = [str(Path(p).resolve()) for p in args.csv]
    else:
        t1, t2 = ROOT / "train.csv", ROOT / "val.csv"
        csv_paths = [str(t1), str(t2)]
        csv_paths = [p for p in csv_paths if Path(p).exists()]
    if not csv_paths:
        print("未找到 CSV 文件", file=sys.stderr)
        sys.exit(1)

    samples = load_csv_paths(csv_paths)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"共加载 {len(samples)} 条样本")

    if not librosa:
        print("未安装 librosa，无法做时长启发式，请 pip install librosa", file=sys.stderr)

    do_asr = args.run_asr or args.asr_suspicious_only
    asr_fn = run_asr_funasr if args.asr_engine == "funasr" else run_asr_whisper

    dirty_rows = []
    clean_rows = []

    for i, (audio_path, text) in enumerate(samples):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  进度 {i+1}/{len(samples)} ...")

        is_suspicious_heuristic, reason = heuristic_suspicious(audio_path, text)
        run_asr_this = do_asr and (args.run_asr or (args.asr_suspicious_only and is_suspicious_heuristic))

        asr_text = ""
        sim = -1.0  # -1 表示未做 ASR
        if run_asr_this:
            asr_text = asr_fn(audio_path)
            if asr_text.startswith("__ASR_ERROR__"):
                reason = asr_text
                sim = 0.0
            else:
                sim = _similarity(text, asr_text)
                if sim < args.min_similarity:
                    reason = f"ASR与标注相似度低({sim:.3f}<{args.min_similarity})"
                elif is_suspicious_heuristic:
                    reason = "启发式可疑但ASR通过"
        else:
            if is_suspicious_heuristic:
                reason = reason or "启发式可疑"

        if is_suspicious_heuristic and not run_asr_this:
            dirty_rows.append((audio_path, text, asr_text or "", sim, reason))
            continue
        if run_asr_this and sim >= 0 and sim < args.min_similarity:
            dirty_rows.append((audio_path, text, asr_text, sim, reason))
            continue

        clean_rows.append((audio_path, text))

    print(f"通过: {len(clean_rows)}，脏/可疑: {len(dirty_rows)}")

    if args.output_dirty and dirty_rows:
        out = Path(args.output_dirty)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["audio_path", "text", "asr_text", "similarity", "reason"])
            for r in dirty_rows:
                w.writerow(r)
        print(f"脏数据已写入 {out}")

    if args.output_clean and clean_rows:
        out = Path(args.output_clean)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["audio_path", "text"])
            w.writerows(clean_rows)
        print(f"清理后数据已写入 {out}")


if __name__ == "__main__":
    main()
