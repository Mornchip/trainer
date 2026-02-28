# tts_process.py：从 womenvoice/dataset 读取标注，生成片段 wav 与 train/val CSV
import os
import sys
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 使用 womenvoice 项目下的数据
ROOT = Path(__file__).resolve().parent
WOMENVOICE = ROOT.parent
LABEL_DIR = WOMENVOICE / "dataset" / "labels"
DOWNLOAD_DIR = WOMENVOICE / "dataset" / "downloads"
SEGMENTS_DIR = ROOT / "segments"

# 使用全部数据：设为 None 不限制条数；设为整数则最多处理该条数（调试用）
MAX_SAMPLES = None
MAX_DURATION_SEC = 10.0
MIN_DURATION_SEC = 0.2

# 文本-音频一致性启发式：字数/时长（字/秒）。中文正常语速约 3～6 字/秒，超出范围多为标注错位或脏数据
MIN_CHARS_PER_SEC = 1.5   # 低于此值：音频过长/文本过短，可能静音或错段
MAX_CHARS_PER_SEC = 7.0  # 高于此值：音频过短/文本过长，可能截断或错标

# 训练用统一采样率（与预训练模型一致，如 LJSpeech/VITS 常用 22050）
TARGET_SAMPLE_RATE = 22050

# 解析失败的行写入该文件，便于检查
FAILED_PARSE_PATH = ROOT / "tts_process_failed.txt"

SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _find_audio(stem: str) -> Path | None:
    """根据标注文件名 stem 在 downloads 下找音频。"""
    for ext in (".wav", ".mp3", ".flac", ".m4a"):
        p = DOWNLOAD_DIR / (stem + ext)
        if p.exists():
            return p
    if stem and stem[0].isdigit():
        i = 0
        while i < len(stem) and stem[i].isdigit():
            i += 1
        if i < len(stem):
            with_space = stem[:i] + " " + stem[i:]
            for ext in (".wav", ".mp3", ".flac", ".m4a"):
                p = DOWNLOAD_DIR / (with_space + ext)
                if p.exists():
                    return p
    return None


def _parse_line(line: str):
    """返回 (basename, start_sec, end_sec, text) 或 None。"""
    line = line.strip()
    base, start, end, speaker, text=None,None,None,None,None
    if not line:
        return None
    parts = line.split(None, 4)
    lenparts=len(parts)
    separator = "--小烟"
    file_name=None  

    # 找到分隔标识的结束位置，提取目标文件名
    if separator in line:
        # 计算分隔标识结束的索引位置
        end_index = line.find(separator) + len(separator)
        # 截取到该位置的字符串就是目标文件名
        file_name = line[:end_index].strip()
        print("提取的文件名：", file_name)
        line=line.replace(file_name, file_name +" ")
        parts = line.split(None, 4)

    else:
        print("未找到分隔标识'--小烟'")
    print("lenparts",lenparts)
    if len(parts) < 5:
        return None
    try:
        match(len(parts)):
            case 5: # 完整格式
                base, start, end, speaker, text = parts[0], float(parts[1].replace(",", "")), float(parts[2].replace(",", "")), parts[3], parts[4].strip()
            case 4:
                base, start, end, text = parts[0], float(parts[1].replace(",","")), float(parts[2].replace(",","")), parts[3].strip()
                speaker = 0
            case _:
                return None
    except (ValueError, IndexError):
        return None
    
    if not text or end <= start or end - start < 0.01:
        return None
    return (base, start, end,speaker, text)


def _filter_text_df(df: pd.DataFrame, min_chars: int = 3, max_chars: int = 120) -> pd.DataFrame:
    """按文本内容进一步过滤样本，去掉极短、全是符号或超长的句子。"""

    def is_valid(text: str) -> bool:
        if not isinstance(text, str):
            return False
        t = text.strip()
        if len(t) < min_chars:
            return False
        if len(t) > max_chars:
            return False
        # 至少包含一个「有信息量」的字符：中文 / 英文 / 数字
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", t):
            return False
        return True

    total = len(df)
    mask = df["text"].apply(is_valid)
    filtered = df[mask].reset_index(drop=True)
    removed = total - len(filtered)
    print(
        f"[文本过滤] 总数={total} -> 保留={len(filtered)}，"
        f"被删除={removed}（过短/全符号/超长）"
    )
    return filtered

#main函数

def main():
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        import librosa
        sf = None

    if not LABEL_DIR.exists():
        print(f"标注目录不存在: {LABEL_DIR}")
        sys.exit(1)
    if not DOWNLOAD_DIR.exists():
        print(f"音频目录不存在: {DOWNLOAD_DIR}，请先准备 dataset/downloads")
        sys.exit(1)

    rows = []
    segment_id = 0
    label_files = sorted(LABEL_DIR.glob("*.txt"))
    iok = 0
    ierror = 0
    skipped_cps_low = 0   # 字数/时长 过低（音频长、字少，疑似静音/错段）
    skipped_cps_high = 0  # 字数/时长 过高（音频短、字多，疑似错标）
    with open(FAILED_PARSE_PATH, "w", encoding="utf-8") as failed_f:
        failed_f.write("audio_path\traw_line\n")
        for label_file in label_files:
            if MAX_SAMPLES is not None and segment_id >= MAX_SAMPLES:
                break
            stem = label_file.stem

            audio_path = _find_audio(stem)
            if audio_path is None:
                continue
            try:
                full_audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
            except Exception as e:
                print(f"加载失败 {audio_path}: {e}")
                continue

            with open(label_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if MAX_SAMPLES is not None and segment_id >= MAX_SAMPLES:
                        break
                    parsed = _parse_line(line)
                    if not parsed:
                        failed_f.write(f"{audio_path}\t{line.strip()}\n")
                        failed_f.flush()
                        print(audio_path, "not parsed", line, ierror)
                        ierror += 1
                        continue
                    iok += 1
                    print(audio_path, "ok", line, iok)
                    base, start_sec, end_sec, speaker, text = parsed
                    dur = end_sec - start_sec
                    if dur > MAX_DURATION_SEC or dur < MIN_DURATION_SEC:
                        continue
                    if len(text) < 2:
                        continue
                    if len(text) > 20:
                        continue
                    start_samp = int(start_sec * sr)
                    end_samp = int(end_sec * sr)
                    if start_samp >= len(full_audio) or end_samp <= start_samp:
                        continue
                    segment_wav = full_audio[start_samp:end_samp]
                    if len(segment_wav) < MIN_DURATION_SEC * sr:
                        continue

                    # 文本-音频一致性启发式：字数/时长 不在合理范围则丢弃（不做 ASR，速度快）
                    chars_per_sec = len(text) / dur
                    if chars_per_sec < MIN_CHARS_PER_SEC:
                        skipped_cps_low += 1
                        continue
                    if chars_per_sec > MAX_CHARS_PER_SEC:
                        skipped_cps_high += 1
                        continue

                    # 统一重采样到 TARGET_SAMPLE_RATE，避免 48000/44100 等混用导致训练不稳定
                    if sr != TARGET_SAMPLE_RATE:
                        segment_wav = librosa.resample(
                            segment_wav, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE, res_type="kaiser_best"
                        )
                    save_sr = TARGET_SAMPLE_RATE

                    seg_name = f"segment_{segment_id:05d}"
                    wav_out = SEGMENTS_DIR / f"{seg_name}.wav"
                    if sf is not None:
                        sf.write(str(wav_out), segment_wav, save_sr)
                    else:
                        import scipy.io.wavfile as wavfile
                        wavfile.write(str(wav_out), save_sr, (segment_wav * 32767).clip(-32768, 32767).astype("int16"))

                    rows.append([str(wav_out.resolve()), text])
                    segment_id += 1
                    if segment_id % 100 == 0 and segment_id:
                        print(f"  已处理 {segment_id} 条...")

    if not rows:
        print("没有生成任何有效样本，请检查 labels 与 downloads")
        sys.exit(1)

    if ierror > 0:
        print(f"解析失败 {ierror} 条，已写入 {FAILED_PARSE_PATH} 便于检查。")
    if skipped_cps_low or skipped_cps_high:
        print(
            f"[文本-音频一致性] 因字数/时长异常丢弃: 过低(<{MIN_CHARS_PER_SEC}字/秒)={skipped_cps_low}，"
            f"过高(>{MAX_CHARS_PER_SEC}字/秒)={skipped_cps_high}"
        )

    df = pd.DataFrame(rows, columns=["audio_path", "text"])
    # 额外按文本内容做一次过滤，避免大量极短呻吟、纯符号句影响对齐
    df = _filter_text_df(df)
    train_df, val_df = train_test_split(df, test_size=0.075, random_state=42)
    train_df.to_csv(ROOT / "train.csv", index=False)
    val_df.to_csv(ROOT / "val.csv", index=False)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    print("train.csv / val.csv 已写入 openaiclaud 目录。")


if __name__ == "__main__":
    main()
