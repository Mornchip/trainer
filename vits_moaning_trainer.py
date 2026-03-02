#!/usr/bin/env python3
"""
VITS 成人语音训练脚本 - 完整版
包含数据清理 + VITS 训练，针对呻吟音频优化
"""
import os
import sys
import re
import csv
import shutil
import random
import wave
import contextlib
import warnings
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 减少终端刷屏：librosa/torch 弃用警告、pkg_resources（训练日志更干净）
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
warnings.filterwarnings("ignore", message=".*stft with return_complex=False.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*deprecated.*")
warnings.filterwarnings("ignore", message=".*GradScaler.*deprecated.*")

# ==================== 5090 配置 ====================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "25"
os.environ["MKL_NUM_THREADS"] = "25"
os.environ["NUMEXPR_NUM_THREADS"] = "25"
# 禁用 Coqui 训练统计上报，避免连 coqui.gateway.scarf.sh 时的代理/网络错误
os.environ["TRAINER_TELEMETRY"] = "0"
# 避免 trainer 探测 git 时刷屏 "fatal: not a git repository"
os.environ["GIT_TERMINAL_PROMPT"] = "0"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "vits_moaning_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# 音频文件必须放在当前目录 lasttraincodebuddy/segments 下（DataLoader 通过 wavs -> segments 读取）
AUDIO_DIR = "segments"
# 成人 TTS 数据源：标注 womenvoice/dataset/labels，音频 womenvoice/dataset/downloads
DATASET_LABELS = ROOT.parent / "dataset" / "labels"
DATASET_DOWNLOADS = ROOT.parent / "dataset" / "downloads"

# 检查 GPU
gpu = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if gpu else "CPU"
print(f"🚀 设备: {device_name}")
if gpu:
    print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ==================== 保守训练参数（避免 loss 爆炸）====================
# 训练轮数、参与训练的最大条数（0=不限制）。可用环境变量 VITS_EPOCHS、VITS_MAX_SAMPLES 覆盖
EPOCHS = int(os.environ.get("VITS_EPOCHS", "20000"))  # 默认 20000 轮，学习率按 T_max 缓慢退火，不急于降 LR
# 数据量配置：可通过环境变量覆盖，或使用快速测试模式
# 默认配置：660条（600训练 + 60验证）
QUICK_TEST_MODE = os.environ.get("VITS_QUICK_TEST", "").strip() == "1"
if QUICK_TEST_MODE:
    MAX_TRAIN_SAMPLES = 110   # 100训练 + 10验证（快速测试）
    TRAIN_SAMPLES_TARGET = 100
    VAL_SAMPLES_TARGET = 10
    print("⚡ 快速测试模式: 使用100条数据先让生成器学会基本重建")
else:
    # 默认配置：2875条数据（2500训练 + 375验证）
    MAX_TRAIN_SAMPLES = int(os.environ.get("VITS_MAX_SAMPLES", "2875"))  # 总共取2875条
    TRAIN_SAMPLES_TARGET = 2500  # 训练集条数
    VAL_SAMPLES_TARGET = 375     # 验证集条数
# 验证集条数（越大 eval 指标越稳定、震荡越小）。要 2500 训练 + 375 验证需总样本 ≥2875；可用 VITS_VAL_SAMPLES 覆盖
VAL_SAMPLES = int(os.environ.get("VITS_VAL_SAMPLES", "375"))
BATCH_SIZE = 16          # 减小batch_size减少震荡（可根据GPU显存自动调整）
BATCH_SIZE_MAX = 64      # 最大batch_size上限
BATCH_SIZE_MIN = 8       # 最小batch_size下限
# DataLoader 进程数：>0 可提高 GPU 利用率；若出现 pickle/tokenizer 错误可设 VITS_NUM_LOADER_WORKERS=0
NUM_LOADER_WORKERS = int(os.environ.get("VITS_NUM_LOADER_WORKERS", "12"))
PREFETCH_FACTOR = 4
GRADIENT_ACCUMULATION = 1  # 多优化器(gen/disc)时不支持>1
SAVE_STEP = 500
PRINT_STEP = 50

# ==================== GAN 训练平衡参数（豆包建议：渐进式训练）====================
# 阶段1：生成器预热 - 前 N 轮只训练生成器，让生成器先学会基本重建
# 阶段2：渐进解冻 - 逐步降低判别器权重
# 阶段3：正常训练 - 平衡 GAN 训练
DISCRIMINATOR_WARMUP_EPOCHS = 100  # 增加到100轮，让生成器充分学习（原50）
# 判别器损失权重缩放（降低以减轻对生成器的压制）
DISC_LOSS_SCALE_START = 0.3  # 初始0.3，逐步增加到0.5
DISC_LOSS_SCALE_END = 0.5    # 最终0.5

# ==================== 发散检测与自动降 LR（20000 轮：不降太快）====================
LR_FLOOR = 5e-7                    # 学习率下限，自动降 LR 时不低于此值
LR_REDUCE_MULTIPLIER = 0.75        # 每次降为当前的 75%（原 0.5 降太快），温和下降
MEL_RISE_THRESHOLD = 0.8           # 连续3轮 loss_mel 总上升超过此值才判为「发散」（0.5 太敏感）
DIVERGENCE_WARNINGS_BEFORE_REDUCE = 3  # 至少出现 3 次「连续3轮显著上升」才第一次降 LR
MAX_LR_REDUCTIONS = 4              # 20000 轮允许最多 4 次温和降 LR，超过后只打日志

# ==================== 断点续训 ====================
# F5 默认从「最新一次 run」续训（自动选 vits_moaning_output 下含 best_model/checkpoint 且最新的目录）。
# 指定目录：export VITS_CONTINUE_PATH="vits_moaning_output/run名"
def _get_latest_run_dir():
    """在 vits_moaning_output 下找含 best_model_*.pth 或 checkpoint_*.pth 且修改时间最新的目录（含子目录）。"""
    out = ROOT / "vits_moaning_output"
    if not out.is_dir():
        return ""
    best_dir, best_mtime = None, 0.0
    for d in out.rglob("*"):
        if not d.is_dir():
            continue
        models = list(d.glob("best_model_*.pth")) + list(d.glob("checkpoint_*.pth"))
        if not models:
            continue
        mtime = max(p.stat().st_mtime for p in models)
        if mtime > best_mtime:
            best_mtime = mtime
            best_dir = d
    if best_dir is None:
        return ""
    try:
        return str(best_dir.relative_to(ROOT))
    except ValueError:
        return str(best_dir)

CONTINUE_PATH = os.environ.get("VITS_CONTINUE_PATH", "").strip()
if not CONTINUE_PATH:
    CONTINUE_PATH = _get_latest_run_dir()
CONTINUE_PATH = CONTINUE_PATH.strip()

# ==================== MySQL 每轮指标写入 ====================
# 使用专用账户 tts_train / 数据库 tts_train / 表 train_round_codebuddy（需先启动 MySQL 并已建库建表）
# 可通过环境变量覆盖: MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "tts_train"),
    "password": os.environ.get("MYSQL_PASSWORD", "tts_train_pwd_2026"),
    "database": os.environ.get("MYSQL_DATABASE", "tts_train"),
    "charset": "utf8mb4",
}
# root@localhost 使用 auth_socket 时需用 Unix socket 连接（1698 错误时生效）
MYSQL_UNIX_SOCKET = os.environ.get("MYSQL_UNIX_SOCKET", "")
# 常见 socket 路径，按顺序尝试
_MYSQL_SOCKET_PATHS = [
    "/var/run/mysqld/mysqld.sock",
    "/tmp/mysql.sock",
    "/var/lib/mysql/mysql.sock",
    "/run/mysqld/mysqld.sock",
]


def _mysql_connect(use_database=True):
    """连接 MySQL：若 TCP 出现 1698/1045，则改用 Unix socket（解决 root@localhost auth_socket 权限问题）。"""
    import pymysql
    cfg = dict(MYSQL_CONFIG)
    db_name = cfg.get("database")
    if not use_database:
        cfg = {k: v for k, v in cfg.items() if k != "database"}
    # 先尝试 TCP
    try:
        return pymysql.connect(**cfg)
    except pymysql.err.OperationalError as e:
        errno = e.args[0] if e.args else 0
        if errno not in (1698, 1045):
            raise
        if cfg.get("host") not in ("localhost", "127.0.0.1") or cfg.get("password"):
            raise
        # 使用 Unix socket 重试
        sock_list = [MYSQL_UNIX_SOCKET] if MYSQL_UNIX_SOCKET else []
        sock_list += [p for p in _MYSQL_SOCKET_PATHS if os.path.exists(p)]
        if not sock_list:
            sock_list = _MYSQL_SOCKET_PATHS  # 仍尝试，让 pymysql 报错
        last_e = None
        for sock in sock_list:
            if not sock:
                continue
            try:
                conn = pymysql.connect(
                    unix_socket=sock,
                    user=cfg["user"],
                    database=db_name if use_database and db_name else None,
                    charset=MYSQL_CONFIG.get("charset", "utf8mb4"),
                )
                return conn
            except Exception as e2:
                last_e = e2
                continue
        raise last_e or e
    finally:
        pass


def _ensure_mysql_db_and_table():
    """确保数据库和 train_round_codebuddy 表存在，不存在则创建。"""
    import pymysql
    db_name = MYSQL_CONFIG.get("database", "tts_train")
    conn = None
    try:
        conn = _mysql_connect(use_database=False)
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
        conn.close()
        conn = _mysql_connect(use_database=True)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS train_round_codebuddy (
                    round_num INT PRIMARY KEY,
                    global_step INT NOT NULL DEFAULT 0,
                    loss_disc FLOAT NULL, loss_gen FLOAT NULL, loss_kl FLOAT NULL, loss_mel FLOAT NULL,
                    loss_duration FLOAT NULL, loss_feat FLOAT NULL, loss_1 FLOAT NULL,
                    eval_loss FLOAT NULL,
                    best_model_path VARCHAR(512) NULL,
                    updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
        print("✅ MySQL 数据库与表 train_round_codebuddy 已就绪")
    except Exception as e:
        print(f"⚠️ MySQL 初始化（建库/建表）跳过: {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ==================== 数据清理函数 ====================
def clean_moaning_text(text):
    """
    清理呻吟文本：仅做基本清理，禁用压缩，保证文本-音频对齐
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 禁用压缩！压缩会导致文本-音频不对齐
    # 例如："啊啊啊好舒服~" 压缩成 "啊~" 但音频还是3秒长
    # 模型学到的是输入2字符->输出3秒音频，造成推理混乱
    # 
    # 不满足条件的数据直接丢弃，绝不压缩！
    
    # 只保留基本清理：去掉竖线（分隔符冲突）
    text = text.replace("|", " ")
    
    # 5. 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text else "啊"  # 保底

def filter_dataset(df, min_duration=1.5, max_duration=7.0, min_text_len=1, max_text_len=50):
    """
    过滤数据集：只保留 1.5-7秒的黄金区间，并检查文本-音频长度匹配
    这个区间最适合训练，信息充足且训练稳定
    """
    original_len = len(df)
    
    # 确保必要列存在
    if 'duration' not in df.columns:
        print("⚠️  缺少 duration 列，尝试计算...")
        try:
            import librosa
            durations = []
            for path in df['audio_path']:
                try:
                    dur = librosa.get_duration(path=str(ROOT / path))
                    durations.append(dur)
                except:
                    durations.append(0)  # 标记为0，会被过滤
            df['duration'] = durations
        except:
            print("⚠️  无法计算时长，跳过时长过滤")
    
    # 严格过滤：只保留 1.5-7秒
    if 'duration' in df.columns:
        before_count = len(df)
        df = df[df['duration'] >= min_duration]  # 过滤超短
        df = df[df['duration'] <= max_duration]  # 过滤超长
        after_count = len(df)
        removed_count = before_count - after_count
        if removed_count > 0:
            print(f"   时长过滤: 移除 {removed_count} 条 (<{min_duration}s 或 >{max_duration}s)")
    
    # 过滤文本长度
    df = df[df['text'].str.len() >= min_text_len]
    df = df[df['text'].str.len() <= max_text_len]
    
    # 清理文本
    df['text'] = df['text'].apply(clean_moaning_text)
    
    # 过滤空文本
    df = df[df['text'].str.len() > 0]
    
    # ========== 新增：文本-音频长度匹配检查 ==========
    # 过滤文本过短但音频过长的样本（如"嗯啊啊"配5秒音频）
    if 'duration' in df.columns:
        MIN_CHARS_PER_SEC = 1.5  # 最小语速：1.5字/秒
        # 向量化计算语速，避免 apply 返回多列导致 ValueError
        text_lens = df['text'].astype(str).str.replace(' ', '', regex=False).str.replace('，', '', regex=False).str.replace('。', '', regex=False).str.len()
        dur = df['duration'].replace(0, float('nan'))
        df['chars_per_sec'] = (text_lens / dur).fillna(0)
        
        before_mismatch = len(df)
        # 保留语速正常的样本（>= 1.5字/秒）
        df = df[df['chars_per_sec'] >= MIN_CHARS_PER_SEC]
        after_mismatch = len(df)
        mismatch_removed = before_mismatch - after_mismatch
        
        if mismatch_removed > 0:
            print(f"   语速过滤: 移除 {mismatch_removed} 条 (文本短但音频长，语速<{MIN_CHARS_PER_SEC}字/秒)")
            # 显示几个被移除的示例
            removed_samples = df[df['chars_per_sec'] < MIN_CHARS_PER_SEC] if len(df) < before_mismatch else []
        
        # 删除临时列
        df = df.drop(columns=['chars_per_sec'])
    
    # 重置索引
    df = df.reset_index(drop=True)
    
    print(f"📊 数据过滤: {original_len} -> {len(df)} ({len(df)/original_len*100:.1f}%)")
    
    if len(df) == 0:
        print(f"❌ 错误: 过滤后没有样本！请检查数据时长是否在 {min_duration}-{max_duration}秒 范围内")
        
    return df


def _count_chinese_chars(text):
    """标注中汉字个数，用于过滤「不到2个汉字」的样本。"""
    if not text or not isinstance(text, str):
        return 0
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _find_mp3_for_basename(basename, downloads_dir):
    """根据标注文件名找 downloads 下同名音频。支持无空格 stem 与「数字 空格 其余」变体。"""
    downloads_dir = Path(downloads_dir)
    candidates = [basename]
    if basename and basename[0].isdigit():
        i = 0
        while i < len(basename) and basename[i].isdigit():
            i += 1
        if i < len(basename):
            candidates.append(basename[:i] + " " + basename[i:])
    for stem in candidates:
        for ext in (".mp3", ".wav", ".flac", ".m4a"):
            p = downloads_dir / (stem + ext)
            if p.exists():
                return p
    return None


def _parse_label_line(line):
    """解析一行标注：basename start_sec end_sec flag text。返回 (basename, start, end, flag, text) 或 None。"""
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


def _check_data_quality():
    """
    检查原始标注数据质量，返回 (是否通过, 统计信息)
    """
    print("\n" + "="*60)
    print("📊 数据质量预检查")
    print("="*60)
    
    if not DATASET_LABELS.is_dir():
        print(f"❌ 标注目录不存在: {DATASET_LABELS}")
        return False, {}
    
    all_texts = []
    all_durations = []
    total_files = 0
    
    for label_file in sorted(DATASET_LABELS.glob("*.txt")):
        total_files += 1
        with open(label_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = _parse_label_line(line)
                if parsed:
                    base, start_sec, end_sec, speaker, text = parsed
                    all_texts.append(text)
                    all_durations.append(end_sec - start_sec)
    
    if not all_texts:
        print("❌ 没有找到任何有效标注数据")
        return False, {}
    
    # 统计文本长度（中文字符）
    lengths = [_count_chinese_chars(t) for t in all_texts]
    unique_texts = len(set(all_texts))
    total_texts = len(all_texts)
    diversity_ratio = unique_texts / total_texts
    
    # 统计重复最严重的文本
    from collections import Counter
    text_counts = Counter(all_texts)
    most_common = text_counts.most_common(5)
    
    print(f"\n📈 数据统计:")
    print(f"   标注文件数: {total_files}")
    print(f"   总样本数: {total_texts}")
    print(f"   唯一文本数: {unique_texts}")
    print(f"   多样性比例: {diversity_ratio:.1%}")
    
    print(f"\n📏 文本长度分布（中文字符）:")
    for i in range(1, 11):
        count = sum(1 for l in lengths if l == i)
        if count > 0:
            print(f"   {i}字: {count}条 ({count/len(lengths)*100:.1f}%)")
    long_count = sum(1 for l in lengths if l > 10)
    if long_count > 0:
        print(f"   >10字: {long_count}条")
    
    print(f"\n⏱️ 时长分布:")
    short_dur = sum(1 for d in all_durations if d < 1.5)
    mid_dur = sum(1 for d in all_durations if 1.5 <= d <= 7.0)
    long_dur = sum(1 for d in all_durations if d > 7.0)
    print(f"   <1.5s: {short_dur}条 ({short_dur/len(all_durations)*100:.1f}%)")
    print(f"   1.5-7s: {mid_dur}条 ({mid_dur/len(all_durations)*100:.1f}%)")
    print(f"   >7s: {long_dur}条 ({long_dur/len(all_durations)*100:.1f}%)")
    
    print(f"\n🔥 重复最多的文本 (Top 5):")
    for text, count in most_common:
        print(f"   \"{text}\" 出现 {count} 次")
    
    # 质量检查标准
    issues = []
    if diversity_ratio < 0.3:
        issues.append(f"多样性比例过低 ({diversity_ratio:.1%} < 30%)")
    if most_common[0][1] > total_texts * 0.1:
        issues.append(f"文本重复严重 (\"{most_common[0][0]}\" 出现 {most_common[0][1]} 次)")
    if sum(1 for l in lengths if l >= 1) < total_texts * 0.5:
        issues.append("有效文本(<1字)占比过低")
    
    print(f"\n" + "="*60)
    if issues:
        print("⚠️ 数据质量问题:")
        for issue in issues:
            print(f"   - {issue}")
        print("="*60)
        return False, {
            "total": total_texts,
            "unique": unique_texts,
            "diversity": diversity_ratio,
            "issues": issues
        }
    else:
        print("✅ 数据质量检查通过")
        print("="*60)
        return True, {
            "total": total_texts,
            "unique": unique_texts,
            "diversity": diversity_ratio
        }


def build_dataset_from_labels(min_chinese=1, min_duration=1.5, max_duration=7.0, sample_rate=22050, max_samples=0, skip_quality_check=False):
    """
    从 dataset/labels + dataset/downloads 构建训练数据：
    - 只保留标注长度 >= min_chinese 个汉字的样本
    - 只保留时长在 [min_duration, max_duration] 秒的片段
    - max_samples>0 时只保留前 max_samples 条（用于快速试训，如 1000 条）
    - 从 mp3 切条为 wav 写入 lasttraincodebuddy/segments，并生成 train.csv / val.csv
    - 自动进行数据质量检查，不合格会停止训练
    """
    import random
    import shutil
    try:
        import librosa
    except ImportError:
        print("❌ 需要 librosa，请安装: pip install librosa")
        return False
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    try:
        from scipy.io import wavfile as scipy_wavfile
    except ImportError:
        scipy_wavfile = None

    if not DATASET_LABELS.is_dir():
        print(f"❌ 标注目录不存在: {DATASET_LABELS}")
        return False
    if not DATASET_DOWNLOADS.is_dir():
        print(f"❌ 音频目录不存在: {DATASET_DOWNLOADS}")
        return False
    
    # 数据质量预检查
    if not skip_quality_check:
        passed, stats = _check_data_quality()
        if not passed:
            print("\n❌ 数据质量检查未通过，停止训练！")
            print("请检查标注数据，解决上述问题后再训练。")
            print("如果确认要跳过检查，设置 skip_quality_check=True")
            return False

    segments_dir = ROOT / AUDIO_DIR
    if segments_dir.exists():
        if segments_dir.is_symlink():
            segments_dir.unlink()
        else:
            try:
                shutil.rmtree(segments_dir)
            except Exception as e:
                print(f"⚠️ 无法清空 {segments_dir}: {e}")
                return False
    segments_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped_short_text = 0
    skipped_duration = 0
    skipped_no_audio = 0
    skipped_error = 0
    skipped_duplicate = 0  # 新增：因重复过多被跳过
    text_occurrence = {}   # 新增：记录每个文本出现次数
    
    # 实时过滤参数
    MAX_TEXT_OCCURRENCE = 50  # 单个文本最多保留50次，超过的丢弃
    MIN_TEXT_LENGTH = 1       # 最少1个汉字
    MAX_TEXT_LENGTH = 50      # 最多50个汉字

    for label_file in sorted(DATASET_LABELS.glob("*.txt")):
        stem = label_file.stem
        mp3_path = _find_mp3_for_basename(stem, DATASET_DOWNLOADS)
        if not mp3_path or not mp3_path.exists():
            skipped_no_audio += 1
            continue
        try:
            full_audio, sr = librosa.load(str(mp3_path), sr=sample_rate, mono=True)
        except Exception as e:
            skipped_error += 1
            continue
        with open(label_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = _parse_label_line(line)
                if not parsed:
                    continue
                base, start_sec, end_sec, flag, text = parsed
                if _count_chinese_chars(text) < min_chinese:
                    skipped_short_text += 1
                    continue
                dur = end_sec - start_sec
                if dur < min_duration or dur > max_duration:
                    skipped_duration += 1
                    continue
                start_samp = int(start_sec * sr)
                end_samp = int(end_sec * sr)
                if start_samp >= len(full_audio) or end_samp <= start_samp:
                    skipped_error += 1
                    continue
                print(mp3_path.name,start_samp, end_samp,text,dur)
                segment_wav = full_audio[start_samp:end_samp]
                if len(segment_wav) < 0.2 * sr:
                    skipped_duration += 1
                    continue
                seg_name = f"segment_{len(rows):06d}"
                wav_path = segments_dir / f"{seg_name}.wav"
                try:
                    if sf:
                        sf.write(str(wav_path), segment_wav, sample_rate)
                    elif scipy_wavfile:
                        import numpy as _np
                        wav_int = (_np.clip(segment_wav, -1, 1) * 32767).astype(_np.int16)
                        scipy_wavfile.write(str(wav_path), sample_rate, wav_int)
                    else:
                        skipped_error += 1
                        continue
                except Exception:
                    skipped_error += 1
                    continue
                text_clean = clean_moaning_text(text)
                text_len = _count_chinese_chars(text_clean)
                
                # 实时过滤1：文本长度检查
                if not text_clean or text_len < MIN_TEXT_LENGTH or text_len > MAX_TEXT_LENGTH:
                    wav_path.unlink(missing_ok=True)
                    skipped_short_text += 1
                    continue
                
                # 实时过滤2：重复文本限制（单个文本最多保留MAX_TEXT_OCCURRENCE次）
                text_occurrence[text_clean] = text_occurrence.get(text_clean, 0) + 1
                if text_occurrence[text_clean] > MAX_TEXT_OCCURRENCE:
                    wav_path.unlink(missing_ok=True)
                    skipped_duplicate += 1
                    continue
                
                rows.append({"audio_path": f"{AUDIO_DIR}/{seg_name}.wav", "text": text_clean})
                if max_samples > 0 and len(rows) >= max_samples:
                    break
        print(f"   已处理 {len(rows)} 条...")
        if max_samples > 0 and len(rows) >= max_samples:
            break
        if len(rows) % 500 == 0 and len(rows) > 0:
            print(f"   已处理 {len(rows)} 条...")

    if not rows:
        print(f"❌ 从 labels 未得到任何合格样本（汉字>={min_chinese}，时长 {min_duration}-{max_duration}s）")
        print(f"   跳过: 标注<{min_chinese}汉字={skipped_short_text}, 时长不符={skipped_duration}, 重复过多={skipped_duplicate}, 无音频/错误={skipped_no_audio}/{skipped_error}")
        return False

    random.seed(42)
    random.shuffle(rows)
    if max_samples > 0 and len(rows) > max_samples:
        rows = rows[:max_samples]
        print(f"   已截断为 {max_samples} 条。")
    # 如果数据够1100条，严格按1000训练+100验证分割；不够则按比例
    total_needed = TRAIN_SAMPLES_TARGET + VAL_SAMPLES_TARGET
    if len(rows) >= total_needed:
        n_train = TRAIN_SAMPLES_TARGET
        n_val = VAL_SAMPLES_TARGET
    else:
        # 数据不足时，验证集至少10条，最多100条，其余给训练集
        n_val = min(VAL_SAMPLES_TARGET, max(10, len(rows) // 11))
        n_train = len(rows) - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    train_csv = ROOT / "train.csv"
    val_csv = ROOT / "val.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")

    # 计算最终数据多样性
    unique_texts_final = len(set(r['text'] for r in rows))
    diversity_final = unique_texts_final / len(rows) if rows else 0
    
    print(f"✅ 已从 labels+downloads 生成: train={n_train}, val={n_val}（验证集=VAL_SAMPLES={VAL_SAMPLES}，汉字>={min_chinese}，时长 {min_duration}-{max_duration}s）")
    print(f"   跳过: 标注<{min_chinese}汉字={skipped_short_text}, 时长不符={skipped_duration}, 重复过多={skipped_duplicate}, 无音频={skipped_no_audio}, 错误={skipped_error}")
    print(f"   数据多样性: {unique_texts_final}/{len(rows)} ({diversity_final:.1%}) 唯一文本")
    print(f"   音频已写入 {segments_dir}，CSV 已写入 {train_csv} / {val_csv}")
    
    # 如果多样性仍然过低，给出警告
    if diversity_final < 0.3:
        print(f"\n⚠️ 警告: 数据多样性过低 ({diversity_final:.1%} < 30%)，建议检查原始标注数据")
    return True


# ==================== 数据处理模块（从原始标注生成训练集）====================
def process_dataset_from_raw():
    """
    从原始标注和音频生成标准化训练数据集
    处理流程:
    1. 读取标注文件 (dataset/labels/*.txt)
    2. 匹配音频文件 (dataset/downloads/*.wav)
    3. 音频格式转换、采样率统一 (22050Hz)
    4. 时长校验、文本清洗
    5. 生成 train.csv / val.csv 和 segments/
    """
    # 配置
    LABELS_DIR = ROOT.parent / "dataset" / "labels"
    DOWNLOADS_DIR = ROOT.parent / "dataset" / "downloads"
    SEGMENTS_DIR = ROOT / "segments"
    
    # 数据处理参数
    TARGET_SAMPLE_RATE = 22050
    MIN_DURATION = 1
    MAX_DURATION = 8.0
    MIN_CHINESE_CHARS = 2
    MIN_CHARS_PER_SEC = 1.5
    VAL_RATIO = 0.13
    
    def get_audio_info(wav_path):
        """获取音频信息 (时长, 采样率)"""
        try:
            with contextlib.closing(wave.open(str(wav_path), 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                return duration, rate
        except Exception as e:
            return None, None
    
    def resample_audio(src_path, dst_path, target_sr=22050):
        """重采样音频到目标采样率"""
        try:
            import librosa
            import soundfile as sf
            y, sr = librosa.load(str(src_path), sr=None, mono=True)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sf.write(str(dst_path), y, target_sr)
            return True
        except Exception as e:
            print(f"    重采样失败: {e}")
            return False
    
    def clean_text(text):
        """清洗文本"""
        if not text:
            return ""
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9]', '', text)
        return text.strip()
    
    def count_chinese_chars(text):
        """统计中文字符数"""
        return len(re.findall(r'[\u4e00-\u9fff]', text))
    
    print("\n" + "="*60)
    print("从原始数据生成训练数据集")
    print("="*60)
    
    # 检查目录
    if not LABELS_DIR.exists():
        print(f"⚠️ 警告: 标注目录不存在 {LABELS_DIR}，跳过数据生成")
        return False
    if not DOWNLOADS_DIR.exists():
        print(f"⚠️ 警告: 音频目录不存在 {DOWNLOADS_DIR}，跳过数据生成")
        return False
    
    print(f"标注目录: {LABELS_DIR}")
    print(f"音频目录: {DOWNLOADS_DIR}")
    
    # 1. 加载标注
    print("\n1. 加载标注文件...")
    all_entries = []
    label_files = sorted(LABELS_DIR.glob("*.txt"))
    print(f"   发现 {len(label_files)} 个标注文件")
    
    for label_file in label_files:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                parts = line.split('|')
                if len(parts) >= 5:
                    filename = parts[0]
                    speaker = parts[1] if len(parts) > 1 else "female"
                    text = parts[4] if len(parts) > 4 else parts[2]
                    all_entries.append({'filename': filename, 'speaker': speaker, 'text': text})
    
    print(f"   总计读取: {len(all_entries)} 条标注")
    
    # 2. 匹配音频
    print("\n2. 匹配音频文件...")
    matched = []
    for entry in all_entries:
        filename = entry['filename']
        wav_path = DOWNLOADS_DIR / f"{filename}.wav"
        mp3_path = DOWNLOADS_DIR / f"{filename}.mp3"
        m4a_path = DOWNLOADS_DIR / f"{filename}.m4a"
        
        if wav_path.exists():
            entry['audio_path'] = wav_path
            entry['format'] = 'wav'
            matched.append(entry)
        elif mp3_path.exists():
            entry['audio_path'] = mp3_path
            entry['format'] = 'mp3'
            matched.append(entry)
        elif m4a_path.exists():
            entry['audio_path'] = m4a_path
            entry['format'] = 'm4a'
            matched.append(entry)
    
    print(f"   匹配成功: {len(matched)} 条")
    
    # 3. 处理音频
    print("\n3. 处理音频文件...")
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    processed = []
    skipped = {'duration': 0, 'resample': 0, 'text_short': 0, 'speed': 0}
    
    for i, entry in enumerate(matched, 1):
        src_path = entry['audio_path']
        dst_name = f"segment_{i:06d}"
        dst_path = SEGMENTS_DIR / f"{dst_name}.wav"
        
        duration, sample_rate = get_audio_info(src_path)
        need_convert = entry['format'] != 'wav' or sample_rate != TARGET_SAMPLE_RATE
        
        if need_convert:
            success = resample_audio(src_path, dst_path, TARGET_SAMPLE_RATE)
            if not success:
                skipped['resample'] += 1
                continue
            duration, _ = get_audio_info(dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        
        if duration is None or duration < MIN_DURATION or duration > MAX_DURATION:
            skipped['duration'] += 1
            dst_path.unlink(missing_ok=True)
            continue
        
        text = clean_text(entry['text'])
        chinese_count = count_chinese_chars(text)
        
        if chinese_count < MIN_CHINESE_CHARS:
            skipped['text_short'] += 1
            dst_path.unlink(missing_ok=True)
            continue
        
        chars_per_sec = chinese_count / duration
        if chars_per_sec < MIN_CHARS_PER_SEC:
            skipped['speed'] += 1
            dst_path.unlink(missing_ok=True)
            continue
        
        processed.append({
            'segment_id': dst_name,
            'text': text,
            'speaker': entry['speaker'],
            'duration': duration,
            'chars_per_sec': chars_per_sec,
            'wav_path': str(dst_path)
        })
        
        if i % 100 == 0:
            print(f"   已处理: {i}/{len(matched)}...")
    
    print(f"   通过检查: {len(processed)} 条")
    print(f"   跳过: 时长{skipped['duration']}, 转换{skipped['resample']}, 文本短{skipped['text_short']}, 语速慢{skipped['speed']}")
    
    if len(processed) == 0:
        print("⚠️ 没有通过检查的数据!")
        return False
    
    # 4. 划分数据集
    print("\n4. 划分数据集...")
    random.seed(42)
    random.shuffle(processed)
    
    val_size = int(len(processed) * VAL_RATIO)
    val_size = min(val_size, VAL_SAMPLES_TARGET)
    
    val_entries = processed[:val_size]
    train_entries = processed[val_size:]
    
    # 限制训练集数量
    if len(train_entries) > TRAIN_SAMPLES_TARGET:
        train_entries = train_entries[:TRAIN_SAMPLES_TARGET]
    
    print(f"   训练集: {len(train_entries)} 条")
    print(f"   验证集: {len(val_entries)} 条")
    
    # 5. 保存
    print("\n5. 保存数据集...")
    train_csv = ROOT / "train.csv"
    val_csv = ROOT / "val.csv"
    
    with open(train_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'audio_path', 'duration'])
        writer.writeheader()
        for entry in train_entries:
            writer.writerow({'text': entry['text'], 'audio_path': entry['wav_path'], 'duration': entry['duration']})
    
    with open(val_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'audio_path', 'duration'])
        writer.writeheader()
        for entry in val_entries:
            writer.writerow({'text': entry['text'], 'audio_path': entry['wav_path'], 'duration': entry['duration']})
    
    # 生成元数据
    meta_train = SEGMENTS_DIR / "meta_train_vits.txt"
    meta_val = SEGMENTS_DIR / "meta_val_vits.txt"
    
    with open(meta_train, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            f.write(f"{entry['segment_id']}|{entry['speaker']}|{entry['text']}\n")
    
    with open(meta_val, 'w', encoding='utf-8') as f:
        for entry in val_entries:
            f.write(f"{entry['segment_id']}|{entry['speaker']}|{entry['text']}\n")
    
    print(f"   已保存: {train_csv}, {val_csv}")
    print(f"   元数据: {meta_train}, {meta_val}")
    
    # 统计
    print("\n6. 数据统计")
    durations = [e['duration'] for e in processed]
    speeds = [e['chars_per_sec'] for e in processed]
    print(f"   总样本: {len(processed)}")
    print(f"   平均时长: {sum(durations)/len(durations):.2f}秒")
    print(f"   平均语速: {sum(speeds)/len(speeds):.2f}字/秒")
    
    return True


def prepare_data():
    """准备数据：加载、清理、生成 meta 文件。若 CSV 不存在，自动从原始数据生成。"""
    train_csv = ROOT / "train.csv"
    val_csv = ROOT / "val.csv"
    
    # 如果 CSV 不存在，尝试从原始数据生成
    if not train_csv.exists():
        print(f"⚠️  未找到 {train_csv}，尝试从原始数据生成...")
        success = process_dataset_from_raw()
        if not success:
            print(f"❌ 错误: 无法生成训练数据")
            sys.exit(1)
    
    print("📂 加载数据...")
    train_df = pd.read_csv(train_csv, dtype={"text": str, "audio_path": str})
    val_df = pd.read_csv(val_csv, dtype={"text": str, "audio_path": str})
    
    # 从 CSV 推断音频所在目录；若当前 segments 下没有 wav，则让 segments 指向该目录（直接用现有数据）
    segments_dir = ROOT / AUDIO_DIR
    def _common_audio_dir(paths):
        if not paths:
            return None
        try:
            parent = Path(str(paths[0])).resolve().parent
            for p in paths[1:]:
                if Path(str(p)).resolve().parent != parent:
                    return None
            return parent
        except Exception:
            return None
    
    all_paths = list(train_df["audio_path"].dropna()) + list(val_df["audio_path"].dropna())
    csv_audio_dir = _common_audio_dir(all_paths)
    if csv_audio_dir is not None:
        csv_audio_dir = Path(csv_audio_dir).resolve()
    n_wav_local = 0
    if segments_dir.exists():
        try:
            n_wav_local = sum(1 for _ in segments_dir.glob("*.wav"))
        except Exception:
            pass
    if n_wav_local == 0 and csv_audio_dir is not None and csv_audio_dir.is_dir():
        try:
            need_link = True
            if segments_dir.exists():
                if segments_dir.is_symlink() and segments_dir.resolve() == csv_audio_dir:
                    need_link = False
                else:
                    # 无 wav 时用 CSV 目录替换：删掉当前 segments（空目录或仅有 meta 或旧链接）
                    import shutil
                    if segments_dir.is_symlink():
                        segments_dir.unlink()
                    else:
                        shutil.rmtree(segments_dir)
            if need_link:
                (ROOT / AUDIO_DIR).symlink_to(csv_audio_dir)
                print(f"📂 检测到音频在 CSV 指定目录，已链接: {AUDIO_DIR} -> {csv_audio_dir}")
        except Exception as e:
            print(f"⚠️  无法链接 segments 到 CSV 音频目录: {e}")
            if not segments_dir.exists():
                segments_dir.mkdir(parents=True, exist_ok=True)
    elif not segments_dir.exists():
        segments_dir.mkdir(parents=True, exist_ok=True)
    
    segments_dir = ROOT / AUDIO_DIR
    print(f"📝 原始数据: train={len(train_df)}, val={len(val_df)}（音频目录: {segments_dir.resolve()}）")
    
    # 数据清理（先用 CSV 原始路径算时长，再统一为 segments/ 并过滤不存在的）
    print("🧹 清理数据...")
    train_df = filter_dataset(train_df)
    val_df = filter_dataset(val_df)
    
    # 统一为「当前目录/segments/文件名」，训练时从 ROOT/segments（或链接目标）读
    train_df["audio_path"] = train_df["audio_path"].apply(lambda p: f"{AUDIO_DIR}/{Path(p).name}")
    val_df["audio_path"] = val_df["audio_path"].apply(lambda p: f"{AUDIO_DIR}/{Path(p).name}")
    
    # 只保留磁盘上存在的音频（路径为 ROOT/segments/，若已链接则指向 CSV 所在目录）
    def exists_wav(row):
        ap = Path(row["audio_path"])
        wav_file = segments_dir / f"{ap.stem}.wav"
        return wav_file.exists()
    train_before = len(train_df)
    val_before = len(val_df)
    train_df = train_df[train_df.apply(exists_wav, axis=1)].reset_index(drop=True)
    val_df = val_df[val_df.apply(exists_wav, axis=1)].reset_index(drop=True)
    if train_before != len(train_df) or val_before != len(val_df):
        print(f"   文件存在性过滤: train {train_before} -> {len(train_df)}, val {val_before} -> {len(val_df)} (已剔除不存在的 wav)")
    
    # 显示样本
    print("\n📝 清理后文本样本:")
    for i, text in enumerate(train_df['text'].head(10)):
        print(f"   {i+1}. {text}")
    
    # 生成 Coqui 格式 meta 文件，写入 segments 目录（与音频同目录，便于统一管理）
    meta_train = segments_dir / "meta_train_vits.txt"
    meta_val = segments_dir / "meta_val_vits.txt"
    
    # ljspeech 从 ROOT/wavs/ 读音频；此处固定用 segments，通过 wavs -> segments 让 DataLoader 读到
    wavs_dir = ROOT / "wavs"
    if wavs_dir.exists():
        if wavs_dir.is_symlink():
            if wavs_dir.resolve() != segments_dir.resolve():
                wavs_dir.unlink()
                wavs_dir.symlink_to(segments_dir)
                print(f"📂 已更新 wavs -> {AUDIO_DIR}")
        else:
            print(f"⚠️  存在非符号链接的 wavs 目录，请删除或移走，脚本仅从 {AUDIO_DIR}/ 读取音频")
    if not wavs_dir.exists() and segments_dir.exists():
        wavs_dir.symlink_to(segments_dir)
        print(f"📂 已创建 wavs -> {AUDIO_DIR}，DataLoader 从 {AUDIO_DIR}/ 读取音频")
    
    def to_meta(df, out_path):
        lines = []
        for _, row in df.iterrows():
            ap = Path(row["audio_path"])
            text = str(row["text"])
            lines.append(f"{ap.stem}|female|{text}")  # ljspeech: cols[0]=id, cols[2]=text
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return len(lines)
    
    train_count = to_meta(train_df, meta_train)
    val_count = to_meta(val_df, meta_val)
    
    # 收集训练+验证集中所有出现过的字符；仅 ASCII 标点放 punctuations，其余（含中文标点）放 characters
    ASCII_PUNCT = set("!'(),-.:;? \t\n\r")
    all_text = "".join(train_df["text"].astype(str).tolist() + val_df["text"].astype(str).tolist())
    chars_from_data = "".join(sorted(set(c for c in all_text if c not in ASCII_PUNCT)))
    print(f"📊 词表字符数: {len(chars_from_data)}")
    
    print(f"\n✅ 生成 meta 文件: train={train_count}, val={val_count}（已写入 {segments_dir}/）")
    
    return str(meta_train), str(meta_val), train_count, chars_from_data


def _ensure_segments_linked_to_csv():
    """若缺乏数据时调用：强制从 train/val.csv 推断音频目录，并把 segments 链接到该目录，再准备数据时能读到。"""
    train_csv = ROOT / "train.csv"
    val_csv = ROOT / "val.csv"
    if not train_csv.exists():
        return
    try:
        train_df = pd.read_csv(train_csv, dtype={"audio_path": str})
        if "audio_path" not in train_df.columns:
            return
    except Exception:
        return
    val_df = pd.read_csv(val_csv, dtype={"audio_path": str}) if val_csv.exists() and val_csv.stat().st_size > 0 else pd.DataFrame()
    all_paths = list(train_df["audio_path"].dropna())
    if "audio_path" in getattr(val_df, "columns", []):
        all_paths += list(val_df["audio_path"].dropna())
    if not all_paths:
        return
    try:
        parent = Path(str(all_paths[0])).resolve().parent
        if not all(Path(str(p)).resolve().parent == parent for p in all_paths):
            return
    except Exception:
        return
    if not parent.is_dir():
        return
    segments_dir = ROOT / AUDIO_DIR
    try:
        import shutil
        if segments_dir.exists():
            if segments_dir.is_symlink():
                segments_dir.unlink()
            else:
                shutil.rmtree(segments_dir)
        segments_dir.symlink_to(parent)
        print(f"📂 已重新链接 {AUDIO_DIR} -> {parent}，将重新加载数据")
    except Exception as e:
        print(f"⚠️  重新链接失败: {e}")


def train():
    """训练 VITS - 针对呻吟数据优化，支持断点续训与 duration 加速收敛"""
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.configs.shared_configs import CharactersConfig
    from TTS.tts.models.vits import Vits, VitsDataset
    from TTS.tts.utils.text import tokenizer as _tts_tokenizer_module
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.datasets import load_tts_samples
    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.config import load_config as tts_load_config

    # 屏蔽「Character 'x' not found in the vocabulary」及整句英文测试句的刷屏（词表为中文时会出现）
    def _encode_quiet(self, text):
        token_ids = []
        for char in text:
            try:
                idx = self.characters.char_to_id(char)
                token_ids.append(idx)
            except KeyError:
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
            # 不再 print(text) 与 print(" [!] Character ...")
        return token_ids
    _tts_tokenizer_module.TTSTokenizer.encode = _encode_quiet

    # 避免 VitsDataset 在 rescue 时 rescue_item_idx 超出 len(samples) 导致 IndexError
    _vits_getitem_orig = VitsDataset.__getitem__
    def _vits_getitem_safe(self, idx):
        n = len(self.samples)
        if n == 0:
            raise IndexError("VitsDataset.samples is empty")
        idx = int(idx) % n
        return _vits_getitem_orig(self, idx)
    VitsDataset.__getitem__ = _vits_getitem_safe

    continue_dir = (ROOT / CONTINUE_PATH).resolve() if CONTINUE_PATH else None
    if continue_dir and not continue_dir.is_dir():
        print(f"❌ 续训目录不存在: {continue_dir}，请检查 CONTINUE_PATH 或 VITS_CONTINUE_PATH")
        sys.exit(1)
    if continue_dir is not None:
        print(f"🔁 本次为断点续训，将从以下目录恢复: {continue_dir}")

    # 检查是否已有数据，有则跳过构建
    train_csv = ROOT / "train.csv"
    val_csv = ROOT / "val.csv"
    segments_dir = ROOT / AUDIO_DIR
    
    if train_csv.exists() and val_csv.exists() and segments_dir.exists() and any(segments_dir.glob("*.wav")):
        print(f"\n✅ 检测到已有数据，跳过数据构建:")
        print(f"   - train.csv: {train_csv}")
        print(f"   - val.csv: {val_csv}")
        print(f"   - segments/: {len(list(segments_dir.glob('*.wav')))} 个音频文件")
        print(f"\n💡 如需重新生成数据，请删除上述文件后重新运行")
    elif not os.environ.get("VITS_SKIP_LABELS") and DATASET_LABELS.is_dir() and DATASET_DOWNLOADS.is_dir():
        print(f"\n📂 从标注+音频构建训练数据（标注至少 1 个汉字，时长 1.5–7 秒，最多 {MAX_TRAIN_SAMPLES or '全部'} 条）...")
        if build_dataset_from_labels(min_chinese=1, min_duration=1.5, max_duration=7.0, max_samples=MAX_TRAIN_SAMPLES or 0):
            print("✅ 数据构建完成，继续训练流程。")
        else:
            print("⚠️ 未从 labels 生成新数据，将使用已有 train.csv/val.csv（若存在）。")

    # 准备数据；若缺乏训练数据则强制按 CSV 重链 segments 并重新加载一次
    meta_train, meta_val, num_samples, chars_from_data = prepare_data()
    if num_samples == 0:
        print("\n⚠️  缺乏训练数据，正在按 CSV 重新链接音频目录并重新加载数据...")
        _ensure_segments_linked_to_csv()
        meta_train, meta_val, num_samples, chars_from_data = prepare_data()
    if num_samples == 0:
        print("❌ 错误: 没有有效的训练样本，请确认 train.csv/val.csv 中的音频路径存在且时长为 1.5–7 秒")
        sys.exit(1)
    
    print("\n🔧 创建 VITS 配置...")
    
    # 断点续训：从 run 目录加载 config，保证与已保存模型一致
    if continue_dir is not None:
        config_path = continue_dir / "config.json"
        if not config_path.exists():
            print(f"❌ 续训目录下未找到 config.json: {config_path}")
            sys.exit(1)
        config = tts_load_config(str(config_path))
        config.output_path = str(continue_dir)  # 续训时写入原 run 目录
        config.dur_loss_alpha = 2.5  # 抑制 duration 缓慢上升，与下方新训练一致
        config.mel_loss_alpha = getattr(config, "mel_loss_alpha", 35.0)
        if config.mel_loss_alpha < 35.0:
            config.mel_loss_alpha = 35.0  # 续训时也提高 mel 权重，推动 loss_mel 向 20 以下
        if not isinstance(getattr(config, "grad_clip", None), list):
            config.grad_clip = [1.0, 1.0]  # 多优化器要求列表
        config.save_json(str(config_path))  # 写回以便 Trainer 加载时使用
        print(f"📂 断点续训: 已从 {continue_dir} 加载配置（dur_loss_alpha=2.5, mel_loss_alpha≥35 已写入）")
    else:
        config = VitsConfig()
        config.model = "vits"
        config.output_path = str(OUTPUT_DIR)
        config.run_name = "vits_moaning_voice"
        config.dur_loss_alpha = 2.5  # 抑制验证集 duration 缓慢上升；mel_loss_alpha 见下方
    
    # ========== 字符模式 + 中文词表（续训时沿用已加载 config，不覆盖）==========
    if continue_dir is None:
        config.use_phonemes = False
        config.characters = CharactersConfig(
            pad="<PAD>",
            punctuations="!'(),-.:;? \t\n\r",
            characters=chars_from_data,
            phonemes="",
        )
    
    # ========== 自动检测GPU显存并计算最优batch_size ==========
    def _get_optimal_batch_size():
        """根据GPU显存自动计算最优batch_size"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    mem_total = float(parts[0])  # MB
                    mem_used = float(parts[1])   # MB
                    mem_free = mem_total - mem_used
                    
                    # VITS模型大约需要：
                    # - 基础显存: ~2GB
                    # - 每batch_size=1大约需要: ~0.5GB (取决于音频长度)
                    # 预留20%安全余量
                    safe_mem = mem_free * 0.8
                    base_mem = 2048  # 2GB基础
                    mem_per_sample = 512  # 每个样本约0.5GB
                    
                    # 计算理论最大batch_size
                    theoretical_max = int((safe_mem - base_mem) / mem_per_sample)
                    
                    # 限制在合理范围内，优先稳定性
                    optimal = max(BATCH_SIZE_MIN, min(theoretical_max, BATCH_SIZE_MAX))
                    
                    # 如果理论值很小，使用最小值
                    if theoretical_max < BATCH_SIZE_MIN:
                        optimal = BATCH_SIZE_MIN
                        print(f"⚠️ 显存紧张(可用{mem_free:.0f}MB)，使用最小batch_size={BATCH_SIZE_MIN}")
                    else:
                        print(f"📊 GPU显存: 总共{mem_total:.0f}MB, 已用{mem_used:.0f}MB, 可用{mem_free:.0f}MB")
                        print(f"   理论最大batch_size: {theoretical_max}, 实际使用: {optimal}")
                    
                    return optimal
        except Exception as e:
            print(f"⚠️ 无法自动检测显存: {e}，使用默认batch_size={BATCH_SIZE}")
        return BATCH_SIZE
    
    # 获取最优batch_size（新训练时计算，续训时使用已有值）
    if continue_dir is None:
        optimal_batch_size = _get_optimal_batch_size()
    else:
        optimal_batch_size = getattr(config, 'batch_size', BATCH_SIZE)
    
    # 训练参数 - 保守配置（避免 loss 爆炸）；续训时沿用已保存值
    if continue_dir is None:
        config.epochs = EPOCHS
        config.batch_size = optimal_batch_size  # 使用自动计算的最优batch_size
        config.eval_batch_size = min(16, optimal_batch_size)  # 验证batch_size不超过训练值
    else:
        config.eval_batch_size = min(16, optimal_batch_size)  # 续训时使用已有batch_size
    config.save_step = SAVE_STEP
    config.print_step = PRINT_STEP
    # 延后 test_run，避免首轮 test_run 时因 test_sentences/合成 触发 IndexError（修好后再改回 0）
    config.test_delay_epochs = getattr(config, "test_delay_epochs", 0) if continue_dir else 9999
    # 若仍出现 IndexError 且堆栈在 eval_log/_log，可设环境变量 VITS_SKIP_EVAL=1 关闭验证
    if os.environ.get("VITS_SKIP_EVAL", "").strip() == "1":
        config.run_eval = False
        print("⚠️ 已关闭验证 (VITS_SKIP_EVAL=1)")
    
    # 数据加载：多进程提升 GPU 利用率（若报 pickle/tokenizer 错可设 VITS_NUM_LOADER_WORKERS=0）
    config.num_loader_workers = max(0, NUM_LOADER_WORKERS)
    config.phoneme_cache_path = None
    config.pin_memory = True
    config.prefetch_factor = PREFETCH_FACTOR
    config.use_amp = True
    config.gradient_accumulation_steps = GRADIENT_ACCUMULATION
    
    # ========== 关键：只保留 1.5-7 秒音频 ==========
    config.min_text_len = 1
    config.max_text_len = 50
    config.min_audio_len = 33075   # 1.5秒 @22050Hz
    config.max_audio_len = 154350  # 7秒 @22050Hz
    
    # 数据集配置（meta 已写入 segments/，路径相对 ROOT）
    config.datasets = [
        BaseDatasetConfig(
            formatter="ljspeech",
            dataset_name="moaning_voice",
            path=str(ROOT),
            meta_file_train=f"{AUDIO_DIR}/meta_train_vits.txt",
            meta_file_val=f"{AUDIO_DIR}/meta_val_vits.txt",
        )
    ]
    # 避免 Trainer 在 test_run 时因 test_sentences 为空或格式错误导致 IndexError
    if not getattr(config, "test_sentences", None) or not config.test_sentences:
        config.test_sentences = [
            ["啊"],
            ["嗯"],
            ["好的"],
        ]
    else:
        # 过滤掉空列表，防止 get_aux_input_from_test_sentences([]) 导致 text=None 引发错误
        config.test_sentences = [s for s in config.test_sentences if (isinstance(s, list) and len(s) > 0) or (not isinstance(s, list) and s is not None)]
        if not config.test_sentences:
            config.test_sentences = [["啊"], ["嗯"], ["好的"]]
    
    # 音频配置
    config.audio.sample_rate = 22050
    config.audio.fft_size = 1024
    config.audio.win_length = 1024
    config.audio.hop_length = 256
    config.audio.num_mels = 80
    config.audio.mel_fmin = 0
    config.audio.mel_fmax = None
    
    # VITS 模型参数（保守，减少参数量）
    config.vits = {
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [8, 8, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "n_layers_q": 3,
        "use_spectral_norm": False
    }
    
    # 训练损失权重（VITS 使用 config.mel_loss_alpha / dur_loss_alpha，非 config.train）
    # 日志分析：验证集 loss_mel 徘徊 27~28 无持续下降，loss_duration 缓慢上升；适当提高 mel 权重推动 mel 向 20 以下
    config.mel_loss_alpha = 35.0   # 提高 mel 权重（默认 45，曾用 20 保守），使验证集 loss_mel 有持续下降趋势、目标 <20
    config.dur_loss_alpha = 2.5   # 略提高时长权重，抑制验证集 duration 缓慢上升（原 2.0）
    config.train = {
        "seg_len": 8192,
        "port": 5000,
        "c_mel": 35,
        "c_kl": 1.0,
        "c_commit": 1.0,
        "c_reconstruct": 1.0
    }
    
    # ========== 自适应学习率调整 ==========
    # 根据历史训练记录动态调整初始学习率
    def _get_adaptive_lr_and_suggestions():
        """根据之前的训练历史，智能选择学习率并给出建议"""
        history_file = OUTPUT_DIR / "training_history.json"
        suggestions = []
        
        if history_file.exists():
            try:
                import json
                with open(history_file, 'r') as f:
                    history = json.load(f)
                last_mel = history.get('last_loss_mel', 999)
                best_mel = history.get('best_loss_mel', 999)
                
                # 如果上一轮训练发散(loss_mel > 40)，大幅降低学习率
                if last_mel > 40:
                    suggestions.append(f"🔴 上一轮训练发散(loss_mel={last_mel:.2f})，建议学习率 0.000002")
                    return 0.000002, suggestions
                # 如果震荡严重(最后loss比最好loss高很多)
                elif last_mel > best_mel * 1.2 and best_mel < 30:
                    suggestions.append(f"🟡 上一轮震荡严重(last={last_mel:.2f}, best={best_mel:.2f})，建议学习率 0.000008")
                    suggestions.append(f"💡 当前batch_size={optimal_batch_size}，建议保持稳定")
                    return 0.000008, suggestions
                # 如果训练停滞在25以上
                elif best_mel > 25:
                    suggestions.append(f"🟠 loss_mel停滞在{best_mel:.2f}")
                    suggestions.append("💡 建议检查数据质量，确保文本-音频对齐")
                    return 0.000015, suggestions
                # 如果训练良好
                elif best_mel < 20:
                    suggestions.append(f"🟢 上一轮训练良好(best={best_mel:.2f})，保持当前配置")
                    return 0.00001, suggestions
            except Exception:
                pass
        return 0.000002, [f"ℹ️ 首次训练，batch_size={optimal_batch_size}，学习率 0.000002（豆包建议：保守起步）"]  # 更保守的学习率
    
    initial_lr, suggestions = _get_adaptive_lr_and_suggestions()
    
    # ========== 保守优化器（避免 loss 爆炸）==========
    # lr 单独设置，不要放在 optimizer_params 中（VITS 内部会处理）
    config.lr = initial_lr
    config.optimizer_params = {
        "betas": [0.8, 0.99],
        "eps": 1e-09,
        "weight_decay": 0.0
    }
    # ========== 余弦退火学习率调度（目标：500 轮后 mel/duration/disc/gen 等损失仍能持续下降）==========
    # VITS 使用 lr_scheduler_gen / lr_scheduler_disc，必须显式设置；T_max 拉长使多指标在 500+ 轮仍有有效学习率
    # 断点续训时此处会覆盖从 config.json 读入的旧 T_max/eta_min，新 schedule 生效
    eta_min = max(initial_lr * 0.55, 8e-7)  # 提高最小学习率：后期 mel/duration/GAN 仍能微调
    t_max = EPOCHS * 6  # 退火周期（按 epoch）：20000 轮时 T_max=120000，学习率在长周期内缓慢下降，不会很快掉到底
    config.lr_scheduler = "CosineAnnealingLR"
    config.lr_scheduler_params = {"T_max": t_max, "eta_min": eta_min}
    # VITS 双优化器：gen/disc 共用同一套余弦退火，保证判别器与生成器同步、多损失均衡下降
    config.lr_scheduler_gen = "CosineAnnealingLR"
    config.lr_scheduler_gen_params = {"T_max": t_max, "eta_min": eta_min}
    config.lr_scheduler_disc = "CosineAnnealingLR"
    config.lr_scheduler_disc_params = {"T_max": t_max, "eta_min": eta_min}
    config.scheduler_after_epoch = True  # 按 epoch 步进，避免按 step 时学习率掉得过快导致 65 轮就接近为 0
    
    # 梯度裁剪（防止梯度爆炸）- 豆包建议：加强裁剪压制梯度
    config.grad_clip = [0.3, 0.3]  # [disc, gen]，从0.5降到0.3，更严格
    
    # 加载数据
    print("\n📂 加载数据集...")
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_size=0.05,
    )
    print(f"✅ 加载完成: train={len(train_samples)}, eval={len(eval_samples)}")
    
    if len(train_samples) == 0:
        print("❌ 错误: 没有加载到训练样本，检查数据路径和格式")
        sys.exit(1)
    if len(eval_samples) == 0:
        # 验证集为空时 Trainer 会 IndexError，从训练集取少量作为验证集
        n_eval = min(10, max(1, len(train_samples) // 10))
        eval_samples = list(train_samples[:n_eval])
        print(f"⚠️ 验证集为空，已从训练集取前 {n_eval} 条作为验证集")
    
    # ========== 数据质量检查 ==========
    print("\n🔍 数据质量检查...")
    
    def check_data_quality(samples, label="train"):
        """检查样本质量，返回通过检查的样本"""
        MIN_CHARS_PER_SEC = 1.5  # 最小语速：1.5字/秒
        MIN_DURATION = 1.5  # 最短1.5秒
        MAX_DURATION = 7.0  # 最长7秒
        
        good_samples = []
        bad_count = 0
        
        for sample in samples:
            text = sample.get('text', '')
            audio_path = sample.get('audio_file', '')
            
            # 获取音频时长（从wav文件读取）
            duration = None
            try:
                import wave
                with wave.open(str(audio_path), 'r') as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
            except:
                # 如果无法读取，尝试用sample中的信息
                duration = sample.get('duration', None)
            
            if duration is None:
                good_samples.append(sample)
                continue
            
            # 计算文本长度（去除标点）
            text_clean = text.replace(' ', '').replace('，', '').replace('。', '').replace('、', '')
            text_len = len(text_clean)
            
            # 检查语速
            chars_per_sec = text_len / duration if duration > 0 else 0
            
            # 过滤条件
            is_bad = False
            if duration < MIN_DURATION:
                is_bad = True
            elif duration > MAX_DURATION:
                is_bad = True
            elif chars_per_sec < MIN_CHARS_PER_SEC:
                is_bad = True
            
            if is_bad:
                bad_count += 1
            else:
                good_samples.append(sample)
        
        print(f"   {label}: 原始 {len(samples)}条 -> 过滤后 {len(good_samples)}条 (移除 {bad_count}条)")
        return good_samples
    
    # 执行质量检查
    train_samples = check_data_quality(train_samples, "训练集")
    eval_samples = check_data_quality(eval_samples, "验证集")
    
    # ========== 强制使用指定数量的样本 ==========
    # 如果样本不足，给出警告；如果样本过多，随机采样
    random.seed(42)  # 固定随机种子，保证可复现
    
    print(f"\n📊 强制使用配置: train={TRAIN_SAMPLES_TARGET}, val={VAL_SAMPLES_TARGET}")
    
    # 处理训练集
    if len(train_samples) < TRAIN_SAMPLES_TARGET:
        print(f"⚠️ 警告: 训练集只有 {len(train_samples)} 条，少于目标 {TRAIN_SAMPLES_TARGET} 条")
        print(f"   将使用全部 {len(train_samples)} 条训练")
    else:
        # 随机采样到目标数量
        random.shuffle(train_samples)
        train_samples = train_samples[:TRAIN_SAMPLES_TARGET]
        print(f"✅ 训练集: 随机采样 {TRAIN_SAMPLES_TARGET} 条")
    
    # 处理验证集
    if len(eval_samples) < VAL_SAMPLES_TARGET:
        print(f"⚠️ 警告: 验证集只有 {len(eval_samples)} 条，少于目标 {VAL_SAMPLES_TARGET} 条")
        print(f"   将使用全部 {len(eval_samples)} 条验证")
    else:
        # 随机采样到目标数量
        random.shuffle(eval_samples)
        eval_samples = eval_samples[:VAL_SAMPLES_TARGET]
        print(f"✅ 验证集: 随机采样 {VAL_SAMPLES_TARGET} 条")
    
    print(f"\n📈 最终数据: train={len(train_samples)}, eval={len(eval_samples)}")
    
    # 初始化音频处理器和 tokenizer
    print("🔧 初始化音频处理器...")
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    # tokenizer.add_blank 时最大 token id = num_chars（即 1583），故 embedding 需 num_chars+1
    n_chars = tokenizer.characters.num_chars
    required_num_chars = n_chars + 1 if tokenizer.add_blank else n_chars
    config.model_args.num_chars = required_num_chars
    print(f"📊 词表大小 num_chars: {config.model_args.num_chars}")
    
    # 初始化 VITS 模型（从头训练）
    print("🏗️  初始化 VITS 模型...")
    # 临时去掉 config.characters，避免 BaseTTS._set_model_args 用 tokenizer 的 num_chars 覆盖
    _saved_characters = getattr(config, "characters", None)
    if _saved_characters is not None and required_num_chars != n_chars:
        config.characters = None
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    if _saved_characters is not None:
        config.characters = _saved_characters
    # 若模型仍被写成 n_chars，则扩 embedding 并复制权重
    if model.args.num_chars < required_num_chars:
        emb = model.text_encoder.emb
        new_emb = torch.nn.Embedding(required_num_chars, emb.embedding_dim, padding_idx=emb.padding_idx if hasattr(emb, "padding_idx") else None).to(emb.weight.device)
        with torch.no_grad():
            new_emb.weight[: model.args.num_chars].copy_(emb.weight)
        model.text_encoder.emb = new_emb
        model.args.num_chars = required_num_chars
    # 强制使用 train_step 路径，避免 Trainer 误调未实现的 optimize()
    model.optimize = None
    
    # 移动到 GPU
    if gpu:
        model = model.cuda()
        # 禁用 torch.compile，避免与 CUDNN 索引越界冲突
        # try:
        #     model = torch.compile(model, mode="reduce-overhead")
        #     print("✅ torch.compile 已启用")
        # except Exception as e:
        #     print(f"⚠️  torch.compile 失败: {e}")
    
    # 训练参数（断点续训时找到 best_model，并生成「仅模型」checkpoint 以重置学习率/调度器）
    restore_path = ""
    if continue_dir is not None:
        # 优先加载 best_model，如果没有则找最新的 checkpoint
        best_models = sorted(continue_dir.glob("best_model_*.pth"), key=lambda p: int(p.stem.split('_')[-1]), reverse=True)
        checkpoints = sorted(continue_dir.glob("checkpoint_*.pth"), key=lambda p: int(p.stem.split('_')[-1]), reverse=True)
        source_ckpt = None
        if best_models:
            source_ckpt = best_models[0]
            print(f"📂 断点续训: 将从最优模型恢复: {source_ckpt.name}")
        elif checkpoints:
            source_ckpt = checkpoints[0]
            print(f"📂 断点续训: 将从 checkpoint 恢复: {source_ckpt.name}")
        else:
            print(f"⚠️ 警告: 在 {continue_dir} 中未找到任何模型文件")
        
        # 生成「仅模型」checkpoint，使 Trainer 只恢复权重、不恢复 optimizer/scheduler，从而使用新学习率与 T_max（500+ 轮内各指标持续下降）
        if source_ckpt is not None:
            try:
                try:
                    ckpt = torch.load(str(source_ckpt), map_location="cpu", weights_only=False)
                except TypeError:
                    ckpt = torch.load(str(source_ckpt), map_location="cpu")
                if isinstance(ckpt, dict) and "model" in ckpt:
                    step = ckpt.get("step", 0)
                    epoch = ckpt.get("epoch", 0)
                    if "best_model_" in source_ckpt.name:
                        try:
                            step = int(source_ckpt.stem.split("_")[-1])
                        except Exception:
                            pass
                    restore_only = {"model": ckpt["model"], "step": step, "epoch": epoch}
                    model_only_path = continue_dir / "restore_model_only.pth"
                    torch.save(restore_only, str(model_only_path))
                    restore_path = str(model_only_path)
                    print(f"🔄 已生成仅模型 checkpoint（重置 LR/调度器）: {model_only_path.name}，目标: 1000+ 轮内 loss_mel/duration/disc/gen 持续下降")
            except Exception as e:
                print(f"⚠️ 生成仅模型 checkpoint 失败 ({e})，将直接使用原文件（可能沿用旧学习率）")
                restore_path = str(source_ckpt)
    
    # 使用 restore_path 加载指定模型，continue_path 为空避免自动恢复
    train_args = TrainerArgs(
        restore_path=restore_path,
        continue_path="",
        gpu=0 if gpu else None,
    )
    
    # 启动时确保 MySQL 库和表存在，避免首轮写入报错
    _ensure_mysql_db_and_table()
    
    # 训练监控：记录loss趋势，检测发散
    _loss_mel_history = []  # 用于检测发散
    _divergence_warning_count = 0
    _lr_reduce_count = 0    # 已执行的自动降 LR 次数（不超过 MAX_LR_REDUCTIONS）
    _gpu_adjust_count = 0  # GPU调整计数
    
    def _get_gpu_info():
        """获取GPU状态信息"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]  # 取第一张卡
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    return {
                        'utilization': float(parts[0]),  # GPU利用率%
                        'memory_used': float(parts[1]),   # 显存使用MB
                        'memory_total': float(parts[2]),  # 显存总量MB
                        'temperature': float(parts[3]),   # 温度°C
                        'power': float(parts[4])          # 功耗W
                    }
        except Exception as e:
            pass
        return None
    
    def _adjust_params_based_on_gpu(trainer, gpu_info, round_num):
        """根据GPU状态动态调整训练参数"""
        if gpu_info is None:
            return
        
        util = gpu_info['utilization']
        mem_used = gpu_info['memory_used']
        mem_total = gpu_info['memory_total']
        temp = gpu_info['temperature']
        mem_percent = (mem_used / mem_total) * 100
        
        global _gpu_adjust_count
        adjustments = []
        
        # 温度过高(>85°C)，降低batch_size或暂停
        if temp > 85:
            print(f"\n🔥 GPU温度过高({temp}°C)！建议暂停训练降温")
            if temp > 90:
                print("🔴 严重警告: 温度超过90°C，建议立即停止训练！")
        
        # 显存使用率过低(<50%)，可以增加batch_size
        if mem_percent < 40 and util > 80:
            adjustments.append(f"💡 显存占用低({mem_percent:.1f}%)，可适当增大batch_size提升效率")
            if config.batch_size < BATCH_SIZE_MAX:
                adjustments.append(f"   建议: 下次训练可尝试 batch_size={min(config.batch_size + 8, BATCH_SIZE_MAX)}")
        
        # 显存使用率过高(>90%)，减少batch_size
        if mem_percent > 90:
            adjustments.append(f"⚠️ 显存占用高({mem_percent:.1f}%)，建议减小batch_size防止OOM")
            if config.batch_size > BATCH_SIZE_MIN:
                adjustments.append(f"   建议: 下次训练可降低至 batch_size={max(config.batch_size - 4, BATCH_SIZE_MIN)}")
        
        # GPU利用率过低(<30%)，可能是数据加载瓶颈
        if util < 30 and round_num > 10:
            adjustments.append(f"🐌 GPU利用率低({util:.1f}%)，建议增加num_workers或检查数据加载")
        
        # GPU利用率很高(>95%)且温度正常，表现良好
        if util > 90 and temp < 80:
            adjustments.append(f"✅ GPU表现良好(利用率{util:.1f}%, 温度{temp}°C)")
        
        # 每10轮输出一次GPU状态报告
        if round_num % 10 == 0 and round_num > 0:
            print(f"\n📊 GPU状态报告 (Round {round_num}):")
            print(f"   利用率: {util:.1f}% | 显存: {mem_used:.0f}/{mem_total:.0f}MB ({mem_percent:.1f}%) | 温度: {temp:.0f}°C")
            if adjustments:
                for adj in adjustments:
                    print(f"   {adj}")
            else:
                print(f"   ✅ 状态正常")
    
    # 判别器冻结状态跟踪
    _discriminator_frozen = False
    _freeze_until_epoch = 0
    
    # 每轮结束后回调函数（暂停MySQL写入，改为GPU监控）
    def _on_epoch_end_callback(trainer):
        nonlocal _discriminator_frozen, _freeze_until_epoch
        round_num = int(trainer.epochs_done)
        
        # 获取训练指标
        ev = getattr(trainer, "keep_avg_eval", None)
        avg = ev.avg_values if ev else {}
        loss_mel = avg.get("avg_loss_mel")
        loss_gen = avg.get("avg_loss_gen")
        loss_disc = avg.get("avg_loss_disc")
        
        # ===== 判别器预热阶段（前N轮冻结判别器） =====
        if round_num <= DISCRIMINATOR_WARMUP_EPOCHS and not _discriminator_frozen:
            if hasattr(model, 'discriminator') and model.discriminator is not None:
                # 冻结判别器参数
                for param in model.discriminator.parameters():
                    param.requires_grad = False
                _discriminator_frozen = True
                print(f"\n🥶 判别器预热阶段: 前{DISCRIMINATOR_WARMUP_EPOCHS}轮冻结判别器，只训练生成器")
        
        # 解冻判别器（预热结束）
        if round_num == DISCRIMINATOR_WARMUP_EPOCHS + 1 and _discriminator_frozen and _freeze_until_epoch == 0:
            if hasattr(model, 'discriminator') and model.discriminator is not None:
                for param in model.discriminator.parameters():
                    param.requires_grad = True
                _discriminator_frozen = False
                print(f"\n🔥 预热结束: 判别器已解冻，开始联合训练")
        
        # 临时解冻（如果是因发散而冻结的）
        if _freeze_until_epoch > 0 and round_num > _freeze_until_epoch:
            if hasattr(model, 'discriminator') and model.discriminator is not None:
                for param in model.discriminator.parameters():
                    param.requires_grad = True
                _discriminator_frozen = False
                _freeze_until_epoch = 0
                print(f"\n🔥 判别器已恢复训练（发散恢复期结束）")
        
        # ===== 智能训练监控 =====
        if loss_mel is not None:
            _loss_mel_history.append(float(loss_mel))
            
            # 只保留最近10轮
            if len(_loss_mel_history) > 10:
                _loss_mel_history.pop(0)
            
            # 检测发散：连续3轮上升 且 总上升 >= MEL_RISE_THRESHOLD（避免 21.81→21.83→22.09 等小幅波动触发）
            if len(_loss_mel_history) >= 4:
                recent_3 = _loss_mel_history[-3:]
                total_rise = recent_3[2] - recent_3[0]
                is_consecutive_rise = recent_3[0] < recent_3[1] < recent_3[2]
                is_significant_rise = total_rise >= MEL_RISE_THRESHOLD
                if is_consecutive_rise and is_significant_rise:
                    nonlocal _divergence_warning_count, _lr_reduce_count
                    _divergence_warning_count += 1
                    print(f"\n⚠️ 警告: loss_mel连续3轮上升 {recent_3[0]:.2f} → {recent_3[1]:.2f} → {recent_3[2]:.2f}（总上升 {total_rise:.2f}）")
                    if _divergence_warning_count >= DIVERGENCE_WARNINGS_BEFORE_REDUCE and _lr_reduce_count < MAX_LR_REDUCTIONS:
                        new_lr = max(config.lr * LR_REDUCE_MULTIPLIER, LR_FLOOR)
                        config.lr = new_lr
                        if hasattr(trainer, 'optimizer') and trainer.optimizer:
                            optimizers = trainer.optimizer if isinstance(trainer.optimizer, list) else [trainer.optimizer]
                            for opt in optimizers:
                                for param_group in opt.param_groups:
                                    param_group['lr'] = new_lr
                        _lr_reduce_count += 1
                        print(f"🔴 严重警告: 检测到训练发散！自动降低学习率到 {new_lr:.6f}（不低于 {LR_FLOOR:.0e}，已降 {_lr_reduce_count}/{MAX_LR_REDUCTIONS} 次）")
                        print(f"✅ 优化器学习率已调整为 {new_lr:.6f}")
                        # 临时冻结判别器10轮，让生成器恢复
                        if hasattr(model, 'discriminator') and model.discriminator is not None and not _discriminator_frozen:
                            for param in model.discriminator.parameters():
                                param.requires_grad = False
                            _discriminator_frozen = True
                            _freeze_until_epoch = round_num + 10
                            print(f"🥶 临时冻结判别器10轮，让生成器恢复学习")
                    elif _divergence_warning_count >= DIVERGENCE_WARNINGS_BEFORE_REDUCE and _lr_reduce_count >= MAX_LR_REDUCTIONS:
                        print(f"ℹ️ 已达自动降LR上限({MAX_LR_REDUCTIONS}次)，仅记录日志，不再降LR；可手动调整 config.lr 后续训")
            
            # 检测停滞：最近5轮变化小于5%
            if len(_loss_mel_history) >= 5:
                recent_5 = _loss_mel_history[-5:]
                variation = (max(recent_5) - min(recent_5)) / (min(recent_5) + 1e-8)
                if variation < 0.05 and loss_mel > 25:
                    print(f"\nℹ️ 提示: loss_mel停滞在 {loss_mel:.2f}，已5轮无明显改善")
                    if round_num > 100:
                        print(f"💡 建议: 如果到200轮仍无法降到22以下，建议增加数据量或检查数据质量")
        
        # ===== GPU状态监控（每10轮） =====
        if round_num % 10 == 0:
            gpu_info = _get_gpu_info()
            _adjust_params_based_on_gpu(trainer, gpu_info, round_num)

    # 初始化训练器
    print("🎯 初始化训练器...")
    trainer = Trainer(
        train_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"audio_processor": ap},
        parse_command_line_args=False,
        callbacks={"on_epoch_end": _on_epoch_end_callback},
    )
    
    # 打印GPU信息
    print("\n" + "="*60)
    print("🎮 GPU状态检测")
    print("="*60)
    gpu_info = _get_gpu_info()
    if gpu_info:
        mem_percent = (gpu_info['memory_used'] / gpu_info['memory_total']) * 100
        print(f"   GPU利用率: {gpu_info['utilization']:.1f}%")
        print(f"   显存使用: {gpu_info['memory_used']:.0f}/{gpu_info['memory_total']:.0f} MB ({mem_percent:.1f}%)")
        print(f"   温度: {gpu_info['temperature']:.0f}°C")
        print(f"   功耗: {gpu_info['power']:.1f}W")
        
        # 给出初始建议
        if gpu_info['temperature'] > 80:
            print(f"\n   ⚠️ 警告: GPU温度较高({gpu_info['temperature']:.0f}°C)，建议改善散热")
        if mem_percent > 90:
            print(f"   ⚠️ 警告: 显存占用高({mem_percent:.1f}%)，建议减小batch_size")
        if gpu_info['utilization'] < 50:
            print(f"   💡 提示: GPU利用率偏低，可以增加num_workers提升效率")
    else:
        print("   ⚠️ 无法获取GPU信息，请确保nvidia-smi可用")
    
    # 打印配置
    print("\n" + "="*60)
    print(f"🚀 VITS 呻吟语音训练配置 (保守版){' [断点续训]' if continue_dir else ''}")
    print(f"="*60)
    
    # 打印智能建议
    if suggestions:
        print("\n📋 训练建议:")
        for s in suggestions:
            print(f"   {s}")
        print()
    
    print(f"📊 训练样本: {len(train_samples)} {'(快速测试模式)' if QUICK_TEST_MODE else ''}")
    print(f"📊 验证样本: {len(eval_samples)}")
    print(f"🔄 训练轮数: {EPOCHS}")
    if QUICK_TEST_MODE:
        print(f"💡 提示: 快速测试模式，先让生成器学会基本重建，稳定后再用完整数据")
    print(f"📦 Batch Size: {config.batch_size} (自适应，等效 {config.batch_size * GRADIENT_ACCUMULATION})")
    print(f"🔄 Gradient Accumulation: {GRADIENT_ACCUMULATION}")
    print(f"📉 学习率: {config.lr} → 余弦退火至 {eta_min:.2e}，T_max={t_max} 轮（目标: 500 轮后 mel/duration/disc/gen 仍可下降）")
    print(f"📅 学习率调度: CosineAnnealingLR (gen/disc 一致，按 epoch 步进)")
    print(f"✂️  梯度裁剪: {config.grad_clip} (disc, gen)")
    print(f"👷 DataLoader Workers: {config.num_loader_workers} (可设 VITS_NUM_LOADER_WORKERS 覆盖)")
    print(f"⏱️  最短音频: {config.min_audio_len / 22050:.2f}秒")
    print(f"🔤 音素支持: 中文 (use_phonemes=True)")
    print(f"⚡ 混合精度: True")
    print(f"🎯 预训练模型: 不需要")
    print(f"\n🎭 GAN 平衡策略（豆包渐进式训练）:")
    print(f"   • mel_loss_alpha: {getattr(config, 'mel_loss_alpha', 35)} (目标: 验证集 loss_mel 逐步降至 20 以下)")
    print(f"   • dur_loss_alpha: {getattr(config, 'dur_loss_alpha', 2.5)} (抑制验证集 duration 缓慢上升)")
    print(f"   • 判别器预热: 前{DISCRIMINATOR_WARMUP_EPOCHS}轮冻结判别器（生成器先学）")
    print(f"   • 梯度裁剪: [0.3, 0.3] (严格限制梯度)")
    print(f"   • 学习率: {config.lr} (保守起步)")
    print(f"   • 策略: 先生成器稳定 → 再逐步平衡GAN → 最后调细节")
    print(f"="*60 + "\n")
    
    # 开始训练
    print("🎯 开始训练 VITS...")
    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\n⚠️  训练被中断")
    except BaseException as e:
        import traceback
        print(f"\n❌ 训练出错: {type(e).__name__}: {e}")
        traceback.print_exc()
        # 若是 IndexError，打印完整异常链便于定位
        if isinstance(e, IndexError) or (getattr(e, "__context__", None) is not None):
            exc = getattr(e, "__context__", None) or e
            while exc is not None:
                if isinstance(exc, IndexError):
                    print("\n--- 原始 IndexError 堆栈 ---")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                exc = getattr(exc, "__context__", None)
        sys.exit(1)
    
    # 保存训练历史，用于下次自适应调整
    try:
        import json
        history_file = OUTPUT_DIR / "training_history.json"
        
        # 从trainer获取训练历史
        train_history = {}
        if hasattr(trainer, 'train_loss') and trainer.train_loss:
            # 获取最近的loss_mel
            recent_evals = [v for k, v in trainer.train_loss.items() if 'eval_loss_mel' in str(k)]
            if recent_evals:
                train_history['last_loss_mel'] = recent_evals[-1]
                train_history['best_loss_mel'] = min(recent_evals)
        
        # 如果没有从trainer获取到，尝试从日志文件读取
        if not train_history:
            log_file = OUTPUT_DIR / f"{config.run_name}" / "trainer_0_log.txt"
            if log_file.exists():
                mel_values = []
                with open(log_file, 'r') as f:
                    for line in f:
                        if 'avg_loss_mel' in line:
                            try:
                                val = float(line.split(':')[1].split()[0])
                                mel_values.append(val)
                            except:
                                pass
                if mel_values:
                    train_history['last_loss_mel'] = mel_values[-1]
                    train_history['best_loss_mel'] = min(mel_values)
        
        if train_history:
            with open(history_file, 'w') as f:
                json.dump(train_history, f, indent=2)
            print(f"📊 训练历史已保存: {train_history}")
    except Exception as e:
        print(f"⚠️ 保存训练历史失败: {e}")
    
    print("\n✅ 训练完成!")
    print(f"📁 模型保存: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
