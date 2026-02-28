"""
TTS 全流程：支持「仅加载检查」与「真正训练」两种模式。
真正训练使用 Coqui 官方 Trainer + 你的 train.csv/val.csv 与 models1 预训练权重。
适配低版本Coqui TTS：移除无效的TrainerArgs参数，手动用DataParallel实现1卡训练。
训练前会过滤掉过短、过长音频，只保留时长在 [MIN_AUDIO_SEC, MAX_AUDIO_SEC] 内的样本。
"""
import logging
import os
import sys
import torch
import pandas as pd
from pathlib import Path

try:
    import librosa
except ImportError:
    librosa = None

import torch

# 启用 TF32 加速（Ampere/Ada 架构，5090 支持）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # 自动寻找最快算法
# 减少显存碎片（OOM 时尝试）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 优化 CPU 线程
os.environ["OMP_NUM_THREADS"] = "32"  # 匹配 25 vCPU
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"
# 旧版 checkpoint 含 defaultdict，PyTorch 2.6+ weights_only=True 会报错
import trainer.io as _trainer_io
_trainer_io._WEIGHTS_ONLY = False

# ----------------------------
# 配置（调试时可改这里）
# ----------------------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models1"
model_path = MODEL_DIR / "model_file.pth"
config_path = MODEL_DIR / "config.json"

# True = 真正训练（用 TTS Trainer）；False = 只加载模型并做数据检查
DO_REAL_TRAIN = True

# ===== 多GPU核心配置（适配低版本Coqui TTS）=====
NUM_GPUS = 1  # 显式指定3块GPU
GPU_IDS = list(range(NUM_GPUS))  # [0,1,2]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))  # 限定使用的GPU
# 自动检测GPU可用性（防止无GPU时报错）
os.environ["OMP_NUM_THREADS"] = "12" 
gpu = torch.cuda.is_available() and NUM_GPUS > 0
print(f"GPU可用: {gpu}, 启用GPU数量: {NUM_GPUS}, GPU ID: {GPU_IDS}")

# 训练规模：200 轮、1000 条数据（DEBUG_TRAIN=True 时为快速调试）
DEBUG_TRAIN = False
if DEBUG_TRAIN:
    DEBUG_SMALL_RUN = 80       # 只用前 N 条样本
    DEBUG_EPOCHS = 2
    DEBUG_BATCH_SIZE = 4       
    DEBUG_SAVE_STEP = 600
    DEBUG_PRINT_STEP = 5
    DEBUG_EVAL_SPLIT_SIZE = 0.08
else:
    DEBUG_SMALL_RUN = None      # None = 用全部数据
    DEBUG_EPOCHS = 10000        # 训练 1 万轮
    DEBUG_BATCH_SIZE = 64       # 32GB 显存：256 易 OOM，64~128 更稳；可逐步提高到 96/128
    DEBUG_SAVE_STEP = 500
    DEBUG_PRINT_STEP = 25
    DEBUG_EVAL_SPLIT_SIZE = 0.15

# 梯度累积（如果单卡batch太小，用累积保证训练稳定性）
GRADIENT_ACCUMULATION_STEPS = max(1, 32 // (DEBUG_BATCH_SIZE * NUM_GPUS))

OUTPUT_DIR = ROOT / "tts_train_output"
META_TRAIN = ROOT / "meta_train_coqui.txt"
META_VAL = ROOT / "meta_val_coqui.txt"

# 断点续训：从头训练就设为 None（只加载 models1 预训练权重）
CONTINUE_RUN_DIR = None
if CONTINUE_RUN_DIR is not None:
    CONTINUE_RUN_DIR = Path(CONTINUE_RUN_DIR).resolve()

# 音频时长过滤：过短/过长片段不参与训练（秒）
MIN_AUDIO_SEC = 1   # 过短（如纯呼吸、噪声）丢弃
MAX_AUDIO_SEC = 7.0  # 过长丢弃，避免训练不稳定


def _compute_duration(path: str) -> float | None:
    """返回音频时长（秒），出错时返回 None。"""
    if librosa is None:
        return None
    try:
        return float(librosa.get_duration(path=path))
    except Exception as e:
        print(f"[时长] 加载失败，跳过 {path}: {e}")
        return None


def filter_by_duration(df: pd.DataFrame, min_sec: float, max_sec: float) -> pd.DataFrame:
    """按音频时长过滤样本，剔除过短/过长的音频。"""
    if librosa is None:
        print("[时长过滤] 未安装 librosa，跳过音频时长过滤")
        return df
    keep_rows = []
    removed_short = 0
    removed_long = 0
    removed_fail = 0
    for _, row in df.iterrows():
        audio_path = str(row["audio_path"])
        dur = _compute_duration(audio_path)
        if dur is None:
            removed_fail += 1
            continue
        if dur < min_sec:
            removed_short += 1
            continue
        if dur > max_sec:
            removed_long += 1
            continue
        keep_rows.append(row)
    out_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    print(
        f"[时长过滤] 总数={len(df)} -> 保留={len(out_df)}，"
        f"过短(<{min_sec}s)删除={removed_short}，过长(>{max_sec}s)删除={removed_long}，加载失败={removed_fail}"
    )
    return out_df


# ----------------------------
# 0. 运行前先执行 tts_process.py，生成 segments、train.csv、val.csv
# ----------------------------
from importlib.util import spec_from_file_location, module_from_spec
_tts_process_spec = spec_from_file_location("tts_process", ROOT / "tts_process.py")
_tts_process = module_from_spec(_tts_process_spec)
_tts_process_spec.loader.exec_module(_tts_process)
_tts_process.main()

# ----------------------------
# 1. 检查并读取 train.csv / val.csv
# ----------------------------
train_csv = ROOT / "train.csv"
val_csv = ROOT / "val.csv"
if not train_csv.exists() or not val_csv.exists():
    print("未找到 train.csv 或 val.csv（tts_process 可能未产生有效样本）")
    sys.exit(1)

# 优化：指定dtype减少内存占用，过滤空值
train_df = pd.read_csv(train_csv, dtype={"text": str, "audio_path": str}, skip_blank_lines=True)
val_df = pd.read_csv(val_csv, dtype={"text": str, "audio_path": str}, skip_blank_lines=True)
train_df = train_df[(train_df["text"].notna()) & (train_df["audio_path"].notna())].reset_index(drop=True)
val_df = val_df[(val_df["text"].notna()) & (val_df["audio_path"].notna())].reset_index(drop=True)
print(f"原始样本: train={len(train_df)}, val={len(val_df)}")

# 按音频时长过滤：删除过短、过长音频
train_df = filter_by_duration(train_df, MIN_AUDIO_SEC, MAX_AUDIO_SEC)
val_df = filter_by_duration(val_df, MIN_AUDIO_SEC, MAX_AUDIO_SEC)
print(f"过滤后: Train samples: {len(train_df)}, Val samples: {len(val_df)}")
if len(train_df) == 0 or len(val_df) == 0:
    print("错误：过滤后训练集或验证集为空，请检查音频文件或放宽 MIN_AUDIO_SEC/MAX_AUDIO_SEC")
    sys.exit(1)

# ----------------------------
# 2. 生成 Coqui 格式的 meta 文件（audio_file|text|speaker_name|emotion_name）
#    audio_file 为相对于 ROOT 的路径，便于 formatter 里 root_path + audio_file
# ----------------------------
def df_to_coqui_meta(df: pd.DataFrame, out_path: Path) -> None:
    lines = ["audio_file|text|speaker_name|emotion_name"]
    for _, row in df.iterrows():
        ap = Path(row["audio_path"])
        try:
            rel = ap.relative_to(ROOT)
        except ValueError:
            rel = ap.name
        text = (row["text"] or "").replace("|", " ")  # 避免破坏分隔
        lines.append(f"{rel}|{text}|female|neutral")
    out_path.write_text("\n".join(lines), encoding="utf-8")

df_to_coqui_meta(train_df, META_TRAIN)
df_to_coqui_meta(val_df, META_VAL)
print(f"已生成 {META_TRAIN.name}, {META_VAL.name}")

if not DO_REAL_TRAIN:
    # ---------- 仅加载模型 + 分桶统计（不训练）----------
    from TTS.api import TTS
    if not model_path.exists() or not config_path.exists():
        print(f"未找到本地模型: {model_path} 或 {config_path}")
        sys.exit(1)
    tts = TTS(model_path=str(model_path), config_path=str(config_path), gpu=gpu)
    def bucket_data(df, max_len):
        return df[df["text"].str.len() <= max_len].reset_index(drop=True)
    for bucket in [{"name": "short", "max_len": 30}, {"name": "medium", "max_len": 60}, {"name": "long", "max_len": 200}]:
        bt = bucket_data(train_df, bucket["max_len"])
        bv = bucket_data(val_df, bucket["max_len"])
        print(f"  {bucket['name']}: train={len(bt)}, val={len(bv)}（已跳过训练）")
    print("训练完成（未执行训练）✅")
    sys.exit(0)

# ----------------------------
# 3. 真正训练：加载 config + 手动DataParallel实现多卡训练
# ----------------------------
from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model
from trainer import Trainer, TrainerArgs
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger

setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())

is_continue = CONTINUE_RUN_DIR is not None and CONTINUE_RUN_DIR.exists()
if is_continue:
    if not (CONTINUE_RUN_DIR / "config.json").exists():
        print(f"续训目录无效（缺少 config.json）: {CONTINUE_RUN_DIR}")
        sys.exit(1)
    print("断点续训：将从", CONTINUE_RUN_DIR, "恢复")
elif not model_path.exists() or not config_path.exists():
    print(f"未找到本地模型: {model_path} 或 {config_path}")
    sys.exit(1)

# 续训时从 run 目录读 config，否则从 models1 读
config = load_config(str(CONTINUE_RUN_DIR / "config.json") if is_continue else str(config_path))

# ===== 适配低版本的核心配置 =====
config.output_path = str(OUTPUT_DIR)
config.audio.stats_path = str(MODEL_DIR / "scale_stats.npy")
config.datasets = [
    BaseDatasetConfig(
        formatter="coqui",
        dataset_name="baker_fine",
        path=str(ROOT),
        meta_file_train=META_TRAIN.name,
        meta_file_val=META_VAL.name,
        language="zh-cn",
    )
]
# 训练超参（总batch = 单卡batch * GPU数）
config.epochs = DEBUG_EPOCHS
config.batch_size = DEBUG_BATCH_SIZE  # 这里填单卡batch，DataParallel会自动拆分
config.save_step = DEBUG_SAVE_STEP
config.print_step = DEBUG_PRINT_STEP
config.eval_split_size = DEBUG_EVAL_SPLIT_SIZE
config.run_eval = True
config.print_eval = True
config.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
config.loader_num_workers = 8   # 降低可省 CPU/内存，避免与 GPU 争资源
config.loader_pin_memory = True  # 加速数据传输到GPU
config.use_amp = True  # 混合精度训练，减少显存占用
# 按长度分组的组大小，略放大有利于大 batch 下减少 padding、提高吞吐
if hasattr(config, 'batch_group_size'):
    config.batch_group_size = min(16, max(8, DEBUG_BATCH_SIZE // 16))
# 重要：NoamLR 在 scheduler_after_epoch=True 时按 step=epoch 计数；warmup_steps=4000 表示 4000 个 epoch 才 warmup 完，
# 导致前期 lr 极小（~1e-7）、align_error 一直卡在 0.88+ 不降。改为 10 个 epoch 内 warmup 完，attention 才能学到对齐。
if getattr(config, "lr_scheduler", None) == "NoamLR" and getattr(config, "lr_scheduler_params", None):
    try:
        config.lr_scheduler_params["warmup_steps"] = 10
    except TypeError:
        config.lr_scheduler_params = {**dict(config.lr_scheduler_params), "warmup_steps": 10}
    print("[学习率] NoamLR warmup_steps 已改为 10（原 4000 导致 align_error 居高不下）")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(config, 'loader_prefetch_factor'):
    config.loader_prefetch_factor = 4   # 降低可减少 CPU 侧缓存，缓解 OOM
# 加载训练样本
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=None,
    eval_split_size=config.eval_split_size,
)
print(f"加载样本: train={len(train_samples)}, eval={len(eval_samples)}")

# 初始化模型
model = setup_model(config, train_samples + eval_samples)
# 加载预训练权重
if not is_continue:
    model.load_checkpoint(config, str(model_path), eval=False)
    print("已加载预训练权重:", model_path)
else:
    print("续训：不加载预训练权重，Trainer 将从 run 目录恢复 checkpoint")

# ===== 关键：手动用DataParallel包装模型实现多卡并行 =====

model = model.cuda()
print("仅使用单GPU训练")

# ===== 修复TrainerArgs：移除所有低版本不支持的参数 =====
train_args = TrainerArgs(
    restore_path="",
    continue_path=str(CONTINUE_RUN_DIR) if is_continue else "",
    small_run=DEBUG_SMALL_RUN,
    gpu=0 if gpu else None,  # 低版本只需要指定主GPU编号
    # 移除 distributed/num_gpus/use_amp 等无效参数
)

# 初始化Trainer
trainer = Trainer(
    train_args,
    model.config if hasattr(model, 'config') else config,  # 适配DataParallel包装后的模型
    config.output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    parse_command_line_args=False,
)

# 续训时强制覆盖关键配置
# if is_continue:
#     trainer.config.epochs = DEBUG_EPOCHS
#     trainer.config.batch_size = DEBUG_BATCH_SIZE
#     trainer.config.audio.stats_path = str(MODEL_DIR / "scale_stats.npy")

# 启动训练
print(f"开始训练（{NUM_GPUS}块GPU，单卡Batch: {DEBUG_BATCH_SIZE}，总Batch: {DEBUG_BATCH_SIZE*NUM_GPUS}）")
try:
    trainer.fit()
    print("训练完成 ✅")
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    if "out of memory" in str(e).lower() or "CUDA" in str(e):
        print("\n[显存不足] 请将 DEBUG_BATCH_SIZE 改小（如 32 或 48）后重试。")
    raise
except NotImplementedError as e:
    if "optimize()" in str(e):
        print("\n[提示] 若此前报 CUDA OOM，请先减小 DEBUG_BATCH_SIZE 再运行。")
    raise