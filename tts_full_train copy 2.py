"""
TTS 全流程：支持「仅加载检查」与「真正训练」两种模式。
真正训练使用 Coqui 官方 Trainer + 你的 train.csv/val.csv 与 models1 预训练权重。
"""
import logging
import os
import sys
import pandas as pd
from pathlib import Path

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
NUM_GPUS = 3  # 必须显式设为3
GPU_IDS = [0, 1, 2]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 限定使用3块卡

# 分布式训练配置
config.distributed_backend = "dp"  # 数据并行
config.num_gpus = NUM_GPUS
config.gpu_ids = GPU_IDS
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
    DEBUG_BATCH_SIZE = 24
    DEBUG_SAVE_STEP = 500
    DEBUG_PRINT_STEP = 25
    DEBUG_EVAL_SPLIT_SIZE = 0.05

OUTPUT_DIR = ROOT / "tts_train_output"
META_TRAIN = ROOT / "meta_train_coqui.txt"
META_VAL = ROOT / "meta_val_coqui.txt"
gpu = True

# 断点续训：设为某次 run 目录则从该目录最新 checkpoint 继续训练，设为 None 则从头训
CONTINUE_RUN_DIR = ROOT / "tts_train_output" / "mandarin_dca_attn_gst_dcc-February-26-2026_11+03AM-0000000"
if CONTINUE_RUN_DIR is not None:
    CONTINUE_RUN_DIR = Path(CONTINUE_RUN_DIR).resolve()

# ----------------------------
# 0. 运行前先执行 tts_process.py，生成 segments、train.csv、val.csv
# ----------------------------
from importlib.util import spec_from_file_location, module_from_spec
_tts_process_spec = spec_from_file_location("tts_process", ROOT / "tts_process.py")
# _tts_process = module_from_spec(_tts_process_spec)
# _tts_process_spec.loader.exec_module(_tts_process)
# _tts_process.main()

# ----------------------------
# 1. 检查并读取 train.csv / val.csv
# ----------------------------
train_csv = ROOT / "train.csv"
val_csv = ROOT / "val.csv"
if not train_csv.exists() or not val_csv.exists():
    print("未找到 train.csv 或 val.csv（tts_process 可能未产生有效样本）")
    sys.exit(1)

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

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
# 3. 真正训练：加载 config、覆盖数据集与输出、加载样本、建模型、加载预训练或续训、Trainer.fit
# ----------------------------
from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model
from trainer import Trainer, TrainerArgs
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger

setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())

is_continue = CONTINUE_RUN_DIR is not None
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
# 覆盖为当前项目路径与数据
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
# 调试/正式 训练超参
config.epochs = DEBUG_EPOCHS
config.batch_size = DEBUG_BATCH_SIZE
config.save_step = DEBUG_SAVE_STEP
config.print_step = DEBUG_PRINT_STEP
config.eval_split_size = DEBUG_EVAL_SPLIT_SIZE
config.run_eval = True
config.print_eval = True
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=None,
    eval_split_size=config.eval_split_size,
)
print(f"加载样本: train={len(train_samples)}, eval={len(eval_samples)}")

model = setup_model(config, train_samples + eval_samples)
if not is_continue:
    model.load_checkpoint(config, str(model_path), eval=False)
    print("已加载预训练权重:", model_path)
else:
    print("续训：不加载预训练权重，Trainer 将从 run 目录恢复 checkpoint")

train_args = TrainerArgs(
    restore_path="",
    continue_path=str(CONTINUE_RUN_DIR) if is_continue else "",
    small_run=DEBUG_SMALL_RUN,
    gpu=0 if gpu else None,
)
trainer = Trainer(
    train_args,
    model.config,
    config.output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    parse_command_line_args=False,
)
# 续训时 init_training 会从 run 目录重新 load_json 覆盖 config，这里把关键项改回当前脚本设置
if is_continue:
    trainer.config.epochs = DEBUG_EPOCHS
    trainer.config.audio.stats_path = str(MODEL_DIR / "scale_stats.npy")
trainer.fit()
print("训练完成 ✅")
