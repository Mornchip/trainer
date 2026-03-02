#!/usr/bin/env python3
"""
VITS 训练脚本 - RTX 5090 优化版
无需预训练模型，直接从头训练
支持短音频（0.36秒+），适合呻吟/端音数据
"""
import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path

# 添加音频对齐修复模块导入
try:
    from audio_align_repair import run_repair
except ImportError:
    run_repair = None
    print("⚠️  audio_align_repair 模块未找到，跳过对齐修复步骤")

# ==================== 5090 极致优化配置 ====================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # 自动寻找最快算法

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "25"
os.environ["MKL_NUM_THREADS"] = "25"
os.environ["NUMEXPR_NUM_THREADS"] = "25"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "vits_train_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 检查 GPU
gpu = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if gpu else "CPU"
print(f"🚀 设备: {device_name}")
if gpu:
    print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ==================== VITS 训练参数 - 5090极致优化 ====================
EPOCHS = 1000
BATCH_SIZE = 64         # 5090 32GB 显存，VITS可以上到128-160
NUM_WORKERS = 8         # CPU核心数，提高数据加载
PREFETCH_FACTOR = 8      # 预取因子，减少CPU等待
GRADIENT_ACCUMULATION = 1  # 如需更大batch可改为2-4
SAVE_STEP = 500
PRINT_STEP = 50

def prepare_data(use_repaired=False):
    """准备数据"""
    # 优先使用修复后的数据
    if use_repaired:
        repaired_train = ROOT / "repaired_data" / "train.csv"
        repaired_val = ROOT / "repaired_data" / "val.csv"
        if repaired_train.exists() and repaired_val.exists():
            train_csv = repaired_train
            val_csv = repaired_val
            print("📂 使用对齐修复后的数据")
        else:
            train_csv = ROOT / "train.csv"
            val_csv = ROOT / "val.csv"
    else:
        train_csv = ROOT / "train.csv"
        val_csv = ROOT / "val.csv"
    
    if not train_csv.exists():
        print(f"❌ 错误: 找不到 {train_csv}")
        sys.exit(1)
    
    train_df = pd.read_csv(train_csv, dtype={"text": str, "audio_path": str})
    val_df = pd.read_csv(val_csv, dtype={"text": str, "audio_path": str})
    
    print(f"📊 数据: train={len(train_df)}, val={len(val_df)}")
    
    # 生成 Coqui 格式
    meta_train = ROOT / "meta_train_vits.txt"
    meta_val = ROOT / "meta_val_vits.txt"
    
    def to_meta(df, out_path):
        lines = ["audio_file|text|speaker_name"]
        for _, row in df.iterrows():
            audio_path = Path(row["audio_path"])
            # 相对于当前训练脚本所在目录存储路径，配合 Coqui `coqui` formatter
            # 这样无需拷贝音频文件，只要保证相对路径正确即可
            rel_path = os.path.relpath(audio_path, ROOT)
            text = str(row["text"]).replace("|", " ")
            lines.append(f"{rel_path}|{text}|female")
        out_path.write_text("\n".join(lines), encoding="utf-8")
    
    to_meta(train_df, meta_train)
    to_meta(val_df, meta_val)
    
    return str(meta_train), str(meta_val)

def train():
    """训练 VITS - 无需预训练模型"""
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.configs.shared_configs import CharactersConfig
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.datasets import load_tts_samples
    from trainer import Trainer, TrainerArgs
    
    # ========== 步骤0: 音频对齐修复 ==========
    if run_repair is not None:
        print("🔧 步骤0: 音频对齐修复...")
        
        # 检查是否已修复过（避免重复处理）
        repaired_train = ROOT / "repaired_data" / "train.csv"
        repaired_val = ROOT / "repaired_data" / "val.csv"
        
        if not repaired_train.exists():
            # 准备输入JSON（从现有train.csv生成）
            original_train = ROOT / "train.csv"
            if original_train.exists():
                train_df_original = pd.read_csv(original_train)
                
                # 转换为JSON格式
                input_data = []
                for _, row in train_df_original.iterrows():
                    input_data.append({
                        "audio_path": str(row["audio_path"]),
                        "text": str(row["text"])
                    })
                
                input_json = ROOT / "align_input.json"
                with open(input_json, 'w', encoding='utf-8') as f:
                    json.dump(input_data, f, ensure_ascii=False, indent=2)
                
                print(f"   准备修复 {len(input_data)} 条样本...")
                
                # 调用对齐修复
                try:
                    result = run_repair(
                        input_json=input_json,
                        output_dir=ROOT / "repaired_data",
                        whisper_model_size="base",
                        trim_silence=True,
                        report_excel=ROOT / "align_report.xlsx"
                    )
                    
                    print(f"✅ 对齐修复完成:")
                    print(f"   保留: {result.get('kept', 0)}条")
                    print(f"   ASR替换: {result.get('replaced', 0)}条") 
                    print(f"   需审核: {result.get('review', 0)}条")
                    if result.get('report_excel'):
                        print(f"   报告: {result['report_excel']}")
                except Exception as e:
                    print(f"⚠️  对齐修复失败: {e}")
                    print("   将使用原始数据继续训练")
        else:
            print("✅ 检测到已修复数据，跳过对齐步骤")
    # ==========================================
    
    # 准备数据（优先使用修复后的数据）
    meta_train, meta_val = prepare_data(use_repaired=True)
    
    print("🔧 创建 VITS 配置...")
    
    # 创建配置（从头开始，无需加载文件）
    config = VitsConfig()
    config.model = "vits"
    config.output_path = str(OUTPUT_DIR)
    config.run_name = "vits_adult_voice"
    
    # 训练参数
    config.epochs = EPOCHS
    config.batch_size = BATCH_SIZE
    config.eval_batch_size = 16
    config.save_step = SAVE_STEP
    config.print_step = PRINT_STEP
    
    # 5090 极致优化
    config.num_loader_workers = NUM_WORKERS
    config.pin_memory = True
    config.prefetch_factor = PREFETCH_FACTOR  # 关键：预取数据
    config.use_amp = True  # 混合精度
    config.gradient_accumulation_steps = GRADIENT_ACCUMULATION
    
    # VITS 关键：支持短音频！
    config.min_text_len = 2       # 最低 2 个字符
    config.max_text_len = 200
    config.min_audio_len = 8000   # 0.36秒 @22050Hz（关键！）
    config.max_audio_len = 180000 # 8秒

    # ===== 文本 & 字符集配置：改为中文拼音/音素流水线，避免中文字符全被丢弃 =====
    config.use_phonemes = True
    config.phoneme_cache_path = str(OUTPUT_DIR / "phoneme_cache")  # 必须设置，否则 DataLoader 报 TypeError
    config.phonemizer = "zh_cn_phonemizer"
    config.phoneme_language = "zh-cn"
    config.text_cleaner = "chinese_mandarin_cleaners"
    config.enable_eos_bos_chars = False
    config.add_blank = False
    # 音素缓存目录，避免 PhonemeDataset 里 cache_path 为 None
    phoneme_cache_dir = ROOT / "phoneme_cache_vits"
    phoneme_cache_dir.mkdir(parents=True, exist_ok=True)
    config.phoneme_cache_path = str(phoneme_cache_dir)
    config.characters = CharactersConfig(
        pad="_",
        eos="~",
        bos="^",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),.:;? ",
        punctuations="，。？！～：；*——-（）【】!'(),-.:;? ",
        phonemes="12345iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧʲ̃ɚ˞ɫ",
        is_unique=False,
        is_sorted=True,
    )
    
    # 数据集
    from TTS.config.shared_configs import BaseDatasetConfig
    config.datasets = [
        BaseDatasetConfig(
            # 使用 Coqui 自带的通用格式：audio_file|text|speaker_name
            formatter="coqui",
            dataset_name="adult_voice",
            path=str(ROOT),
            meta_file_train=Path(meta_train).name,
            meta_file_val=Path(meta_val).name,
        )
    ]
    
    # 音频配置
    config.audio.sample_rate = 22050
    config.audio.fft_size = 1024
    config.audio.win_length = 1024
    config.audio.hop_length = 256
    config.audio.num_mels = 80
    config.audio.mel_fmin = 0
    config.audio.mel_fmax = None
    
    # VITS 模型参数
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
    
    # 训练参数
    config.train = {
        "seg_len": 8192,
        "port": 5000,
        "c_mel": 45,
        "c_kl": 1.0,
        "c_commit": 1.0,
        "c_reconstruct": 1.0
    }
    
    # 优化器
    config.optimizer_params = {
        "betas": [0.8, 0.99],
        "eps": 1e-09,
        "weight_decay": 0.0
    }
    config.lr_scheduler = "ExponentialLR"
    config.lr_scheduler_params = {"gamma": 0.998875}
    
    # 加载数据
    print("📂 加载数据集...")
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_size=0.05,
    )
    print(f"✅ 加载完成: train={len(train_samples)}, eval={len(eval_samples)}")
    
    # 初始化音频处理器和 tokenizer
    print("🔧 初始化音频处理器...")
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # 初始化 VITS 模型（从头训练，无需预训练！）
    print("🏗️  初始化 VITS 模型...")
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    
    # 移动到 GPU
    if gpu:
        model = model.cuda()
        print("💨 启用 torch.compile 加速...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except:
            pass
    
    # 训练参数
    train_args = TrainerArgs(
        restore_path="",
        continue_path="",
        gpu=0 if gpu else None,
    )
    
    # 初始化训练器
    trainer = Trainer(
        train_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"audio_processor": ap},
        parse_command_line_args=False,
    )
    
    # 打印配置
    print("\n" + "="*60)
    print(f"🚀 VITS 训练配置 (RTX 5090 极致优化)")
    print(f"="*60)
    print(f"📊 训练样本: {len(train_samples)}")
    print(f"📊 验证样本: {len(eval_samples)}")
    print(f"🔄 训练轮数: {EPOCHS}")
    print(f"📦 Batch Size: {BATCH_SIZE} (预计显存占用 ~20-24GB)")
    print(f"🔄 Gradient Accumulation: {GRADIENT_ACCUMULATION}")
    print(f"👷 DataLoader Workers: {NUM_WORKERS}")
    print(f"📥 Prefetch Factor: {PREFETCH_FACTOR}")
    print(f"⏱️  最短音频: 0.36秒 (支持呻吟/短音频)")
    print(f"⚡ 混合精度: True")
    print(f"🔥 torch.compile: 已启用")
    print(f"🎯 预训练模型: 不需要（从头训练）")
    print(f"="*60 + "\n")
    
    # 开始训练
    print("🎯 开始训练 VITS...")
    trainer.fit()
    
    print("\n✅ 训练完成!")
    print(f"📁 模型保存: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
