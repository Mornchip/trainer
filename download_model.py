#!/usr/bin/env python3
"""
手动下载 FastSpeech2 模型 - 使用 Coqui 官方 API
"""
import os
import sys
import shutil

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 创建目录
save_dir = "models_fastspeech2"
os.makedirs(save_dir, exist_ok=True)

print("📥 正在下载 FastSpeech2 模型...")
print("   使用国内镜像: https://hf-mirror.com")

try:
    from TTS.utils.manage import ModelManager
    
    manager = ModelManager()
    
    # 下载 FastSpeech2
    print("\n1️⃣ 下载 FastSpeech2...")
    model_path, config_path, items = manager.download_model("tts_models/zh-CN/baker/fastspeech2")
    
    print(f"   模型路径: {model_path}")
    print(f"   配置路径: {config_path}")
    
    # 复制到项目目录
    shutil.copy(model_path, os.path.join(save_dir, "model_file.pth"))
    shutil.copy(config_path, os.path.join(save_dir, "config.json"))
    
    # 尝试复制 scale_stats
    model_dir = os.path.dirname(model_path)
    stats_path = os.path.join(model_dir, "scale_stats.npy")
    if os.path.exists(stats_path):
        shutil.copy(stats_path, os.path.join(save_dir, "scale_stats.npy"))
        print("   ✅ scale_stats.npy 已复制")
    
    print("\n2️⃣ 下载声码器 (HiFi-GAN)...")
    vocoder_path, vocoder_config, _ = manager.download_model("vocoder_models/universal/libri-tts/hifigan_v2")
    
    shutil.copy(vocoder_path, os.path.join(save_dir, "vocoder_model.pth"))
    shutil.copy(vocoder_config, os.path.join(save_dir, "vocoder_config.json"))
    
    print("\n✅ 所有文件下载完成!")
    print(f"📁 保存位置: {save_dir}/")
    print("\n文件列表:")
    for f in os.listdir(save_dir):
        size = os.path.getsize(os.path.join(save_dir, f)) / 1024 / 1024
        print(f"   - {f} ({size:.1f} MB)")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n尝试备用方案...")
    
    # 备用：直接用 huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        
        print("\n使用 huggingface_hub 下载...")
        
        # FastSpeech2
        model_file = hf_hub_download(
            repo_id="coqui/TTS",
            filename="tts_models--zh-CN--baker--fastspeech2/model_file.pth",
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ 模型: {model_file}")
        
    except Exception as e2:
        print(f"❌ 备用方案也失败: {e2}")
        print("\n请检查:")
        print("1. 网络连接是否正常")
        print("2. 是否可以访问 https://hf-mirror.com")
        print("3. 尝试直接访问 HuggingFace: https://huggingface.co/coqui")
        sys.exit(1)
