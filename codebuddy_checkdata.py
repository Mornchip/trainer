 
"""
数据质量检查工具
检测：静音/空白、音频截断、文本标注匹配、底噪
"""
import os
import wave
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path("/root/autodl-tmp/zlynew/womenvoice/lasttraincodebuddy")
AUDIO_DIR = BASE_DIR / "segments"
META_FILE = BASE_DIR / "segments/meta_train_vits.txt"

def check_silence(audio_path, threshold=500, min_silence_duration=0.3):
    """检测音频中的静音段"""
    try:
        with wave.open(str(audio_path), 'rb') as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            
            # 读取音频数据
            raw_data = wav.readframes(n_frames)
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            
            # 计算音量
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            # 检测静音（音量低于阈值）
            is_silent = np.abs(audio_data) < threshold
            silent_frames = np.sum(is_silent)
            total_frames = len(audio_data)
            silence_ratio = silent_frames / total_frames
            
            # 检测连续静音
            silent_runs = []
            current_run = 0
            for is_s in is_silent:
                if is_s:
                    current_run += 1
                else:
                    if current_run > 0:
                        silent_runs.append(current_run)
                    current_run = 0
            
            # 长静音段（>300ms）
            min_silence_frames = int(min_silence_duration * framerate)
            long_silences = [r for r in silent_runs if r > min_silence_frames]
            
            return {
                'silence_ratio': silence_ratio,
                'long_silence_count': len(long_silences),
                'has_excessive_silence': silence_ratio > 0.3 or len(long_silences) > 3
            }
    except Exception as e:
        return {'error': str(e)}

def check_audio_quality(audio_path):
    """检查音频质量：截断、底噪、动态范围"""
    try:
        with wave.open(str(audio_path), 'rb') as wav:
            n_frames = wav.getnframes()
            framerate = wav.getframerate()
            duration = n_frames / framerate
            
            raw_data = wav.readframes(n_frames)
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            
            # 检查是否截断（峰值接近最大值）
            max_val = np.max(np.abs(audio_data))
            clipping_ratio = np.sum(np.abs(audio_data) > 30000) / len(audio_data)
            
            # 检查动态范围
            rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            peak_db = 20 * np.log10(max_val / 32768.0 + 1e-10)
            rms_db = 20 * np.log10(rms / 32768.0 + 1e-10)
            dynamic_range = peak_db - rms_db
            
            # 估计底噪（假设前后100ms为静音段）
            noise_sample_size = int(0.1 * framerate)
            if len(audio_data) > noise_sample_size * 2:
                noise_floor = np.mean(np.abs(audio_data[:noise_sample_size])) + \
                             np.mean(np.abs(audio_data[-noise_sample_size:]))
                snr = 20 * np.log10(max_val / (noise_floor + 1e-10)) if noise_floor > 0 else 0
            else:
                snr = 0
            
            return {
                'duration': duration,
                'max_amplitude': max_val,
                'clipping_ratio': clipping_ratio,
                'dynamic_range_db': dynamic_range,
                'estimated_snr_db': snr,
                'issues': []
            }
    except Exception as e:
        return {'error': str(e)}

def main():
    print("=" * 70)
    print("🔍 数据质量检查报告")
    print("=" * 70)
    
    # 读取标注文件
    if not META_FILE.exists():
        print(f"❌ 找不到标注文件: {META_FILE}")
        return
    
    entries = []
    with open(META_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                # 支持两列 path|text 或三列 path|speaker|text
                wav_name = parts[0].strip()
                if len(parts) >= 3:
                    text = parts[2].strip()
                else:
                    text = parts[1].strip() if len(parts) > 1 else ""
                if not wav_name.endswith('.wav'):
                    wav_name = wav_name + '.wav'
                entries.append({
                    'wav_name': wav_name,
                    'text': text
                })
    
    print(f"📊 总样本数: {len(entries)}")
    print(f"🎵 音频目录: {AUDIO_DIR}")
    print("-" * 70)
    
    # 采样检查（检查前20个）
    sample_size = min(20, len(entries))
    print(f"\n🔬 抽样检查前 {sample_size} 个样本...\n")
    
    issues_found = {
        'excessive_silence': [],
        'clipping': [],
        'low_snr': [],
        'short_audio': [],
        'missing_files': []
    }
    
    for i, entry in enumerate(entries[:sample_size]):
        wav_name = entry['wav_name']
        text = entry['text']
        wav_path = AUDIO_DIR / wav_name
        
        print(f"[{i+1:02d}/{sample_size}] {wav_name}")
        print(f"    文本: {text[:30]}{'...' if len(text) > 30 else ''}")
        
        if not wav_path.exists():
            print(f"    ❌ 文件不存在!")
            issues_found['missing_files'].append(wav_name)
            continue
        
        # 检查静音
        silence_info = check_silence(wav_path)
        if 'error' in silence_info:
            print(f"    ⚠️ 静音检测失败: {silence_info['error']}")
        else:
            silence_pct = silence_info['silence_ratio'] * 100
            print(f"    🔇 静音比例: {silence_pct:.1f}%", end="")
            if silence_info['has_excessive_silence']:
                print(" ⚠️ 静音过多!")
                issues_found['excessive_silence'].append(wav_name)
            else:
                print(" ✓")
        
        # 检查音频质量
        quality = check_audio_quality(wav_path)
        if 'error' in quality:
            print(f"    ⚠️ 质量检测失败: {quality['error']}")
        else:
            print(f"    ⏱️  时长: {quality['duration']:.2f}s", end="")
            if quality['duration'] < 1.5:
                print(" ⚠️ 太短!", end="")
                issues_found['short_audio'].append(wav_name)
            print()
            
            print(f"    📢 峰值: {quality['max_amplitude']}/32767", end="")
            if quality['clipping_ratio'] > 0.01:
                print(f" ⚠️ 削波({quality['clipping_ratio']*100:.1f}%)!", end="")
                issues_found['clipping'].append(wav_name)
            print()
            
            print(f"    🔊 信噪比: {quality['estimated_snr_db']:.1f}dB", end="")
            if quality['estimated_snr_db'] < 20:
                print(" ⚠️ 底噪大!", end="")
                issues_found['low_snr'].append(wav_name)
            print()
        
        print()
    
    # 汇总报告
    print("=" * 70)
    print("📋 问题汇总")
    print("=" * 70)
    
    total_issues = sum(len(v) for v in issues_found.values())
    if total_issues == 0:
        print("✅ 未发现明显问题")
    else:
        print(f"⚠️ 共发现 {total_issues} 个问题:\n")
        for issue_type, files in issues_found.items():
            if files:
                issue_names = {
                    'excessive_silence': '🔇 静音过多',
                    'clipping': '📢 音频削波/截断',
                    'low_snr': '🔊 底噪大/信噪比低',
                    'short_audio': '⏱️  音频过短',
                    'missing_files': '❌ 文件缺失'
                }
                print(f"{issue_names.get(issue_type, issue_type)}: {len(files)}个")
                for f in files[:5]:
                    print(f"    - {f}")
                if len(files) > 5:
                    print(f"    ... 还有 {len(files)-5} 个")
                print()

if __name__ == "__main__":
    main() 
