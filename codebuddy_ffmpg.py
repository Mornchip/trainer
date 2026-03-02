#!/usr/bin/env python3
"""
FFmpeg 音频专业处理工具 (Python版)
功能：去静音、音量标准化、统一格式、批量处理
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


# ==================== 配置 ====================
class Config:
    """处理配置"""
    # 路径配置
    BASE_DIR = Path("/root/autodl-tmp/zlynew/womenvoice/lasttraincodebuddy")
    INPUT_DIR = BASE_DIR / "segments"
    BACKUP_DIR = BASE_DIR / f"segments_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_DIR = BASE_DIR / "segments_processed"
    
    # FFmpeg 参数
    SILENCE_THRESHOLD = "-45dB"      # 静音检测阈值
    MIN_SILENCE_DURATION = 0.05      # 最小静音时长（秒）
    TARGET_LOUDNESS = -16            # 目标响度（dB）
    LOUDNESS_RANGE = 11              # 响度范围（dB）
    TRUE_PEAK = -1.5                 # 峰值限制（dB）
    TARGET_SAMPLE_RATE = 22050       # 目标采样率
    
    # 处理参数
    MAX_WORKERS = 4                  # 并行处理数


# ==================== 工具函数 ====================

def run_command(cmd, capture_output=True):
    """运行shell命令"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timeout"
    except Exception as e:
        return False, "", str(e)


def check_ffmpeg():
    """检查FFmpeg是否安装"""
    success, stdout, stderr = run_command("ffmpeg -version")
    if success:
        version = stdout.split('\n')[0]
        print(f"✅ FFmpeg已安装: {version}")
        return True
    else:
        print("❌ 错误：未找到FFmpeg")
        print("正在安装...")
        os.system("apt-get update && apt-get install -y ffmpeg")
        return check_ffmpeg()


def get_audio_info(filepath):
    """获取音频文件信息"""
    cmd = f'ffprobe -v error -show_entries format=duration -show_entries stream=sample_rate,channels -of json "{filepath}"'
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        return None
    
    try:
        data = json.loads(stdout)
        info = {
            'duration': float(data['format']['duration']),
            'sample_rate': int(data['streams'][0]['sample_rate']),
            'channels': int(data['streams'][0]['channels'])
        }
        return info
    except:
        return None


def detect_silence(filepath):
    """检测音频中的静音段"""
    cmd = f'ffmpeg -i "{filepath}" -af "silencedetect=noise={Config.SILENCE_THRESHOLD}:d={Config.MIN_SILENCE_DURATION}" -f null - 2>&1'
    success, stdout, stderr = run_command(cmd, capture_output=False)
    
    # 从stderr中提取静音信息
    output = stdout + stderr
    silence_lines = [line for line in output.split('\n') if 'silence_' in line]
    
    silence_periods = []
    silence_start = None
    
    for line in silence_lines:
        if 'silence_start:' in line:
            try:
                silence_start = float(line.split('silence_start:')[1].strip())
            except:
                pass
        elif 'silence_end:' in line and silence_start is not None:
            try:
                silence_end = float(line.split('silence_end:')[1].split('|')[0].strip())
                silence_periods.append((silence_start, silence_end))
                silence_start = None
            except:
                pass
    
    return silence_periods


def process_audio(input_file, output_file):
    """
    处理单个音频文件
    功能：去静音 + 音量标准化 + 统一格式
    """
    # 构建FFmpeg命令
    filter_complex = (
        f"silenceremove="
        f"start_periods=1:"
        f"start_duration={Config.MIN_SILENCE_DURATION}:"
        f"start_threshold={Config.SILENCE_THRESHOLD}:"
        f"stop_periods=1:"
        f"stop_duration={Config.MIN_SILENCE_DURATION}:"
        f"stop_threshold={Config.SILENCE_THRESHOLD},"
        f"loudnorm="
        f"I={Config.TARGET_LOUDNESS}:"
        f"LRA={Config.LOUDNESS_RANGE}:"
        f"TP={Config.TRUE_PEAK}"
    )
    
    cmd = (
        f'ffmpeg -y -i "{input_file}" '
        f'-af "{filter_complex}" '
        f'-ar {Config.TARGET_SAMPLE_RATE} '
        f'-ac 1 '
        f'-c:a pcm_s16le '
        f'"{output_file}"'
    )
    
    success, stdout, stderr = run_command(cmd)
    return success


def process_single_file(args):
    """处理单个文件的包装函数（用于多进程）"""
    input_file, output_file, index, total = args
    
    filename = input_file.name
    
    # 获取处理前信息
    before_info = get_audio_info(input_file)
    if before_info is None:
        return {
            'status': 'error',
            'file': filename,
            'error': '无法读取音频信息'
        }
    
    # 处理音频
    success = process_audio(input_file, output_file)
    
    if not success:
        return {
            'status': 'error',
            'file': filename,
            'error': 'FFmpeg处理失败'
        }
    
    # 获取处理后信息
    after_info = get_audio_info(output_file)
    if after_info is None:
        return {
            'status': 'error',
            'file': filename,
            'error': '无法读取处理后音频'
        }
    
    return {
        'status': 'success',
        'file': filename,
        'before_duration': before_info['duration'],
        'after_duration': after_info['duration'],
        'before_size': input_file.stat().st_size,
        'after_size': output_file.stat().st_size
    }


class AudioProcessor:
    """音频处理器主类"""
    
    def __init__(self):
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_before_duration': 0,
            'total_after_duration': 0
        }
    
    def setup(self):
        """初始化设置"""
        print("=" * 70)
        print("🎵 FFmpeg 音频专业处理工具 (Python版)")
        print("=" * 70)
        print()
        
        # 检查FFmpeg
        if not check_ffmpeg():
            sys.exit(1)
        
        # 创建目录
        print("📂 创建目录...")
        Config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"   备份: {Config.BACKUP_DIR}")
        print(f"   输出: {Config.OUTPUT_DIR}")
        print()
    
    def backup_files(self):
        """备份原始文件"""
        print("💾 备份原始文件...")
        
        wav_files = list(Config.INPUT_DIR.glob("*.wav"))
        self.stats['total'] = len(wav_files)
        
        for i, f in enumerate(wav_files, 1):
            shutil.copy2(f, Config.BACKUP_DIR / f.name)
            print(f"\r   已备份: {i}/{len(wav_files)}", end="")
        
        print()
        print(f"✅ 备份完成: {len(wav_files)} 个文件")
        print()
    
    def process_all(self, max_workers=4):
        """批量处理所有文件"""
        print("🔧 开始处理音频文件...")
        print("=" * 70)
        
        wav_files = list(Config.INPUT_DIR.glob("*.wav"))
        
        # 准备任务列表
        tasks = []
        for i, f in enumerate(wav_files):
            output_file = Config.OUTPUT_DIR / f.name
            tasks.append((f, output_file, i + 1, len(wav_files)))
        
        # 并行处理
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, task): task for task in tasks}
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                # 实时显示进度
                if result['status'] == 'success':
                    self.stats['success'] += 1
                    duration_diff = result['before_duration'] - result['after_duration']
                    print(f"\r✅ {result['file'][:40]:<40} "
                          f"({result['before_duration']:.2f}s → "
                          f"{result['after_duration']:.2f}s, "
                          f"-{duration_diff:.2f}s)")
                else:
                    self.stats['failed'] += 1
                    print(f"\r❌ {result['file'][:40]:<40} "
                          f"错误: {result.get('error', 'unknown')}")
        
        return results
    
    def generate_report(self, results):
        """生成处理报告"""
        print()
        print("=" * 70)
        print("📊 处理报告")
        print("=" * 70)
        
        # 统计
        success_results = [r for r in results if r['status'] == 'success']
        
        total_before_duration = sum(r['before_duration'] for r in success_results)
        total_after_duration = sum(r['after_duration'] for r in success_results)
        total_silence_removed = total_before_duration - total_after_duration
        
        total_before_size = sum(r['before_size'] for r in success_results)
        total_after_size = sum(r['after_size'] for r in success_results)
        
        print(f"总文件数: {self.stats['total']}")
        print(f"✅ 成功: {self.stats['success']}")
        print(f"❌ 失败: {self.stats['failed']}")
        print()
        print("时长变化:")
        print(f"   处理前: {total_before_duration:.2f} 秒")
        print(f"   处理后: {total_after_duration:.2f} 秒")
        print(f"   去除静音: {total_silence_removed:.2f} 秒 ({total_silence_removed/total_before_duration*100:.1f}%)")
        print()
        print("大小变化:")
        print(f"   处理前: {total_before_size/1024/1024:.2f} MB")
        print(f"   处理后: {total_after_size/1024/1024:.2f} MB")
        print(f"   减少: {(total_before_size-total_after_size)/1024/1024:.2f} MB")
        print()
        print("=" * 70)
        print("📁 输出位置:")
        print(f"   备份: {Config.BACKUP_DIR}")
        print(f"   处理后: {Config.OUTPUT_DIR}")
        print("=" * 70)
        print()
        print("💡 下一步:")
        print("   1. 试听对比原始和处理后的音频")
        print("   2. 如果效果满意，用处理后的数据替换原始数据:")
        print(f"      mv {Config.INPUT_DIR} {Config.INPUT_DIR}_old")
        print(f"      mv {Config.OUTPUT_DIR} {Config.INPUT_DIR}")
        print("   3. 重新训练模型")
        print()
    
    def demo_process_two(self):
        """演示处理2个文件"""
        print("🎵 演示模式：处理2个文件")
        print("=" * 70)
        
        wav_files = list(Config.INPUT_DIR.glob("*.wav"))[:2]
        
        if len(wav_files) < 2:
            print("❌ 错误：音频文件不足2个")
            return
        
        for i, f in enumerate(wav_files, 1):
            print(f"\n文件 {i}: {f.name}")
            print("-" * 50)
            
            # 备份
            backup_path = Config.BACKUP_DIR / f.name
            shutil.copy2(f, backup_path)
            print(f"✅ 已备份到: {backup_path}")
            
            # 处理前信息
            before_info = get_audio_info(f)
            if before_info:
                print(f"📊 处理前: 时长={before_info['duration']:.2f}s, "
                      f"采样率={before_info['sample_rate']}Hz, "
                      f"声道={before_info['channels']}")
            
            # 检测静音
            silence_periods = detect_silence(f)
            if silence_periods:
                total_silence = sum(end - start for start, end in silence_periods)
                print(f"🔇 检测到 {len(silence_periods)} 段静音, "
                      f"总计 {total_silence:.2f}s")
            
            # 处理
            output_path = Config.OUTPUT_DIR / f.name
            success = process_audio(f, output_path)
            
            if success:
                after_info = get_audio_info(output_path)
                if after_info:
                    print(f"✅ 处理后: 时长={after_info['duration']:.2f}s")
                    duration_diff = before_info['duration'] - after_info['duration']
                    print(f"   去除静音: {duration_diff:.2f}s")
            else:
                print("❌ 处理失败")
        
        print()
        print("=" * 70)
        print("📁 文件位置:")
        print(f"   备份: {Config.BACKUP_DIR}")
        print(f"   处理后: {Config.OUTPUT_DIR}")
        print()
        print("🎧 试听命令:")
        for f in wav_files:
            print(f"   ffplay {Config.BACKUP_DIR / f.name}  # 原始")
            print(f"   ffplay {Config.OUTPUT_DIR / f.name}  # 处理后")
            print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FFmpeg音频处理工具')
    parser.add_argument('--demo', action='store_true', help='演示模式：只处理2个文件')
    parser.add_argument('--workers', type=int, default=4, help='并行处理数 (默认: 4)')
    args = parser.parse_args()
    
    processor = AudioProcessor()
    processor.setup()
    
    if args.demo:
        # 演示模式
        processor.demo_process_two()
    else:
        # 完整处理
        processor.backup_files()
        results = processor.process_all(max_workers=args.workers)
        processor.generate_report(results)


if __name__ == "__main__":
    main()
