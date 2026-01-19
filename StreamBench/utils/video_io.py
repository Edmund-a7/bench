"""视频读写和加载工具"""
import os
import subprocess
from vbench.utils import load_video  # 返回 list[PIL.Image]


def load_segments(segment_paths):
    """加载视频分段为帧列表"""
    # 确保返回 PIL Image 列表
    return [load_video(p) for p in segment_paths]


def merge_segments(segment_paths, output_path):
    """合并分段视频为完整视频"""
    list_file = output_path.replace('.mp4', '_list.txt')
    with open(list_file, 'w') as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    
    subprocess.run([
        'ffmpeg', '-f', 'concat', '-safe', '0',
        '-i', list_file, '-c', 'copy', output_path, '-y'
    ], capture_output=True)
    
    if os.path.exists(list_file):
        os.remove(list_file)
    return output_path
