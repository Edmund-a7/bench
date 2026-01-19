"""数据预处理：解析 prompts + 分割视频"""
import cv2
import os
import json


def split_video_to_segments(video_path, output_dir, segment_duration=10):
    """将完整视频按时长分割为多段"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback
    w, h = int(cap.get(3)), int(cap.get(4))
    frames_per_seg = int(fps * segment_duration)
    
    paths, idx = [], 0
    while True:
        frames = []
        for _ in range(frames_per_seg):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            break
        
        path = os.path.join(output_dir, f"seg_{idx}.mp4")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        paths.append(path)
        idx += 1
    
    cap.release()
    return paths


def parse_prompts_file(path):
    """解析 prompts 文件 (jsonl 或 json)"""
    samples = []
    if path.endswith('.jsonl'):
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            samples = data
        else:
            samples = [data]
    return samples


def prepare_evaluation_data(video_dir, prompts_file, output_dir, segment_duration=10):
    """准备评估数据"""
    os.makedirs(output_dir, exist_ok=True)
    samples = parse_prompts_file(prompts_file)
    eval_data = []
    
    for idx, sample in enumerate(samples):
        video_path = os.path.join(video_dir, f"sample_{idx}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found, skipping")
            continue
        
        seg_dir = os.path.join(output_dir, f"sample_{idx}")
        segment_paths = split_video_to_segments(video_path, seg_dir, segment_duration)
        
        eval_data.append({
            'sample_id': idx,
            'prompts': sample['prompts'],
            'segment_paths': segment_paths
        })
    
    output_json = os.path.join(output_dir, 'eval_data.json')
    with open(output_json, 'w') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(eval_data)} samples -> {output_json}")
    return eval_data
