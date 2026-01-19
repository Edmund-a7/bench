"""时序闪烁：计算相邻帧的像素差异"""
import numpy as np
from utils.video_io import load_segments
from utils.aggregation import mean_aggregation
from tqdm import tqdm


def compute_temporal_flickering(eval_data, device, config=None, path_config=None, **kwargs):
    # 不需要加载模型，纯 CV 计算
    results = {}
    for sample in tqdm(eval_data, desc="Temporal Flickering"):
        segment_paths = sample['segment_paths']
        segments = load_segments(segment_paths)
        
        scores = []
        for seg_frames in segments:
            if len(seg_frames) < 2:
                continue
            
            flicker_scores = []
            for i in range(len(seg_frames) - 1):
                img1 = np.array(seg_frames[i]).astype(np.float32)
                img2 = np.array(seg_frames[i+1]).astype(np.float32)
                
                diff = np.abs(img1 - img2)
                score = 1.0 - (np.mean(diff) / 255.0)
                flicker_scores.append(score)
            
            if flicker_scores:
                scores.append(sum(flicker_scores) / len(flicker_scores))
        
        results[sample['sample_id']] = mean_aggregation(scores)
    return results
