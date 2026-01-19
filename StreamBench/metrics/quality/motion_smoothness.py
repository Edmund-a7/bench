"""运动平滑度：使用 RAFT 计算光流平滑度"""
from models.raft_flow import RAFTFlow
from utils.video_io import load_segments
from utils.aggregation import mean_aggregation
from tqdm import tqdm
import torch


def compute_motion_smoothness(eval_data, device, config=None, path_config=None, **kwargs):
    print("Loading RAFT model for Motion Smoothness...")
    raft = RAFTFlow(device=device, config=config, path_config=path_config)
    
    results = {}
    for sample in tqdm(eval_data, desc="Motion Smoothness"):
        segment_paths = sample['segment_paths']
        segments = load_segments(segment_paths)
        
        scores = []
        for seg_frames in segments:
            if len(seg_frames) < 3:
                scores.append(1.0)
                continue
            
            indices = list(range(0, len(seg_frames)-1, max(1, len(seg_frames)//5)))
            motion_scores = []
            
            last_flow = None
            for i in indices:
                flow = raft.compute_flow(seg_frames[i], seg_frames[i+1])
                if last_flow is not None:
                    flow_diff = torch.norm(flow - last_flow, dim=0).mean().item()
                    score = 1.0 / (1.0 + flow_diff)
                    motion_scores.append(score)
                last_flow = flow
            
            if motion_scores:
                scores.append(sum(motion_scores) / len(motion_scores))
            else:
                scores.append(1.0)
                
        results[sample['sample_id']] = mean_aggregation(scores)
    return results
