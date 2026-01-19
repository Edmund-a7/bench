"""边界平滑一致性：相邻段边界帧光流"""
from models.raft_flow import RAFTFlow
from utils.video_io import load_segments
from utils.aggregation import mean_aggregation
from tqdm import tqdm


def compute_boundary_smoothness(eval_data, device, config=None, path_config=None, **kwargs):
    print("Loading RAFT model...")
    raft = RAFTFlow(device=device, config=config, path_config=path_config)
    
    results = {}
    for sample in tqdm(eval_data, desc="Boundary Smoothness"):
        segment_paths = sample['segment_paths']
        segments = load_segments(segment_paths)
        
        scores = []
        for i in range(len(segments) - 1):
            last_frame = segments[i][-1]
            first_frame = segments[i+1][0]
            
            flow = raft.compute_flow(last_frame, first_frame)
            flow_mag = flow.norm(dim=0).mean().item()
            scores.append(1.0 / (1.0 + flow_mag))
        
        results[sample['sample_id']] = mean_aggregation(scores)
    return results
