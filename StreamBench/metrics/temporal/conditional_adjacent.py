"""条件相邻一致性：MLLM 判断后计算相邻段相似度"""
from models.dino_encoder import DINOEncoder
from utils.mllm_utils import mllm_judge_scene_change
from utils.video_io import load_segments
from utils.aggregation import mean_aggregation
from tqdm import tqdm
import torch.nn.functional as F


def compute_conditional_adjacent(eval_data, device, config=None, path_config=None, **kwargs):
    print("Loading DINO model for Adjacent Consistency...")
    dino = DINOEncoder(device=device, config=config, path_config=path_config)
    
    results = {}
    for sample in tqdm(eval_data, desc="Conditional Adjacent"):
        prompts = sample['prompts']
        segments = load_segments(sample['segment_paths'])
        
        scores = []
        for i in range(len(prompts) - 1):
            should_keep = mllm_judge_scene_change(prompts[i], prompts[i+1])
            
            if should_keep:
                feat_i = dino.encode_segment(segments[i])
                feat_i1 = dino.encode_segment(segments[i+1])
                sim = F.cosine_similarity(feat_i, feat_i1, dim=-1).item()
                scores.append(sim)
        
        results[sample['sample_id']] = mean_aggregation(scores) if scores else 1.0
    return results
