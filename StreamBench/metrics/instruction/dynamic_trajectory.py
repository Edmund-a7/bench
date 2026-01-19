"""动态轨迹对齐：视频变化与文本变化的一致性"""
from models.languagebind_encoder import LanguageBindEncoder
from utils.aggregation import mean_aggregation
import torch.nn.functional as F
from tqdm import tqdm


def compute_dynamic_trajectory(eval_data, device, config=None, path_config=None, **kwargs):
    print("Loading LanguageBind model...")
    encoder = LanguageBindEncoder(device=device, config=config, path_config=path_config)
    
    results = {}
    for sample in tqdm(eval_data, desc="Dynamic Trajectory"):
        segment_paths = sample['segment_paths']
        prompts = sample['prompts']
        
        v_feats = []
        p_feats = []
        
        for path, prompt in zip(segment_paths, prompts):
            v_feats.append(encoder.encode_video(path))
            p_feats.append(encoder.encode_text(prompt))
            
        scores = []
        for i in range(len(v_feats) - 1):
            v_diff = v_feats[i+1] - v_feats[i]
            p_diff = p_feats[i+1] - p_feats[i]
            
            if v_diff.norm() < 0.1 or p_diff.norm() < 0.1:
                continue
            
            v_diff = F.normalize(v_diff, dim=-1)
            p_diff = F.normalize(p_diff, dim=-1)
            scores.append(F.cosine_similarity(v_diff, p_diff, dim=-1).item())
        
        results[sample['sample_id']] = mean_aggregation(scores) if scores else 0.0
    return results
