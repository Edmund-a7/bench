"""主体一致性：DINO 特征 + VDE 融合"""
import torch
import torch.nn.functional as F
from models.dino_encoder import DINOEncoder
from utils.video_io import load_segments
from utils.aggregation import vde_decay
from tqdm import tqdm


def compute_subject_consistency(eval_data, device, config=None, path_config=None, **kwargs):
    print("Loading DINO model...")
    dino = DINOEncoder(device=device, config=config, path_config=path_config)
    
    results = {}
    for sample in tqdm(eval_data, desc="Subject Consistency"):
        segment_paths = sample['segment_paths']
        segments = load_segments(segment_paths)
        segment_scores = []
        
        for seg_frames in segments:
            n = len(seg_frames)
            if n == 0:
                segment_scores.append(0.0)
                continue
                
            key_indices = sorted(list(set([i for i in [0, n//4, n//2, 3*n//4, n-1] if i < n])))
            
            feats = []
            for i in key_indices:
                feat = dino.encode_frame(seg_frames[i])
                feats.append(feat)
            
            sim = 0.0
            if len(feats) > 1:
                for i in range(1, len(feats)):
                    sim_pre = F.cosine_similarity(feats[i-1], feats[i]).clamp(min=0).item()
                    sim_fir = F.cosine_similarity(feats[0], feats[i]).clamp(min=0).item()
                    sim += (sim_pre + sim_fir) / 2
                segment_scores.append(sim / (len(feats) - 1))
            else:
                segment_scores.append(1.0)
        
        results[sample['sample_id']] = vde_decay(segment_scores)
    return results
