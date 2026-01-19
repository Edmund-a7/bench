"""背景一致性：CLIP 图像编码器计算背景相似度"""
import torch
import torch.nn.functional as F
from models.clip_encoder import ViCLIPEncoder
from utils.video_io import load_segments
from utils.aggregation import vde_decay
from tqdm import tqdm


def compute_background_consistency(eval_data, device, config=None, path_config=None, **kwargs):
    print("Loading CLIP model for Background Consistency...")
    clip_encoder = ViCLIPEncoder(device=device, config=config, path_config=path_config)
    
    results = {}
    for sample in tqdm(eval_data, desc="Background Consistency"):
        segment_paths = sample['segment_paths']
        segments = load_segments(segment_paths)
        segment_scores = []
        
        for seg_frames in segments:
            if len(seg_frames) < 2:
                segment_scores.append(1.0)
                continue
                
            frames = [seg_frames[0], seg_frames[-1]]
            feats = []
            with torch.no_grad():
                for frame in frames:
                    img = clip_encoder.preprocess(frame).unsqueeze(0).to(device)
                    feats.append(clip_encoder.model.encode_image(img))
            
            sim = F.cosine_similarity(feats[0], feats[1], dim=-1).item()
            segment_scores.append(sim)
            
        results[sample['sample_id']] = vde_decay(segment_scores)
    return results
