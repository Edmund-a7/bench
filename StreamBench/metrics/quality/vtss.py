"""VTSS (Video-Text Semantic Similarity): 视频-文本语义一致性"""
from models.clip_encoder import ViCLIPEncoder
from utils.video_io import load_segments
from utils.aggregation import vde_decay
from tqdm import tqdm


def compute_vtss(eval_data, device, config=None, path_config=None, **kwargs):
    print("Loading ViCLIP model for VTSS...")
    clip = ViCLIPEncoder(device=device, config=config, path_config=path_config)
    
    results = {}
    for sample in tqdm(eval_data, desc="VTSS"):
        prompts = sample['prompts']
        segments = load_segments(sample['segment_paths'])
        
        scores = []
        for seg, prompt in zip(segments, prompts):
            sim = clip.compute_similarity(seg, prompt)
            scores.append(sim)
            
        results[sample['sample_id']] = vde_decay(scores)
    return results
