"""分段语义对齐：各段视频与 prompt 的 CLIP Score"""
from models.clip_encoder import ViCLIPEncoder
from utils.video_io import load_segments
from utils.aggregation import mean_aggregation
from tqdm import tqdm


def compute_segment_alignment(eval_data, device, config=None, path_config=None, **kwargs):
    print("Loading ViCLIP model...")
    clip = ViCLIPEncoder(device=device, config=config, path_config=path_config)
    
    results = {}
    for sample in tqdm(eval_data, desc="Segment Alignment"):
        prompts = sample['prompts']
        segments = load_segments(sample['segment_paths'])
        
        scores = []
        for seg, prompt in zip(segments, prompts):
            sim = clip.compute_similarity(seg, prompt)
            scores.append(sim)
        
        results[sample['sample_id']] = mean_aggregation(scores)
    return results
