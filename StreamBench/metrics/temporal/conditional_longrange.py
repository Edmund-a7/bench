"""条件长程一致性：MLLM 确定同主体段落"""
from models.dino_encoder import DINOEncoder
from utils.mllm_utils import mllm_extract_entity_groups
from utils.aggregation import reverse_weighted, mean_aggregation
from utils.video_io import load_segments
from tqdm import tqdm
import torch.nn.functional as F


def compute_conditional_longrange(eval_data, device, config=None, path_config=None, **kwargs):
    print("Loading DINO model for Longrange...")
    dino = DINOEncoder(device=device, config=config, path_config=path_config)
    
    results = {}
    for sample in tqdm(eval_data, desc="Conditional Longrange"):
        prompts = sample['prompts']
        segments = load_segments(sample['segment_paths'])
        
        entity_groups = mllm_extract_entity_groups(prompts)
        
        all_scores = []
        for entity, indices in entity_groups.items():
            if len(indices) < 2:
                continue
            
            first_feat = dino.encode_segment(segments[indices[0]])
            entity_scores = []
            for idx in indices[1:]:
                feat = dino.encode_segment(segments[idx])
                sim = F.cosine_similarity(first_feat, feat, dim=-1).item()
                entity_scores.append(sim)
            
            all_scores.append(reverse_weighted(entity_scores))
        
        results[sample['sample_id']] = mean_aggregation(all_scores) if all_scores else 1.0
    return results
