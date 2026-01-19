"""VLM 评分：GPT-4o 视频评分"""
from models.vlm_client import VLMClient
from utils.video_io import merge_segments
from tqdm import tqdm
import tempfile
import os

PROMPT_TEMPLATE = '''你是一个视频评估专家。下面是一个流式生成的视频和对应的提示词序列。
请判断视频是否在正确的时间点执行了正确的动作。

提示词序列：
{prompts}

请给出 1-5 分的评分：
1分：完全不遵循提示词
2分：大部分不遵循
3分：部分遵循
4分：大部分遵循
5分：完全正确执行

只回复数字评分，不要其他内容。'''


def compute_vlm_score(eval_data, device, config=None, path_config=None, **kwargs):
    print("Initializing VLM client...")
    vlm = VLMClient(
        model=kwargs.get('vlm_model', 'gpt-4o'),
        config=config,
        path_config=path_config
    )
    
    results = {}
    for sample in tqdm(eval_data, desc="VLM Score"):
        prompts = sample['prompts']
        segment_paths = sample['segment_paths']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_path = os.path.join(tmpdir, 'merged.mp4')
            merge_segments(segment_paths, merged_path)
            
            prompt_text = PROMPT_TEMPLATE.format(prompts="\n".join(
                [f"{i+1}. {p}" for i, p in enumerate(prompts)]
            ))
            
            try:
                raw_score = vlm.evaluate(merged_path, prompt_text)
                score = (float(raw_score.strip()) - 1) / 4
                score = max(0.0, min(1.0, score))
            except Exception as e:
                print(f"VLM Scoring failed for sample {sample['sample_id']}: {e}")
                score = 0.5
        
        results[sample['sample_id']] = score
    return results
