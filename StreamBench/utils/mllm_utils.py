"""MLLM 调用工具"""
from openai import OpenAI
import json


# 不要全局初始化，避免 import 时报错
def get_client():
    return OpenAI()


def mllm_judge_scene_change(prompt_i, prompt_i1):
    """判断两个相邻 prompt 之间是否需要保持视觉一致性"""
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f'''分析这两个视频片段描述，判断它们是否应该保持视觉上的连续性（同一场景、同一人物）。
片段1: {prompt_i}
片段2: {prompt_i1}
只回答 "是" 或 "否"。'''
        }],
        max_tokens=10
    )
    return "是" in response.choices[0].message.content


def mllm_extract_entity_groups(prompts):
    """从 prompt 序列中提取主体出现的段落索引"""
    client = get_client()
    prompt_text = "\n".join([f"{i}: {p}" for i, p in enumerate(prompts)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f'''分析以下视频片段描述序列，识别出现的主要实体，并标注每个实体在哪些片段中出现（0-indexed）。
{prompt_text}
以 JSON 格式输出，例如：{{"wizard": [0,1,2,3], "dragon": [1,2,3,4]}}'''
        }],
        max_tokens=200
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        # 默认返回：假设主角贯穿始终
        return {"main_subject": list(range(len(prompts)))}
