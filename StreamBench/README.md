# StreamBench

流式交互视频生成 Benchmark，评估 IAMFlow/MemFlow/LongLive 等方法。

## 评估维度

| 维度 | 指标 | 聚合策略 |
|------|------|----------|
| **Quality** | Subject Consistency, Background Consistency, Temporal Flickering, Motion Smoothness, VTSS | VDE Decay / Mean |
| **Temporal** | Boundary Smoothness, Conditional Adjacent, Conditional Longrange | Mean / Reverse-weighted |
| **Instruction** | Segment Alignment, Dynamic Trajectory, VLM Score | Mean |

## 使用方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
python preprocess.py \
    --video_dir ./outputs/memflow/videos/ \
    --prompts ./interactive_10.jsonl \
    --output ./processed/ \
    --segment_duration 10
```

### 3. 运行评估

```bash
python evaluate.py \
    --eval_data ./processed/eval_data.json \
    --output ./results/ \
    --metrics all
```

## 输出格式

```json
{
  "aggregated": {
    "subject_consistency": 0.85,
    "quality_score": 0.85,
    "temporal_score": 0.77,
    "instruction_score": 0.80,
    "total_score": 0.81
  },
  "per_sample": [...]
}
```
