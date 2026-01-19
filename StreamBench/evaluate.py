#!/usr/bin/env python
"""评估入口"""
import argparse
import json
import torch
from streambench import StreamBench


def main():
    parser = argparse.ArgumentParser(description='StreamBench Evaluation')
    parser.add_argument('--eval_data', required=True, help='eval_data.json 路径')
    parser.add_argument('--output', required=True, help='结果输出目录')
    parser.add_argument('--config', default=None, help='配置文件路径')
    parser.add_argument('--path_config', default=None, help='模型路径配置')
    parser.add_argument('--metrics', nargs='+', default=None, help='指标列表，默认全部')
    parser.add_argument('--vlm_model', default='gpt-4o', help='VLM 模型')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    bench = StreamBench(
        device=device,
        output_path=args.output,
        config_path=args.config,
        path_config=args.path_config
    )
    
    metrics = None if args.metrics == ['all'] else args.metrics
    
    results = bench.evaluate(
        args.eval_data, 
        metric_list=metrics,
        vlm_model=args.vlm_model
    )
    
    print("=== Evaluation Results ===")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
