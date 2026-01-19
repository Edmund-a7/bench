"""StreamBench 核心评估类"""
import importlib
import json
import os
import yaml
from datetime import datetime


def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        # 默认配置路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'configs', 'default.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_path_config(path_config=None):
    """加载模型路径配置"""
    if path_config is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path_config = os.path.join(base_dir, 'configs', 'path.yml')
    
    if os.path.exists(path_config):
        with open(path_config, 'r') as f:
            return yaml.safe_load(f)
    return {}


class StreamBench:
    def __init__(self, device, output_path, config_path=None, path_config=None):
        self.device = device
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # 加载配置
        self.config = load_config(config_path)
        self.path_config = load_path_config(path_config)
        
        self.metric_folder_map = {
            "subject_consistency": "quality",
            "background_consistency": "quality",
            "temporal_flickering": "quality",
            "motion_smoothness": "quality",
            "vtss": "quality",
            "boundary_smoothness": "temporal",
            "conditional_adjacent": "temporal",
            "conditional_longrange": "temporal",
            "segment_alignment": "instruction",
            "dynamic_trajectory": "instruction",
            "vlm_score": "instruction",
        }
    
    def evaluate(self, eval_data_path, metric_list=None, **kwargs):
        """评估所有 samples"""
        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)
        
        if metric_list is None:
            metric_list = list(self.metric_folder_map.keys())
        
        # 结果存储: {sample_id: {metric: score}}
        per_sample_results = {s['sample_id']: {} for s in eval_data}
        # 结果存储: {metric: avg_score}
        aggregated_results = {}
        
        # Metric-centric 循环：每个指标加载一次模型，处理所有 samples
        for metric in metric_list:
            print(f"Evaluating metric: {metric} ...")
            folder = self.metric_folder_map[metric]
            module = importlib.import_module(f'metrics.{folder}.{metric}')
            compute_func = getattr(module, f'compute_{metric}')
            
            # 传入所有数据和配置，函数内部加载模型一次处理所有
            metric_scores = compute_func(
                eval_data=eval_data,
                device=self.device,
                config=self.config,
                path_config=self.path_config,
                **kwargs
            )
            
            # 记录结果 (metric_scores 是 {sample_id: score})
            avg_score = 0.0
            if metric_scores:
                for sid, score in metric_scores.items():
                    per_sample_results[sid][metric] = score
                    avg_score += score
                aggregated_results[metric] = avg_score / len(metric_scores)
            else:
                aggregated_results[metric] = 0.0
        
        # 计算维度汇总分
        final_aggregated = self._calculate_dimension_scores(aggregated_results)
        
        self._save_results(final_aggregated, per_sample_results)
        return final_aggregated
    
    def _calculate_dimension_scores(self, aggregated):
        """计算 Quality/Temporal/Instruction 维度分"""
        quality_metrics = ['subject_consistency', 'background_consistency', 
                          'temporal_flickering', 'motion_smoothness', 'vtss']
        temporal_metrics = ['boundary_smoothness', 'conditional_adjacent', 
                           'conditional_longrange']
        instruction_metrics = ['segment_alignment', 'dynamic_trajectory', 'vlm_score']
        
        # 从配置读取权重
        weights = self.config.get('weights', {})
        w_quality = weights.get('quality', 1.0)
        w_temporal = weights.get('temporal', 1.0)
        w_instruction = weights.get('instruction', 1.0)
        
        def avg(metrics):
            vals = [aggregated[m] for m in metrics if m in aggregated]
            return sum(vals) / len(vals) if vals else 0.0
        
        aggregated['quality_score'] = avg(quality_metrics)
        aggregated['temporal_score'] = avg(temporal_metrics)
        aggregated['instruction_score'] = avg(instruction_metrics)
        
        # 加权总分
        total_weight = w_quality + w_temporal + w_instruction
        aggregated['total_score'] = (
            w_quality * aggregated['quality_score'] + 
            w_temporal * aggregated['temporal_score'] + 
            w_instruction * aggregated['instruction_score']
        ) / total_weight
        
        return aggregated
    
    def _save_results(self, aggregated, per_sample):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 将 per_sample 转换为 list 格式
        per_sample_list = []
        for sid, metrics in per_sample.items():
            item = metrics.copy()
            item['sample_id'] = sid
            per_sample_list.append(item)
            
        output = {
            'aggregated': aggregated,
            'per_sample': per_sample_list,
            'config': self.config  # 保存使用的配置
        }
        path = os.path.join(self.output_path, f'results_{timestamp}.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {path}")
