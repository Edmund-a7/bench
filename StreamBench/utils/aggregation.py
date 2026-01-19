"""聚合策略"""
import math


def mean_aggregation(scores):
    """简单均值"""
    return sum(scores) / len(scores) if scores else 0.0


def vde_decay(scores, weight_type='linear'):
    """VDE 漂移衰减：惩罚后续段分数下降"""
    if len(scores) < 2:
        return scores[0] if scores else 0.0
    
    baseline, n = scores[0], len(scores)
    weighted_sum, weight_sum = 0.0, 0.0
    
    for i, score in enumerate(scores[1:], start=2):
        delta = max(0, baseline - score) / (baseline + 1e-6)
        w = (n - i + 1) / n if weight_type == 'linear' else math.exp(-0.5 * (i-1))
        weighted_sum += w * delta
        weight_sum += w
    
    penalty = weighted_sum / weight_sum if weight_sum > 0 else 0
    return mean_aggregation(scores) * (1 - penalty)


def reverse_weighted(scores):
    """逆序加权：后面权重更大，惩罚长程不一致"""
    if not scores:
        return 0.0
    n = len(scores)
    weights = [(i + 1) / n for i in range(n)]
    return sum(w * s for w, s in zip(weights, scores)) / sum(weights)
