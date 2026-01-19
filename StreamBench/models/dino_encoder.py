"""DINO 编码器封装"""
import torch
import torch.nn.functional as F
from vbench.utils import dino_transform


class DINOEncoder:
    def __init__(self, device='cuda', config=None, path_config=None):
        self.device = device
        
        # 从配置读取参数
        model_config = config.get('models', {}).get('dino', {}) if config else {}
        paths = path_config.get('dino', {}) if path_config else {}
        
        input_size = model_config.get('input_size', 224)
        weights_path = paths.get('weights_path')
        
        # 加载模型
        if weights_path and torch.cuda.is_available():
            # 离线加载
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
            self.model.load_state_dict(torch.load(weights_path))
        else:
            # 在线下载
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = dino_transform(input_size)
    
    def encode_frame(self, frame):
        """编码单帧 (PIL Image)"""
        with torch.no_grad():
            tensor = self.transform(frame).unsqueeze(0).to(self.device)
            feat = self.model(tensor)
            return F.normalize(feat, dim=-1)
    
    def encode_segment(self, frames):
        """编码视频段 (list of PIL)，返回平均特征"""
        if len(frames) == 0:
            return torch.zeros(1, 384).to(self.device)
        
        n = len(frames)
        indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
        indices = [i for i in indices if i < n]
        
        feats = [self.encode_frame(frames[i]) for i in indices]
        return torch.stack(feats).mean(dim=0)
