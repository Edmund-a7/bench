"""RAFT 光流模型封装"""
import torch
import numpy as np
from PIL import Image


class RAFTFlow:
    def __init__(self, device='cuda', config=None, path_config=None):
        self.device = device
        
        # 从配置读取参数（预留扩展）
        # paths = path_config.get('raft', {}) if path_config else {}
        # weights_path = paths.get('weights_path')
        
        # 使用 torchvision 内置的 RAFT
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device)
        self.model.eval()
        self.transforms = Raft_Large_Weights.DEFAULT.transforms()
    
    def compute_flow(self, frame1, frame2):
        """计算两帧之间的光流"""
        # 转换 PIL 到 tensor
        if isinstance(frame1, Image.Image):
            frame1 = torch.from_numpy(np.array(frame1)).permute(2, 0, 1).float()
        if isinstance(frame2, Image.Image):
            frame2 = torch.from_numpy(np.array(frame2)).permute(2, 0, 1).float()
        
        # 应用变换
        img1, img2 = self.transforms(frame1, frame2)
        img1 = img1.unsqueeze(0).to(self.device)
        img2 = img2.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            flow = self.model(img1, img2)[-1]  # 取最后一个迭代结果
        
        return flow.squeeze(0)  # [2, H, W]
