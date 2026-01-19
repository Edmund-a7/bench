"""LanguageBind 编码器封装"""
import torch
from languagebind import LanguageBind, to_device, transform_dict


class LanguageBindEncoder:
    def __init__(self, device='cuda', config=None, path_config=None):
        self.device = device
        
        # 从配置读取参数
        model_config = config.get('models', {}).get('languagebind', {}) if config else {}
        paths = path_config.get('languagebind', {}) if path_config else {}
        
        cache_dir = paths.get('cache_dir') or model_config.get('cache_dir', './pretrained')
        video_model = paths.get('video_model', 'LanguageBind_Video_FT')
        
        # 加载 Video + Language 模型
        self.model = LanguageBind(
            clip_type={
                'video': video_model,
                'language': video_model  # 共享
            },
            cache_dir=cache_dir
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        # 视频和语言转换器
        self.video_transform = transform_dict['video'](self.model)
        self.language_transform = transform_dict['language'](self.model)

    def encode_video(self, video_path):
        """编码视频文件 (路径)"""
        with torch.no_grad():
            inputs = self.video_transform({'video': [video_path]})
            inputs = to_device(inputs, self.device)
            feat = self.model(inputs)['video']
            return feat.squeeze(0)

    def encode_text(self, text):
        """编码文本"""
        with torch.no_grad():
            inputs = self.language_transform({'language': [text]})
            inputs = to_device(inputs, self.device)
            feat = self.model(inputs)['language']
            return feat.squeeze(0)
