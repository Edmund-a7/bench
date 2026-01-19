"""VLM 客户端封装"""
import base64
import cv2
from openai import OpenAI


class VLMClient:
    def __init__(self, model='gpt-4o', config=None, path_config=None):
        # 从配置读取参数
        model_config = config.get('models', {}).get('vlm', {}) if config else {}
        
        self.model = model or model_config.get('model', 'gpt-4o')
        self.max_tokens = model_config.get('max_tokens', 10)
        self.client = OpenAI()
    
    def _encode_video(self, video_path):
        """编码视频为 base64（提取关键帧）"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        
        indices = [0, total//4, total//2, 3*total//4, total-1]
        indices = sorted(list(set([i for i in indices if i < total])))
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(base64.b64encode(buffer).decode('utf-8'))
        cap.release()
        return frames
    
    def evaluate(self, video_path, prompt):
        """评估视频"""
        frames_b64 = self._encode_video(video_path)
        if not frames_b64:
            return "1"
            
        content = [{"type": "text", "text": prompt}]
        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
