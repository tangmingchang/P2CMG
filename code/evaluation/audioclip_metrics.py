import os
import sys
import torch
import torchaudio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# 将AudioCLIP添加到环境变量
if os.environ.get('AUDIOCLIP_SOURCE_DIR'):
    sys.path.insert(0, os.environ['AUDIOCLIP_SOURCE_DIR'])

# 导入AudioCLIP
try:
    from model.audioclip import AudioCLIP
    AUDIOCLIP_AVAILABLE = True
except ImportError as e:
    logging.error(f"无法导入AudioCLIP: {e}")
    AUDIOCLIP_AVAILABLE = False

logger = logging.getLogger(__name__)

# 定义图像预处理转换
def get_image_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))
    ])

class AudioCLIPMetrics:
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化AudioCLIP评价指标
        
        Args:
            model_path: 预训练模型路径，如果为None则使用默认路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not AUDIOCLIP_AVAILABLE:
            raise ImportError("AudioCLIP模块不可用，请检查安装")
        
        # 如果没有指定模型路径，使用默认路径
        if model_path is None:
            model_path = os.environ.get('AUDIOCLIP_CHECKPOINT')
        if not model_path:
            raise ValueError('Specify model_path or AUDIOCLIP_CHECKPOINT')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 加载模型
        try:
            self.model = AudioCLIP(pretrained=model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"成功加载AudioCLIP模型: {model_path}")
        except Exception as e:
            logger.error(f"加载AudioCLIP模型失败: {e}")
            raise
        
        # 初始化图像转换
        self.image_transform = get_image_transform()
    
    def load_audio(self, audio_path: str, target_sr: int = 44100) -> torch.Tensor:
        """
        加载并预处理音频
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率
            
        Returns:
            torch.Tensor: 预处理后的音频张量
        """
        try:
            # 加载音频
            audio, sr = torchaudio.load(audio_path)
            
            # 如果需要，重采样
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                audio = resampler(audio)
            
            # 确保音频长度至少为1秒
            if audio.shape[1] < target_sr:
                padding = torch.zeros(audio.shape[0], target_sr - audio.shape[1])
                audio = torch.cat([audio, padding], dim=1)
            
            # 如果有多个通道，取均值转为单通道
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # 标准化音频
            audio = audio / (audio.abs().max() + 1e-8)
            
            return audio
            
        except Exception as e:
            logger.error(f"加载音频文件失败: {e}")
            raise
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        加载并预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用预处理
            image_tensor = self.image_transform(image)
            
            return image_tensor.unsqueeze(0)  # 添加批次维度
            
        except Exception as e:
            logger.error(f"加载图像文件失败: {e}")
            raise
    
    def compute_audio_image_similarity(self, 
                                     audio: torch.Tensor, 
                                     image: torch.Tensor) -> float:
        """
        计算音频和图像之间的相似度
        
        Args:
            audio: 音频张量
            image: 图像张量
            
        Returns:
            float: 相似度得分
        """
        try:
            # 确保输入在正确的设备上
            audio = audio.to(self.device)
            image = image.to(self.device)
            
            with torch.no_grad():
                # 编码音频和图像
                audio_features = self.model.encode_audio(audio)
                image_features = self.model.encode_image(image)
                
                # 标准化特征
                audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 计算相似度
                similarity = F.cosine_similarity(audio_features, image_features)
                
                return similarity.item()
                
        except Exception as e:
            logger.error(f"计算音频-图像相似度时发生错误: {e}")
            raise
    
    def compute_image_rank(self, 
                         audio: torch.Tensor, 
                         image_list: List[torch.Tensor]) -> Dict[str, Union[float, int]]:
        """
        计算音频对应图像的排名
        
        Args:
            audio: 音频张量
            image_list: 图像张量列表
            
        Returns:
            Dict: 包含排名和得分的字典
        """
        try:
            # 确保音频在正确的设备上
            audio = audio.to(self.device)
            
            # 编码音频
            with torch.no_grad():
                audio_features = self.model.encode_audio(audio)
                audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
            
            # 计算与每个图像的相似度
            similarities = []
            for img in image_list:
                img = img.to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(img)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    similarity = F.cosine_similarity(audio_features, image_features).item()
                similarities.append(similarity)
            
            # 计算排名
            sorted_indices = np.argsort(similarities)[::-1]  # 降序排序
            
            return {
                "audio_image_similarity": similarities[0],  # 第一个图像的相似度
                "audio_image_compatibility": (similarities[0] + 1) / 2 * 100,  # 兼容性分数
                "rank": np.where(sorted_indices == 0)[0][0] + 1,  # 第一个图像的排名（不作为指标返回）
                "percentile": (1 - np.where(sorted_indices == 0)[0][0] / len(similarities)) * 100  # 百分位数（不作为指标返回）
            }
            
        except Exception as e:
            logger.error(f"计算图像排名时发生错误: {e}")
            raise
    
    def evaluate_audio_image_correspondence(self, 
                                         audio_path: str, 
                                         image_path: str) -> Dict[str, float]:
        """
        评估音频和图像之间的对应关系
        
        Args:
            audio_path: 音频文件路径
            image_path: 图像文件路径
            
        Returns:
            Dict: 包含评估指标的字典
        """
        try:
            # 加载音频和图像
            audio = self.load_audio(audio_path)
            image = self.load_image(image_path)
            
            # 计算相似度
            similarity = self.compute_audio_image_similarity(audio, image)
            
            return {
                "audio_image_similarity": similarity,
                "audio_image_compatibility": (similarity + 1) / 2 * 100  # 转换为0-100的分数
            }
            
        except Exception as e:
            logger.error(f"评估音频-图像对应关系时发生错误: {e}")
            raise
    
    def evaluate_audio_image_retrieval(self, 
                                    audio_path: str, 
                                    image_paths: List[str]) -> Dict[str, Union[float, int]]:
        """
        评估音频检索图像的性能
        
        Args:
            audio_path: 音频文件路径
            image_paths: 图像文件路径列表
            
        Returns:
            Dict: 包含评估指标的字典
        """
        try:
            # 加载音频和图像
            audio = self.load_audio(audio_path)
            images = [self.load_image(img_path) for img_path in image_paths]
            
            # 计算排名
            rank_metrics = self.compute_image_rank(audio, images)
            
            return rank_metrics
            
        except Exception as e:
            logger.error(f"评估音频-图像检索性能时发生错误: {e}")
            raise 