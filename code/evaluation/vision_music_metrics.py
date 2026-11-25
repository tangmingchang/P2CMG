import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class VisionMusicMetrics:
    def __init__(self):
        """初始化Vision-Music指标计算器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_image_bind_score(self, 
                               audio_embedding: torch.Tensor,
                               image_embedding: torch.Tensor,
                               temperature: float = 0.07) -> float:
        """
        计算Image Bind Score
        
        Args:
            audio_embedding: 音频嵌入向量 [batch_size, embedding_dim]
            image_embedding: 图像嵌入向量 [batch_size, embedding_dim]
            temperature: 温度参数，用于控制softmax的平滑程度
            
        Returns:
            float: Image Bind Score
        """
        try:
            # 确保输入是PyTorch张量
            if not isinstance(audio_embedding, torch.Tensor):
                audio_embedding = torch.tensor(audio_embedding)
            if not isinstance(image_embedding, torch.Tensor):
                image_embedding = torch.tensor(image_embedding)
            
            # 确保张量在正确的设备上
            audio_embedding = audio_embedding.to(self.device)
            image_embedding = image_embedding.to(self.device)
            
            # 计算相似度矩阵
            similarity = F.cosine_similarity(audio_embedding.unsqueeze(1), 
                                          image_embedding.unsqueeze(0), 
                                          dim=2)
            
            # 应用温度缩放
            similarity = similarity / temperature
            
            # 计算softmax
            probs = F.softmax(similarity, dim=1)
            
            # 计算对角线上的概率（正确匹配的概率）
            diagonal_probs = torch.diag(probs)
            
            # 计算平均分数
            score = diagonal_probs.mean().item()
            
            return score
            
        except Exception as e:
            logger.error(f"计算Image Bind Score时发生错误: {str(e)}")
            raise
    
    def compute_cross_model_relevance(self,
                                    audio_embedding: torch.Tensor,
                                    image_embedding: torch.Tensor,
                                    text_embedding: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        计算Cross-model Relevance指标
        
        Args:
            audio_embedding: 音频嵌入向量 [batch_size, embedding_dim]
            image_embedding: 图像嵌入向量 [batch_size, embedding_dim]
            text_embedding: 文本嵌入向量 [batch_size, embedding_dim]，可选
            
        Returns:
            Dict[str, float]: 包含各项指标的字典
        """
        try:
            # 确保输入是PyTorch张量
            if not isinstance(audio_embedding, torch.Tensor):
                audio_embedding = torch.tensor(audio_embedding)
            if not isinstance(image_embedding, torch.Tensor):
                image_embedding = torch.tensor(image_embedding)
            if text_embedding is not None and not isinstance(text_embedding, torch.Tensor):
                text_embedding = torch.tensor(text_embedding)
            
            # 确保张量在正确的设备上
            audio_embedding = audio_embedding.to(self.device)
            image_embedding = image_embedding.to(self.device)
            if text_embedding is not None:
                text_embedding = text_embedding.to(self.device)
            
            # 计算音频-图像相关性
            audio_image_correlation = F.cosine_similarity(
                audio_embedding, image_embedding, dim=1
            ).mean().item()
            
            results = {
                "audio_image_correlation": audio_image_correlation
            }
            
            # 如果提供了文本嵌入，计算音频-文本和图像-文本相关性
            if text_embedding is not None:
                audio_text_correlation = F.cosine_similarity(
                    audio_embedding, text_embedding, dim=1
                ).mean().item()
                
                image_text_correlation = F.cosine_similarity(
                    image_embedding, text_embedding, dim=1
                ).mean().item()
                
                results.update({
                    "audio_text_correlation": audio_text_correlation,
                    "image_text_correlation": image_text_correlation,
                    "cross_modal_alignment": (audio_image_correlation + 
                                           audio_text_correlation + 
                                           image_text_correlation) / 3
                })
            
            return results
            
        except Exception as e:
            logger.error(f"计算Cross-model Relevance时发生错误: {str(e)}")
            raise
    
    def evaluate_vision_music_correspondence(self,
                                          audio_embedding: torch.Tensor,
                                          image_embedding: torch.Tensor,
                                          text_embedding: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        评估Vision-Music Correspondence的所有指标
        
        Args:
            audio_embedding: 音频嵌入向量
            image_embedding: 图像嵌入向量
            text_embedding: 文本嵌入向量，可选
            
        Returns:
            Dict[str, float]: 包含所有评估指标的字典
        """
        try:
            # 计算Image Bind Score
            image_bind_score = self.compute_image_bind_score(
                audio_embedding, image_embedding
            )
            
            # 计算Cross-model Relevance
            cross_model_metrics = self.compute_cross_model_relevance(
                audio_embedding, image_embedding, text_embedding
            )
            
            # 合并所有指标
            results = {
                "image_bind_score": image_bind_score,
                **cross_model_metrics
            }
            
            return results
            
        except Exception as e:
            logger.error(f"评估Vision-Music Correspondence时发生错误: {str(e)}")
            raise 