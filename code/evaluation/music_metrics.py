import torch
import torchaudio
import numpy as np
import librosa
from scipy.stats import entropy
from scipy.signal import find_peaks
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MusicMetricsCalculator:
    """音乐质量评估指标计算器"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def _ensure_numpy(self, audio):
        """确保音频数据是numpy格式"""
        if isinstance(audio, torch.Tensor):
            if audio.dim() > 2:  # [B, C, T]
                audio = audio.squeeze(0)  # 移除批次维度
            if audio.dim() > 1:  # [C, T]
                audio = audio.mean(dim=0)  # 单声道化
            audio = audio.cpu().numpy()
        return audio
    
    def _extract_pitch(self, audio, fmin=80, fmax=2000):
        """提取音高"""
        # 确保音频是numpy数组
        audio = self._ensure_numpy(audio)
        
        # 使用librosa提取音高
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate, 
                                               fmin=fmin, fmax=fmax)
        return pitches, magnitudes
    
    def _extract_onsets(self, audio, hop_length=512):
        """提取音符起始点"""
        # 确保音频是numpy数组
        audio = self._ensure_numpy(audio)
        
        # 使用librosa提取音符起始时间
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate, 
                                           hop_length=hop_length)
        return onsets, onset_env
    
    def _extract_beats(self, audio, hop_length=512):
        """提取节拍"""
        # 确保音频是numpy数组
        audio = self._ensure_numpy(audio)
        
        # 使用librosa提取节拍
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate, hop_length=hop_length)
        return tempo, beats
    
    def _extract_chroma(self, audio, hop_length=512):
        """提取色度图"""
        # 确保音频是numpy数组
        audio = self._ensure_numpy(audio)
        
        # 使用librosa提取色度图
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate, hop_length=hop_length)
        return chroma
    
    def scale_consistency(self, audio):
        """计算音阶一致性
        
        音阶一致性衡量音频中的音符是否遵循一致的音阶/调式
        """
        try:
            # 提取色度图
            chroma = self._extract_chroma(audio)
            
            # 计算每个音符的平均存在强度
            chroma_mean = np.mean(chroma, axis=1)
            
            # 找出最强的5个音符
            top5_notes = np.argsort(chroma_mean)[-5:]
            
            # 计算这5个音符的和与总和的比例
            top5_energy = np.sum(chroma_mean[top5_notes])
            total_energy = np.sum(chroma_mean)
            
            # 音阶一致性：越高说明越集中在几个特定音符上，形成明确的调式
            consistency = top5_energy / total_energy if total_energy > 0 else 0
            
            return consistency
        except Exception as e:
            logger.error(f"计算音阶一致性时出错: {str(e)}")
            return None
    
    def pitch_entropy(self, audio):
        """计算音高熵
        
        音高熵衡量音高分布的复杂度，越低表示音高分布越集中
        """
        try:
            # 提取色度图
            chroma = self._extract_chroma(audio)
            
            # 计算每个音符的平均存在强度
            chroma_mean = np.mean(chroma, axis=1)
            
            # 归一化
            chroma_norm = chroma_mean / np.sum(chroma_mean) if np.sum(chroma_mean) > 0 else chroma_mean
            
            # 计算熵
            pitch_entropy = entropy(chroma_norm)
            
            return pitch_entropy
        except Exception as e:
            logger.error(f"计算音高熵时出错: {str(e)}")
            return None
    
    def pitch_class_histogram(self, audio):
        """生成音高分布直方图
        
        返回每个音高类别的能量分布
        """
        try:
            # 提取色度图
            chroma = self._extract_chroma(audio)
            
            # 计算每个音符的平均存在强度
            pitch_histogram = np.mean(chroma, axis=1)
            
            # 创建带有音符名称的直方图字典
            result = {note: float(energy) for note, energy in zip(self.note_names, pitch_histogram)}
            
            return result
        except Exception as e:
            logger.error(f"计算音高分布直方图时出错: {str(e)}")
            return None
    
    def empty_beat_rate(self, audio, hop_length=512):
        """计算空拍率
        
        空拍率表示没有音符起始的节拍比例
        """
        try:
            # 提取节拍和音符起始点
            _, beats = self._extract_beats(audio, hop_length=hop_length)
            onsets, _ = self._extract_onsets(audio, hop_length=hop_length)
            
            if len(beats) == 0:
                return 0.0
                
            # 定义窗口大小（节拍前后的帧数）
            window_size = 2
            
            # 统计每个节拍附近是否有音符起始
            empty_beats = 0
            for beat in beats:
                # 检查节拍附近是否有音符起始
                beat_has_onset = False
                for onset in onsets:
                    if abs(beat - onset) <= window_size:
                        beat_has_onset = True
                        break
                if not beat_has_onset:
                    empty_beats += 1
            
            # 计算空拍率
            empty_rate = empty_beats / len(beats)
            
            return empty_rate
        except Exception as e:
            logger.error(f"计算空拍率时出错: {str(e)}")
            return None
    
    def average_ioi(self, audio, hop_length=512):
        """计算平均音符间隔
        
        平均音符间隔（Inter-Onset Interval）表示相邻音符起始点之间的平均时间
        """
        try:
            # 提取音符起始点
            onsets, _ = self._extract_onsets(audio, hop_length=hop_length)
            
            if len(onsets) < 2:
                return 0.0
                
            # 计算相邻音符起始点之间的时间间隔
            ioi = np.diff(onsets) * hop_length / self.sample_rate  # 转换为秒
            
            # 计算平均间隔
            avg_ioi = np.mean(ioi)
            
            return avg_ioi
        except Exception as e:
            logger.error(f"计算平均音符间隔时出错: {str(e)}")
            return None
    
    def groove_similarity(self, audio, reference=None, hop_length=512):
        """计算律动模式相似度
        
        如果提供了参考音频，计算与参考音频的律动相似度；
        否则，计算音频内部的律动一致性
        """
        try:
            # 确保音频是numpy数组
            audio = self._ensure_numpy(audio)
            
            # 提取音符起始强度包络
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=hop_length)
            
            if reference is not None:
                # 计算与参考音频的律动相似度
                reference = self._ensure_numpy(reference)
                ref_onset_env = librosa.onset.onset_strength(y=reference, sr=self.sample_rate, 
                                                             hop_length=hop_length)
                
                # 调整长度使两者相同
                min_length = min(len(onset_env), len(ref_onset_env))
                onset_env = onset_env[:min_length]
                ref_onset_env = ref_onset_env[:min_length]
                
                # 计算相关性
                correlation = np.corrcoef(onset_env, ref_onset_env)[0, 1]
                return (correlation + 1) / 2  # 归一化到 [0, 1]
            else:
                # 计算音频内部的律动一致性
                # 将音频分为两半，计算它们的律动相似度
                half = len(onset_env) // 2
                if half > 0:
                    first_half = onset_env[:half]
                    second_half = onset_env[half:2*half]
                    correlation = np.corrcoef(first_half, second_half)[0, 1]
                    return (correlation + 1) / 2  # 归一化到 [0, 1]
                else:
                    return 0.0
        except Exception as e:
            logger.error(f"计算律动相似度时出错: {str(e)}")
            return None
    
    def structure_indicator(self, audio, hop_length=512):
        """计算结构指标
        
        结构指标衡量音频中的结构重复性和组织性
        """
        try:
            # 确保音频是numpy数组
            audio = self._ensure_numpy(audio)
            
            # 计算自相似矩阵
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, hop_length=hop_length, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            combined = np.vstack([mfcc, mfcc_delta])
            
            # 计算自相似矩阵
            S = librosa.segment.recurrence_matrix(combined, mode='affinity')
            
            # 计算块对角性，这反映了结构的存在
            n_blocks = 4
            block_size = S.shape[0] // n_blocks
            
            block_score = 0
            for i in range(n_blocks):
                for j in range(n_blocks):
                    block = S[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    block_score += np.mean(block)
            
            block_score /= (n_blocks * n_blocks)
            
            # 归一化
            structure_score = block_score / np.mean(S)
            
            return structure_score
        except Exception as e:
            logger.error(f"计算结构指标时出错: {str(e)}")
            return None
    
    def beat_consistency(self, audio, hop_length=512):
        """计算节拍一致性分数 (BCS)
        
        节拍一致性分数衡量节拍间隔的稳定性
        """
        try:
            # 提取节拍
            tempo, beats = self._extract_beats(audio, hop_length=hop_length)
            
            if len(beats) < 3:
                return 0.0
                
            # 计算节拍间隔
            beat_intervals = np.diff(beats)
            
            # 计算间隔的变异系数（标准差/平均值）
            cv = np.std(beat_intervals) / np.mean(beat_intervals) if np.mean(beat_intervals) > 0 else 0
            
            # 归一化到 [0, 1]，值越高表示节拍越一致
            consistency = np.exp(-cv)
            
            return consistency
        except Exception as e:
            logger.error(f"计算节拍一致性分数时出错: {str(e)}")
            return None
    
    def beat_histogram_similarity(self, audio, reference=None, hop_length=512):
        """计算节拍直方图相似度 (BHS)
        
        如果提供了参考音频，计算与参考音频的节拍直方图相似度；
        否则，返回节拍直方图的峰度（作为节拍清晰度的度量）
        """
        try:
            # 确保音频是numpy数组
            audio = self._ensure_numpy(audio)
            
            # 计算节拍直方图
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=hop_length)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sample_rate, 
                                                  hop_length=hop_length)
            beat_histogram = np.mean(tempogram, axis=1)
            
            if reference is not None:
                # 计算与参考音频的节拍直方图相似度
                reference = self._ensure_numpy(reference)
                ref_onset_env = librosa.onset.onset_strength(y=reference, sr=self.sample_rate, 
                                                             hop_length=hop_length)
                ref_tempogram = librosa.feature.tempogram(onset_envelope=ref_onset_env, sr=self.sample_rate, 
                                                          hop_length=hop_length)
                ref_beat_histogram = np.mean(ref_tempogram, axis=1)
                
                # 计算余弦相似度
                dot_product = np.dot(beat_histogram, ref_beat_histogram)
                norm_product = np.linalg.norm(beat_histogram) * np.linalg.norm(ref_beat_histogram)
                similarity = dot_product / norm_product if norm_product > 0 else 0
                
                return similarity
            else:
                # 计算节拍直方图的峰度
                normalized_hist = beat_histogram / np.sum(beat_histogram) if np.sum(beat_histogram) > 0 else beat_histogram
                peaks, _ = find_peaks(normalized_hist)
                
                if len(peaks) > 0:
                    peak_heights = normalized_hist[peaks]
                    # 峰值的显著性，越高表示节拍越明显
                    peak_prominence = np.sum(peak_heights) / len(peaks)
                    return peak_prominence * 5  # 缩放到合理范围
                else:
                    return 0.0
        except Exception as e:
            logger.error(f"计算节拍直方图相似度时出错: {str(e)}")
            return None
    
    def inception_score(self, audio, hop_length=512):
        """计算Inception Score (IS)
        
        Inception Score衡量音频特征分布的多样性和清晰度
        """
        try:
            # 确保音频是numpy数组
            audio = self._ensure_numpy(audio)
            
            # 提取特征
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, hop_length=hop_length, n_mfcc=20)
            
            # 归一化特征
            mfcc_norm = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-6)
            
            # 计算帧之间的KL散度的平均值
            n_frames = mfcc_norm.shape[1]
            if n_frames < 2:
                return 0.0
                
            # 将MFCC特征转换为概率分布
            features_softmax = np.exp(mfcc_norm) / np.sum(np.exp(mfcc_norm), axis=0, keepdims=True)
            
            # 计算平均分布
            mean_distribution = np.mean(features_softmax, axis=1)
            
            # 计算每帧与平均分布的KL散度
            kl_divs = []
            for i in range(n_frames):
                frame_distribution = features_softmax[:, i]
                kl_div = np.sum(frame_distribution * np.log(frame_distribution / mean_distribution + 1e-10))
                kl_divs.append(kl_div)
            
            # Inception Score是KL散度的指数平均值
            is_score = np.exp(np.mean(kl_divs))
            
            # 归一化到合理范围
            return min(max(is_score / 10, 0), 1)
        except Exception as e:
            logger.error(f"计算Inception Score时出错: {str(e)}")
            return None
    
    def all_metrics(self, audio, reference=None):
        """计算所有音乐指标
        
        返回包含所有指标的字典
        """
        results = {
            "音阶一致性": self.scale_consistency(audio),
            "音高熵": self.pitch_entropy(audio),
            "空拍率": self.empty_beat_rate(audio),
            "平均音符间隔": self.average_ioi(audio),
            "律动一致性": self.groove_similarity(audio),
            "结构指标": self.structure_indicator(audio),
            "节拍一致性": self.beat_consistency(audio),
            "节拍清晰度": self.beat_histogram_similarity(audio),
            "Inception分数": self.inception_score(audio)
        }
        
        # 如果提供了参考音频，计算比较指标
        if reference is not None:
            results["律动相似度"] = self.groove_similarity(audio, reference)
            results["节拍直方图相似度"] = self.beat_histogram_similarity(audio, reference)
        
        return results 