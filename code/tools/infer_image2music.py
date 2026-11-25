import argparse
import os
import sys
from pathlib import Path

import torch

# 确保优先使用本仓库中修改过的 audiocraft 包
_REPO_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audiocraft'))
if _REPO_PKG_ROOT not in sys.path:
    sys.path.insert(0, _REPO_PKG_ROOT)

from audiocraft.train import get_solver_from_sig
from audiocraft.models import MultiBandDiffusion, DiffusionCompressionWrapper, build_diffusion_step_list
from audiocraft.data.audio import audio_write
from audiocraft.modules.conditioners import ConditioningAttributes
sys.path.append(str(Path(__file__).resolve().parents[1])) 
from music_prompt import GuohuaToMusicPromptor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sig', required=True, help='训练好的 Dora 实验签名')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--out', default='out.wav')
    parser.add_argument('--model_dir', default='./models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--use_diffusion', action='store_true', help='启用 Multi-Band Diffusion 解码器')
    parser.add_argument('--diffusion_steps', type=int, default=None, help='自定义扩散步数（例如 36）')
    args = parser.parse_args()

    # 1) 用 VLM 产生描述
    promptor = GuohuaToMusicPromptor(model_dir=args.model_dir)
    result = promptor.run(os.path.abspath(args.image))
    description = result['final_prompt']

    # 2) 取训练好的 solver
    solver = get_solver_from_sig(args.sig, load_best=True, disable_fsdp=True)
    solver.model.eval()

    if args.use_diffusion:
        model = solver.model
        # 兼容 LMModel / MusicGen，统一通过参数张量获取 device
        model_device = next(model.parameters()).device
        diffusion = MultiBandDiffusion.get_mbd_musicgen(device=model_device)
        step_list = None
        if args.diffusion_steps is not None:
            step_list = build_diffusion_step_list(args.diffusion_steps)
        # 如果是 MusicGen 模型，直接启用其扩散解码器；否则（LMModel），包装 solver.compression_model
        if hasattr(model, 'compression_model'):
            if isinstance(model.compression_model, DiffusionCompressionWrapper):
                model.set_diffusion_step_list(step_list)
            else:
                model.enable_diffusion_decoder(diffusion=diffusion, step_list=step_list)
        else:
            if isinstance(solver.compression_model, DiffusionCompressionWrapper):
                solver.compression_model.set_step_list(step_list)
            else:
                solver.compression_model = DiffusionCompressionWrapper(
                    solver.compression_model, diffusion, step_list=step_list
                )

    # 3) 构造条件（文本 + 图像路径，后续由 CLIP 模块转向量）
    cond = ConditioningAttributes()
    cond.text['description'] = description
    cond.image['image_clip'] = os.path.abspath(args.image)

    with torch.no_grad():
        total_gen_len = int(args.duration * solver.compression_model.frame_rate)
        # 直接用 LM 生成 token，再用（可能被扩散包裹的）压缩模型解码为音频
        gen_tokens = solver.model.generate(
            prompt=None,
            conditions=[cond],
            max_gen_len=total_gen_len,
            **solver.generation_params,
        )
        audio = solver.compression_model.decode(gen_tokens, None)[0]

    # 4) 保存音频（自动根据后缀选择格式）
    out_path = Path(args.out)
    suffix = out_path.suffix.lower().lstrip('.') or 'wav'
    audio_write(out_path if out_path.suffix else out_path.with_suffix('.wav'),
                audio.cpu(), solver.cfg.sample_rate, format=suffix, add_suffix=False)
    print('saved to', args.out)


if __name__ == '__main__':
    main()


