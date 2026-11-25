import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch

# 确保优先使用本仓库中修改过的 audiocraft 包与本地模块
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PKG_ROOT = os.path.join(_REPO_ROOT, 'audiocraft')
for _p in (_REPO_ROOT, _PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audiocraft.train import get_solver_from_sig
from audiocraft.models import MultiBandDiffusion, DiffusionCompressionWrapper, build_diffusion_step_list
from audiocraft.data.audio import audio_write
from audiocraft.modules.conditioners import ConditioningAttributes
sys.path.append(str(Path(__file__).resolve().parents[1])) 
from music_prompt import GuohuaToMusicPromptor


def list_images(directory: Path, exts: List[str]) -> List[Path]:
    directory = directory.expanduser().resolve()
    files: List[Path] = []
    for ext in exts:
        files.extend(directory.glob(f'**/*.{ext}'))
        files.extend(directory.glob(f'**/*.{ext.upper()}'))
    # 去重并排序
    files = sorted(list({f.resolve() for f in files}))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sig', required=True, help='训练好的 Dora 实验签名')
    parser.add_argument('--image_dir', required=True, help='输入图像所在目录')
    parser.add_argument('--out_dir', default='batch_outputs', help='输出目录')
    parser.add_argument('--k', type=int, default=2, help='每张图生成的样本数量')
    parser.add_argument('--duration', type=float, default=10.0, help='每首音乐时长（秒）')
    parser.add_argument('--model_dir', default='./models/Qwen2.5-VL-7B-Instruct', help='VLM 模型目录')
    parser.add_argument('--format', default='wav', choices=['wav', 'mp3', 'ogg', 'flac'], help='输出音频格式')
    parser.add_argument('--use_diffusion', action='store_true', help='启用 Multi-Band Diffusion 解码器')
    parser.add_argument('--diffusion_steps', type=int, default=None, help='自定义扩散步数（如 36）')
    parser.add_argument('--seed_base', type=int, default=1234, help='随机种子基数，用于多样性')
    parser.add_argument('--skip_if_exists', action='store_true', help='若目标音频已存在则跳过')
    parser.add_argument('--exts', default='png,jpg,jpeg', help='要扫描的图片扩展名，逗号分隔')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = [e.strip('. ').lower() for e in args.exts.split(',') if e.strip()]
    images = list_images(image_dir, exts)
    if not images:
        print(f'未在目录中找到图像: {image_dir} 扩展名: {exts}')
        return

    # 1) 准备 VLM 提示器与训练好的 solver（全局复用）
    promptor = GuohuaToMusicPromptor(model_dir=args.model_dir)
    solver = get_solver_from_sig(args.sig, load_best=True, disable_fsdp=True)
    solver.model.eval()

    diffusion_model = None
    if args.use_diffusion:
        # 统一通过参数张量获取设备，兼容 LMModel / MusicGen
        model_device = next(solver.model.parameters()).device
        diffusion_model = MultiBandDiffusion.get_mbd_musicgen(device=model_device)
        step_list = None
        if args.diffusion_steps is not None:
            step_list = build_diffusion_step_list(args.diffusion_steps)
        # 如果是 MusicGen，走其自带的启用逻辑；否则对 solver.compression_model 进行包装
        if hasattr(solver.model, 'compression_model'):
            if isinstance(solver.model.compression_model, DiffusionCompressionWrapper):
                solver.model.set_diffusion_step_list(step_list)
            else:
                solver.model.enable_diffusion_decoder(diffusion=diffusion_model, step_list=step_list)
        else:
            if isinstance(solver.compression_model, DiffusionCompressionWrapper):
                solver.compression_model.set_step_list(step_list)
            else:
                solver.compression_model = DiffusionCompressionWrapper(
                    solver.compression_model, diffusion_model, step_list=step_list
                )

    # 2) 遍历图像，先生成描述，再连续生成 k 首音乐
    for img_path in images:
        try:
            abs_img = str(img_path.resolve())
            result = promptor.run(abs_img)
            description = result['final_prompt']

            # 准备条件（一次复用，图像路径后续交给 CLIP 模块转换向量）
            cond = ConditioningAttributes()
            cond.text['description'] = description
            cond.image['image_clip'] = abs_img

            # 输出基名：不含后缀
            stem = img_path.stem

            for i in range(1, args.k + 1):
                # 输出文件名：<stem>_<i>.<format>
                out_path = out_dir / f"{stem}_{i}.{args.format}"
                if args.skip_if_exists and out_path.exists():
                    print(f'[skip] 已存在: {out_path}')
                    continue

                # 设置不同随机种子以增加多样性
                seed = args.seed_base + i
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                with torch.no_grad():
                    total_gen_len = int(args.duration * solver.compression_model.frame_rate)
                    # 先用 LM 生成 token，再用（可能被扩散包裹的）压缩模型解码
                    gen_tokens = solver.model.generate(
                        prompt=None,
                        conditions=[cond],
                        max_gen_len=total_gen_len,
                        **solver.generation_params,
                    )
                    audio = solver.compression_model.decode(gen_tokens, None)[0]

                # 保存音频
                audio_write(out_path.with_suffix(''), audio.cpu(), solver.cfg.sample_rate,
                            format=args.format, add_suffix=True)
                print(f'saved to {out_path}')

        except Exception as exc:
            print(f'[error] 处理图像失败: {img_path} -> {exc}')


if __name__ == '__main__':
    main()


