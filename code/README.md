# P2CMG source code

P2CMG uses hierarchical semantic mapping to connect traditional Chinese paintings with musical conditions for audio generation. This release contains the recovered AudioCraft-based implementation, the semantic representation and conditioning modules, evaluation utilities, and a separate codec-conditioned spectral refiner.

## Repository layout

| Path | Purpose |
| --- | --- |
| [music_prompt.py](music_prompt.py) | Extract musical proposals from a painting using Qwen2.5-VL |
| [audiocraft/](audiocraft/) | AudioCraft fork with image conditioning, MusicGen integration, and decoder interfaces |
| [train_repaired.py](train_repaired.py) | Validate source-disjoint splits and prepare or launch training |
| [tools/](tools/) | Generate music for one image or a folder using a compatible trained experiment |
| [semantic/](semantic/) | Evidence-linked schema, extraction questions, and independent-field conditioning used in the supplementary study |
| [fdmbd/](fdmbd/) | Codec-conditioned spectral refiner and token-to-waveform interface |
| [evaluation/](evaluation/) | Audio preprocessing, semantic, music, and optional AudioCLIP utilities |

## Setup

From this directory, use a Linux or WSL CUDA environment with FFmpeg. The archived AudioCraft fork pins PyTorch 2.1.0. Install the corresponding CUDA build for your machine and then run:

```sh
python -m pip install -e ./audiocraft
python -m pip install dora-search==0.1.12 'transformers>=4.49,<5' qwen-vl-utils
```

Use local Qwen2.5-VL model files for visual interpretation. AudioCraft, Qwen and AudioCLIP pretrained assets have separate terms. Obtain them from their respective publishers.

## Extract a musical prompt

```sh
python music_prompt.py --image /path/to/painting.jpg --model_dir /path/to/Qwen2.5-VL-7B-Instruct
```

## Prepare training and run image-to-music inference

Supply separate train, validation, and test JSONL files with audio metadata, `path`, `image_path`, `split`, and `source_group_id`. The launcher checks that sources, recordings, and image paths are disjoint. Run with `--prepare-only` to verify the manifests and inspect the run plan before starting training.

```sh
python train_repaired.py --train train.jsonl --valid valid.jsonl --test test.jsonl \
  --run-dir runs/p2cmg --epochs 1 --updates-per-epoch 1 --prepare-only
```

To generate music, supply a locally trained compatible Dora experiment and its signature. The default inference script uses the archived joint visual/text conditioning with EnCodec. Its optional `--use_diffusion` flag selects AudioCraft's pretrained multi-band diffusion decoder.

```sh
export AUDIOCRAFT_CLUSTER=default
export AUDIOCRAFT_DORA_DIR=/path/to/dora
export AUDIOCRAFT_REFERENCE_DIR=/path/to/dora/reference
python tools/infer_image2music.py --sig YOUR_DORA_SIGNATURE --image /path/to/painting.jpg \
  --model_dir /path/to/Qwen2.5-VL-7B-Instruct --out outputs/example.wav
```

The separately trained codec-conditioned refiner requires matching 32-kHz EnCodec RVQ tokens and a trusted refiner checkpoint. Its CLI records the raw correction, codec reconstruction, output waveform and decode receipt.

```sh
python fdmbd/decode_tokens.py --tokens /path/to/tokens.pt --weights /path/to/refiner.pt \
  --codec /path/to/compression_state_dict.bin --output-dir outputs/refiner
```

For the optional AudioCLIP scorer, set AUDIOCLIP_SOURCE_DIR to the original AudioCLIP checkout and AUDIOCLIP_CHECKPOINT to its local weights.
For the optional AudioCraft FAD and CLAP scorers, set P2CMG_FAD_CHECKPOINT and P2CMG_CLAP_CHECKPOINT to local checkpoint files. Use explicit local paths for historical checkpoint references in saved configurations.

## Release scope

This is a source-code release. It does not distribute the training or evaluation corpus, project checkpoints, private experiment files, or pretrained weights. The codec-conditioned refiner and the AudioCraft multi-band diffusion option have distinct model identities. The recovered pipeline and utilities do not reproduce the original complete-system results without the original compatible checkpoints and evaluation inputs.

The P2CMG code is available under the [MIT license](LICENSE). The bundled [AudioCraft fork](audiocraft/) retains the [upstream MIT notice](audiocraft/LICENSE). The [existing project audio examples](../) accompany the paper.
