# -*- coding: utf-8 -*-
"""
Guohua (Chinese ink painting) -> English music prompt, multi-turn with Qwen2.5-VL.

Requirements:
  pip install "git+https://github.com/huggingface/transformers" accelerate
  pip install "qwen-vl-utils[decord]==0.0.8"  # or: pip install qwen-vl-utils
"""

from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class GuohuaToMusicPromptor:
    """
    A reusable component that:
      1) loads Qwen2.5-VL-7B-Instruct from a local path,
      2) runs a carefully-designed, English-only, multi-turn dialogue about a Chinese ink painting (国画),
      3) returns a structured result + a final concise music prompt.

    Usage:
        g = GuohuaToMusicPromptor(model_dir="./models/Qwen2.5-VL-7B-Instruct")
        result = g.run("/abs/path/to/guohua.jpg")
        print(result["final_prompt"])
    """

    def __init__(
        self,
        model_dir: str = "./models/Qwen2.5-VL-7B-Instruct",
        device_preference: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_image_each_turn: bool = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        use_flash_attn2: bool = True,
    ):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_image_each_turn = repeat_image_each_turn

        # Choose device
        if device_preference:
            self.device = device_preference
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        if use_flash_attn2:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype="auto",
                device_map="auto",
            )

        # Load processor with pixel constraints (perf/cost trade-off)
        self.processor = AutoProcessor.from_pretrained(
            model_dir, min_pixels=min_pixels, max_pixels=max_pixels
        )

        # A compact system instruction to keep answers in English and music-focused
        self.system_message = {
            "role": "system",
            "content": (
                "You are a senior music prompt engineer specialized in translating Chinese ink-wash paintings "
                "(guóhuà) into concise, production-ready English prompts for text-to-music systems. "
                "ALWAYS answer in English. Be culturally aware and musically precise. "
                "Avoid Chinese words; write only in English. Prefer bullet lists and compact phrasing. "
                "Favor practical directives over verbose prose."
            ),
        }

        # Multi-turn plan (each step is an English-only prompt).
        self.turns: List[str] = [
            # 1) Scene & composition
            (
                "Step 1 — Scene & Composition:\n"
                "Given the painting, identify the scene and setting (landscape/city/temple/river/mountain), "
                "season/time/weather, vantage point, and 5–10 key visual motifs (e.g., mist, peaks, water flow, "
                "boats, pavilions, cranes, scholars). Output concise bullets: "
                "• scene • season/time • weather • perspective • key motifs/actions."
            ),
            # 2) Brushwork & aesthetics -> musical analogues
            (
                "Step 2 — Brushwork & Aesthetics:\n"
                "Describe ink techniques (wash gradations, dry vs. wet strokes, line delicacy/boldness), density, "
                "negative space, balance, rhythm, and stylistic school (e.g., shan shui, bird-and-flower). "
                "Map each major visual cue to a musical analogue (texture density, register, orchestration breadth, "
                "motion) in a short two-column listing: Visual → Musical analogue."
            ),
            # 3) Mood & 3-act emotion arc
            (
                "Step 3 — Mood & Narrative Arc:\n"
                "Infer the core mood and a 3-phase emotional arc (beginning → middle → end). "
                "Provide 3–5 evocative adjectives per phase."
            ),
            # 4) Instrumentation mapping (Chinese + optional Western)
            (
                "Step 4 — Instrumentation:\n"
                "Propose instruments to express the motifs, prioritizing Chinese instruments (at least one of: "
                "guzheng, erhu, dizi, xiao, pipa, ruan, yangqin, sheng, suona, konghou). "
                "You may add strings/pads/percussion. For each: role, register, and techniques/articulations."
            ),
            # 5) Tempo & rhythm
            (
                "Step 5 — Tempo & Rhythm:\n"
                "Recommend BPM range, time signature (e.g., 4/4, 6/8), rhythmic cells/ostinati, swing vs. straight, "
                "and rubato usage. Keep it practical for music generation control."
            ),
            # 6) Tonality & harmony
            (
                "Step 6 — Tonality & Harmony:\n"
                "Choose a primary Chinese pentatonic mode (Gong, Shang, Jue, Zhi, or Yu) and optional secondary mode. "
                "State pitch center (e.g., D Gong ≈ D major pentatonic). "
                "Give harmonic hints (open fifths, parallel fourths, drones, sus2/sus4), avoiding dense functional harmony."
            ),
            # 7) Texture, space, production
            (
                "Step 7 — Texture & Space:\n"
                "Plan layering and dynamics over time, reverb/space (e.g., mountain valley, temple hall), "
                "ambience SFX (wind, stream, birds), and brief mixing notes (warm midrange, smooth highs, light low end)."
            ),
            # 8) Negative constraints
            (
                "Step 8 — Negative Constraints:\n"
                "List what to avoid to keep authenticity (e.g., EDM risers, heavy rock drum kits, distorted guitar walls, "
                "atonality unless intended, harsh metallic cymbal wash, overly busy counterpoint)."
            ),
            # 9) Final
            (
                "Step 9 — FINAL ASSEMBLY:\n"
                "Synthesize a single production-ready English prompt (80–150 words) for a text-to-music model. "
                "Include BPM, meter, mode, instrumentation, texture plan, mood adjectives, ambience, and brief mixing notes. "
                "Also produce a separate negative prompt line. "
                "Integrate all the above information and output it into smooth natural language"
                "And it is required that the main style of the music be traditional Chinese music"
                "No need to add any other information, just output the prompt"
                "Using an entire paragraph to output the description content, without line breaks, bullet points, or paragraph divisions"
                "No markdown, no backticks, no extra text"
            ),
        ]

    # ---------------- Core generation helpers ---------------- #

    def _generate_once(self, messages: List[Dict[str, Any]]) -> str:
        """Run a single Qwen chat generation and return the assistant text."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Put inputs on a device that exists even with device_map='auto'
        # 'cuda' moves tensors to GPU, otherwise CPU is fine.
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        # Trim the echoed prompt
        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated)
        ]
        text_out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text_out.strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract first JSON object from model text."""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            # Try to repair common trailing commas or bad quotes if necessary later
            try:
                snippet = re.sub(r",\s*}", "}", snippet)
                snippet = re.sub(r",\s*\]", "]", snippet)
                return json.loads(snippet)
            except Exception:
                return None

    # ---------------- Public pipeline ---------------- #

    def run(
        self,
        image_path: str,
        extra_turns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full multi-turn pipeline and return:
            {
              "trace": [ {"step": 1, "prompt": "...","answer": "..."}, ... ],
              "final": { ... JSON from Step 9 ... }
            }
        """
        messages: List[Dict[str, Any]] = [self.system_message]

        # First turn includes the image
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": self.turns[0]},
            ],
        })
        trace: List[Dict[str, Any]] = []

        # Step 1
        ans = self._generate_once(messages)
        trace.append({"step": 1, "prompt": self.turns[0], "answer": ans})
        messages.append({"role": "assistant", "content": ans})

        # Remaining planned turns
        planned = self.turns[1:]
        if extra_turns:
            planned = self.turns[1:-1] + extra_turns + [self.turns[-1]]

        step_index = 2
        for turn in planned:
            turn_message: Dict[str, Any] = {"role": "user", "content": []}

            if self.repeat_image_each_turn:
                turn_message["content"].append({"type": "image", "image": f"file://{image_path}"})
            turn_message["content"].append({"type": "text", "text": turn})
            messages.append(turn_message)

            ans = self._generate_once(messages)
            trace.append({"step": step_index, "prompt": turn, "answer": ans})
            messages.append({"role": "assistant", "content": ans})
            step_index += 1

        # Try to parse final JSON from the last answer
        final_answer_text = trace[-1]["answer"] if trace else ""
        final_prompt = final_answer_text.strip()

        return {"trace": trace, "final_prompt": final_prompt}


# ---------------- Convenience demo ---------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a guohua image.")
    parser.add_argument("--model_dir", default="./models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--flash2", action="store_true", help="Enable FlashAttention-2.")
    args = parser.parse_args()

    promptor = GuohuaToMusicPromptor(
        model_dir=args.model_dir,
        use_flash_attn2=args.flash2,
        repeat_image_each_turn=False,  # True = reattach image every turn (more VRAM)
    )
    result = promptor.run(args.image)

    print("\n=== FINAL MUSIC PROMPT ===")
    print(result["final_prompt"])