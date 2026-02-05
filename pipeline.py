import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from diffusers import DiffusionPipeline


@dataclass
class SamplingConfig:
    greedy: bool = False
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    temperature: float = 1.0
    out_seq_length: int = 40960


DEFAULT_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)

ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


SPLIT_PROMPT_TEMPLATE = """You are an expert in product planning and image-generation prompts. Structure and split the previous model output as follows:
1. Identify all distinct products or product bundles mentioned.
2. For each product, produce an independent, complete, English descriptive prompt that can be directly used by an image generation model.
3. Each prompt must include: product category, material/texture, color palette, key design elements, usage/placement scene, and photographic style.
4. If the original text lacks attributes, reasonably infer them but do not fabricate specific brands.
5. Output a JSON array where each element includes: "title" (product name) and "prompt" (image-generation prompt).
6. Output JSON only, no extra explanation.

Previous model output: {model_output}
"""


def load_history(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("history json must be a list of items")
    for item in data:
        if "image" not in item or "description" not in item:
            raise ValueError("each history item must have 'image' and 'description'")
    return data


def build_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You will see the user's shopping history (product images + descriptions). "
                "Summarize the user's shopping preferences and predict the most likely next "
                "product type they will purchase, with detailed descriptions. Output in English."
            ),
        }
    ]
    for item in history:
        content.append({"type": "image", "image": item["image"]})
        content.append({"type": "text", "text": item["description"]})
    return [{"role": "user", "content": content}]


def generate_preference(
    history: List[Dict[str, Any]],
    model_name: str,
    sampling: SamplingConfig,
) -> str:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    messages = build_messages(history)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generate_kwargs = {
        "max_new_tokens": sampling.out_seq_length,
        "do_sample": not sampling.greedy,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "repetition_penalty": sampling.repetition_penalty,
        "temperature": sampling.temperature,
    }

    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    result = output_text[0] if output_text else ""
    del model
    del processor
    torch.cuda.empty_cache()
    return result


def split_products(preference_output: str, model_name: str) -> str:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    prompt = SPLIT_PROMPT_TEMPLATE.format(model_output=preference_output)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    result = output_text[0] if output_text else "[]"
    del model
    del processor
    torch.cuda.empty_cache()
    return result


def generate_images(
    prompts: List[Dict[str, str]],
    model_name: str,
    aspect_ratio: str,
    output_dir: str,
    negative_prompt: str,
    seed: int,
) -> List[str]:
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"unsupported aspect ratio: {aspect_ratio}")

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    width, height = ASPECT_RATIOS[aspect_ratio]
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for index, item in enumerate(prompts, start=1):
        prompt = item["prompt"]
        generator = torch.Generator(device=device).manual_seed(seed + index)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=generator,
        ).images[0]
        output_path = os.path.join(output_dir, f"product_{index}.png")
        image.save(output_path)
        image_paths.append(output_path)

    del pipe
    torch.cuda.empty_cache()
    return image_paths


def parse_products(output_text: str) -> List[Dict[str, str]]:
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse product JSON from model output")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL preference + generation pipeline")
    parser.add_argument("--history", default="sample_history.json", help="Path to shopping history JSON")
    parser.add_argument("--vl-model", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--image-model", default="Qwen/Qwen-Image-2512")
    parser.add_argument("--aspect", default="16:9")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sampling = SamplingConfig(
        greedy=os.getenv("greedy", "false").lower() == "true",
        top_p=float(os.getenv("top_p", "0.95")),
        top_k=int(os.getenv("top_k", "20")),
        repetition_penalty=float(os.getenv("repetition_penalty", "1.0")),
        presence_penalty=float(os.getenv("presence_penalty", "0.0")),
        temperature=float(os.getenv("temperature", "1.0")),
        out_seq_length=int(os.getenv("out_seq_length", "40960")),
    )

    history = load_history(args.history)
    preference_output = generate_preference(history, args.vl_model, sampling)
    print("Preference output:\n", preference_output)

    product_json = split_products(preference_output, args.vl_model)
    print("Split products JSON:\n", product_json)

    products = parse_products(product_json)
    image_paths = generate_images(
        products,
        args.image_model,
        args.aspect,
        args.output_dir,
        args.negative_prompt,
        args.seed,
    )
    print("Generated images:", image_paths)


if __name__ == "__main__":
    main()
