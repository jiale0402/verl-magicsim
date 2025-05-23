import argparse
import os
import re
import textwrap
from io import BytesIO

import datasets
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA_SOURCE_NAME = "openai/gsm8k"
INSTRUCTION_FOLLOWING_TEXT = "Let's think step by step and output the final answer after `####`."
DEFAULT_IMAGE_KEY = "images"

def extract_solution(solution_str):
    """Extracts the final numerical solution from the GSM8K answer string."""
    solution_match = re.search(r"####\s*(-?[\d\.,]+)", solution_str)
    assert solution_match is not None, f"Solution not found in: {solution_str}"
    final_solution = solution_match.group(1).replace(",", "")
    return final_solution

def text_to_pil_image(text_content, width=400, font_size=24, padding=10,
                      bg_color="white", text_color="black", font_path=None):
    """
    Converts a string of text into a downscaled PIL Image.
    """
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            if font_path:
                print(f"Warning: Font at '{font_path}' not found. Using default.")
            font = ImageFont.load_default()
    except IOError:
        print(f"Warning: Font at '{font_path}' caused IOError. Using default.")
        font = ImageFont.load_default()

    # estimate chars per line
    char_width = getattr(font, "getlength", lambda t: len(t) * font_size * 0.6)("x") or font_size * 0.6
    chars_per_line = max(1, int((width - 2 * padding) / char_width))

    wrapper = textwrap.TextWrapper(width=chars_per_line, break_long_words=True, replace_whitespace=False)
    lines = wrapper.wrap(text=text_content) or [" "]

    # compute line height
    try:
        tmp = ImageDraw.Draw(Image.new("RGB", (1,1)))
        bbox = tmp.textbbox((0,0), "Tg", font=font)
        line_height = (bbox[3] - bbox[1]) + 5
    except AttributeError:
        line_height = font_size + 5
    if line_height <= 5:
        line_height = font_size + 10

    img_height = max(50, len(lines) * line_height + 2 * padding)
    img = Image.new("RGB", (width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=text_color)
        y += line_height

    return img

def make_map_fn(split, image_key, font_path_for_image):
    """
    Creates the mapping function for dataset processing.
    Also saves out the first NUM_SAVE images under /images/{split}/ for inspection.
    """
    NUM_SAVE = 5
    inspect_dir = os.path.join("images", split)
    os.makedirs(inspect_dir, exist_ok=True)

    def process_fn(example, idx):
        question_raw = example.pop("question")
        answer_raw   = example.pop("answer")

        # render downscaled image
        pil_img = text_to_pil_image(
            question_raw,
            width=400,
            font_size=24,
            font_path=font_path_for_image
        )

        # save first few for inspection
        if idx < NUM_SAVE:
            out_path = os.path.join(inspect_dir, f"{split}_{idx}.png")
            pil_img.save(out_path, format="PNG")
            print(f"[inspect] saved {out_path}")

        # convert to bytes for dataset
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        solution = extract_solution(answer_raw)
        prompt_content = "<image>\n" + INSTRUCTION_FOLLOWING_TEXT

        return {
            "data_source": DATA_SOURCE_NAME,
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a math expert. You are given a question and you need to solve it step by step. "
                        "Reasoning step by step before any tool call. "
                        "You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
                        "before generate final answer at least once and refine your answer if necessary. "
                        "Put your final answer in the format of `#### <answer>`."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt_content,
                },
            ],
            image_key: [{"bytes": image_bytes}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question_text_original": question_raw,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "calc_gsm8k_reward": {
                        "create_kwargs": {"ground_truth": solution},
                    },
                },
            },
        }
    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess GSM8K to Parquet with downscaled images and inspection dumps"
    )
    parser.add_argument("--local_dir", default="~/a100/data/gsm8k_images_multiturn")
    parser.add_argument("--font_path", default=None)
    parser.add_argument("--image_key", default=DEFAULT_IMAGE_KEY)
    parser.add_argument("--num_train_samples", type=int, default=128)
    parser.add_argument("--num_test_samples",  type=int, default=32)
    args = parser.parse_args()

    raw = datasets.load_dataset(DATA_SOURCE_NAME, "main")
    train_ds, test_ds = raw["train"], raw["test"]

    def subset_and_process(ds, split, n):
        ds_sub = ds.select(range(min(n, len(ds)))) if n>0 else None
        if not ds_sub:
            print(f"Skipping {split}")
            return
        print(f"{split}: processing {len(ds_sub)} samples")
        map_fn = make_map_fn(split, args.image_key, args.font_path)
        out = ds_sub.map(function=map_fn, with_indices=True, num_proc=max(1, os.cpu_count()//2))
        out_path = os.path.expanduser(args.local_dir)
        os.makedirs(out_path, exist_ok=True)
        parquet = os.path.join(out_path, f"{split}.parquet")
        print(f"Saving {split} → {parquet}")
        out.to_parquet(parquet)

    subset_and_process(train_ds, "train", args.num_train_samples)
    subset_and_process(test_ds,  "test",  args.num_test_samples)

    print("Done!")