from typing import Any
from verl.trainer.magicros.env_init import start_magicros_env_single
from PIL import Image

REACH_SYSTEM_PROMPT = """
You are controlling a robotic arm in a 3D space.
The two images you see are the front view and side view of the robotic arm.
Based on the image and instruction, choose an action in [forward, backward, left, right, up, down].
Your goal is to reach the red cube with the fingers on the graper.
Decide the next action to take and only output one action in your final answer.<think.</think>, then the final answer inside <answer../answer>, in the format <answer>n</answer>
First output your reasoning inside <think>...</think>, then output your final answer inside <answer>...</answer>, in the format <answer>n</answer>
"""

def get_start_system_prompt(env: str, obs: list[float]) -> dict:
    match env.lower():
        case "reach":
            # reach only cares about eef position
            xyz = obs[0:3]
            prompt_str = REACH_SYSTEM_PROMPT.format(posn=xyz)
            content = {
                "role": "system",
                "content": {"type": "text", "text": prompt_str},
            }
            return content
        case _:
            raise NotImplementedError(f"Environment {env} is not supported.")


def img_obs_to_contents(obs: dict[str, Any]) -> list[dict[str, Any]]:
    front_view = Image.fromarray(obs["front_view"])
    side_view = Image.fromarray(obs["side_view"])
    return [
        {
            "role": "user",
            "content": {
                "type": "image",
                "image": front_view,
            },
        },
        {
            "role": "user",
            "content": {
                "type": "image",
                "image": side_view,
            },
        },
    ], [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": f"Current position: {obs['eef_pos']}",
            },
        }
    ]

def get_dataproto_single(
    env: str, simulator_server_url: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial_obs = start_magicros_env_single(env)
    batch_dict = {}
    contents = [get_start_system_prompt(env)]
    img_contents, text_contents = img_obs_to_contents(initial_obs)
    contents += text_contents
    batch_dict["prompt"] = text_contents
    batch_dict["multi_modal_inputs"] = img_contents
    batch_dict["extra_info"] = {
        "tools_kwargs": {
            "env_id": env,
            "simulator_server_url": simulator_server_url,
        }
    }
    return text_contents, img_contents
