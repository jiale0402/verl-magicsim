import logging
import os
import requests
from typing import Any, Optional, Tuple
from uuid import uuid4
import pickle
import base64

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def encode_data(data):
    """
    Encode data using base64 encoding and pickle serialization.

    Args:
        data: The data to be encoded.

    Returns:
        str: The base64 encoded string representation of the data.
    """
    # Serialize the data using pickle
    serialized_data = pickle.dumps(data)
    # Encode the serialized data using base64
    encoded_data = base64.b64encode(serialized_data).decode("utf-8")
    return encoded_data


def decode_data(encoded_data):
    """
    Decode data from a base64 encoded string and deserialize it using pickle.

    Args:
        encoded_data (str): The base64 encoded string representation of the data.

    Returns:
        The original data.
    """
    # Decode the base64 encoded string
    decoded_data = base64.b64decode(encoded_data.encode("utf-8"))
    # Deserialize the data using pickle
    data = pickle.loads(decoded_data)
    return data

def format_string_feedback(
    obs: dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> str:
    """
    Format feedback data.

    Args:
        obs (dict): Observations.
        reward (float): Reward.
        terminated (bool): Termination status.
        truncated (bool): Truncation status.
        info (dict): Additional information.

    Returns:
        dict: Formatted feedback data.
    """
    return {
        "observations": obs,
        "rewards": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


class MagicsimReachRestAPITool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._endpoint = config["endpoint"].rstrip("/")
        self._instance_dict: dict[str, dict] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        env_id: str,  # you can ignore this or repurpose it
        instance_id: Optional[str] = None,
        **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        resp = requests.post(f"{self._endpoint}/reset")
        resp.raise_for_status()
        raw = resp.json().get("data")
        obs, _ = decode_data(raw)

        self._instance_dict[instance_id] = {
            "env_id": env_id,
            "obs": obs,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        return instance_id

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> Tuple[Any, float, dict]:
        record = self._instance_dict[instance_id]

        action = int(parameters["action"])

        param_encoded = encode_data({"action": action})
        resp = requests.post(f"{self._endpoint}/step", json={"param": param_encoded})
        resp.raise_for_status()
        raw = resp.json().get("data")

        obs, reward, _, _, _ = decode_data(raw)
        feedback = {
            "observations": obs,
            "rewards": reward,
        }
        record.update(feedback)
        return f"Env ID {record['env_id']} received reward: {reward}", reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)