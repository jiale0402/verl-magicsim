# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL.Image import Image
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from qwen_vl_utils import process_vision_info, extract_vision_info
from transformers import AutoProcessor

import torch
from pydantic import BaseModel, Field, ConfigDict
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")

IMAGE_TYPE = Union[Image]
class Message(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) 
    role: str
    content: list[dict[str, Any]]
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None

class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    tools: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    reward_scores: Dict[str, float]
    max_response_len: int = 8192
    max_model_len: int = 32768

    format_config: dict = {
        "chatml": {
            "assistant_prefix_msg": "\n<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>",
            "tool_prefix_msg": "\n<|im_start|>tool\n",
            "tool_suffix_msg": "<|im_end|>",
        },
        "qwen": {
            "assistant_prefix_msg": "\n<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>",
            "merge_tool_response": True,
            "tool_prefix_msg": "\n<|im_start|>user",
            "tool_suffix_msg": "<|im_end|>",
            "tool_response_prefix_msg": "\n<tool_response>\n",
            "tool_response_suffix_msg": "\n</tool_response>",
        }
    }

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> str:
        print("generation prompt: ", [msg.model_dump() for msg in self.messages])
        return tokenizer.apply_chat_template(  # type: ignore
            conversation=[msg.model_dump() for msg in self.messages],
            tools=[tool.model_dump() for tool in self.tools] if self.tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
    
    def get_generation_prompt(
        self, tokenizer: PreTrainedTokenizer
    ):
        all_messages = [msg.model_dump() for msg in self.messages]
        image_inputs, video_inputs = process_vision_info(all_messages)
        text_only_msgs = []
        for msg in all_messages:
            pieces = [
                c["text"]
                for c in msg["content"]
                if c.get("type") == "text"
            ]
            text_only_msgs.append({
                "role": msg["role"],
                "content": " ".join(pieces).strip()
            })
        prompt = tokenizer.apply_chat_template(
            conversation=all_messages,
            tools=[tool.model_dump() for tool in self.tools] if self.tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
        return prompt, image_inputs, video_inputs

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: list[dict[str, Any]],
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        format: Literal["chatml", "qwen"] = "qwen",
        already_over_long: bool = False,
    ) -> None:
        
        msg = Message(role="assistant", content=content, processed_pil_images=None, tool_calls=tool_calls)
        self.messages.append(msg)

        temp_conversation_for_this_turn = [{"role": msg.role, "content": msg.content}]
        if tool_calls:
            temp_conversation_for_this_turn[0]["tool_calls"] = [tc.model_dump() for tc in tool_calls]

        string_to_tokenize_for_assistant_turn: str = tokenizer.apply_chat_template(
            conversation=temp_conversation_for_this_turn,
            add_generation_prompt=False,
            tokenize=False
        )
        
        if format in self.format_config:
            prefix_msg = self.format_config[format]["assistant_prefix_msg"]
            prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
            suffix_msg = self.format_config[format]["assistant_suffix_msg"]
            suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)

            content_for_encode = string_to_tokenize_for_assistant_turn
            content_token_ids = tokenizer.encode(content_for_encode, add_special_tokens=False)
            append_token_ids: List[int]
            current_loss_mask_extension: List[int]

            if self.input_ids and self.input_ids[-len(prefix_token_ids):] == prefix_token_ids:
                append_token_ids = content_token_ids
                current_loss_mask_extension = [1] * len(content_token_ids)
            elif self.input_ids and self.input_ids[-len(suffix_token_ids):] == suffix_token_ids:
                append_token_ids = prefix_token_ids + content_token_ids
                current_loss_mask_extension = [0] * len(prefix_token_ids) + [1] * len(content_token_ids)
            elif not self.input_ids: # If this is the first message
                append_token_ids = prefix_token_ids + content_token_ids
                current_loss_mask_extension = [0] * len(prefix_token_ids) + [1] * len(content_token_ids)

            if not already_over_long:
                append_token_ids += suffix_token_ids
                current_loss_mask_extension += [0] * len(suffix_token_ids)
            
            self.input_ids += append_token_ids
            _attention_mask_to_append = [1] * len(append_token_ids)
            self.attention_mask += _attention_mask_to_append
            
            if append_token_ids:
                appended_attention_tensor = torch.tensor(_attention_mask_to_append, dtype=torch.int)
                _delta_position_ids_list = compute_position_id_with_mask(appended_attention_tensor)
                
                last_position_id = self.position_ids[-1] if self.position_ids else -1
                _new_position_ids = [pos_id + (last_position_id + 1) for pos_id in _delta_position_ids_list]
                self.position_ids += _new_position_ids
            
            self.loss_mask += current_loss_mask_extension

        else:
            raise ValueError(f"Unsupported format: {format}")

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), \
            f"Request {self.request_id} has mismatched lengths after adding assistant message: " \
            f"input_ids({len(self.input_ids)}), attention_mask({len(self.attention_mask)}), " \
            f"position_ids({len(self.position_ids)}), loss_mask({len(self.loss_mask)})"

    def add_tool_response_message(self, tokenizer: PreTrainedTokenizer, content: str, last_tool: bool, format: Literal["chatml", "qwen"] = "chatml") -> None:
        """Currently, we only support chatml format."""
        msg = Message(role="tool", content=content)
        self.messages.append(msg)
        # TODO: support other formats
        if format in self.format_config:
            merge_tool_responses = self.format_config[format].get("merge_tool_response", False)
            prefix_msg = self.format_config[format]["tool_prefix_msg"]
            prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
            suffix_msg = self.format_config[format]["tool_suffix_msg"]
            suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
            prefix_resp = self.format_config[format].get("tool_response_prefix_msg", '')
            prefix_resp_token_ids = tokenizer.encode(prefix_resp, add_special_tokens=False)
            suffix_resp = self.format_config[format].get("tool_response_suffix_msg", '')
            suffix_resp_token_ids = tokenizer.encode(suffix_resp, add_special_tokens=False)
            full_suffix_token_ids = suffix_resp_token_ids + (suffix_token_ids if last_tool or not merge_tool_responses else [])
            content_token_ids = tokenizer.encode(content, add_special_tokens=False)
            if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids or self.input_ids[-len(suffix_resp_token_ids) :] == suffix_resp_token_ids:
                append_token_ids = prefix_resp_token_ids + content_token_ids + full_suffix_token_ids
            elif self.input_ids[-len(prefix_resp_token_ids) :] == prefix_resp_token_ids:
                append_token_ids = content_token_ids + full_suffix_token_ids
            elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
                append_token_ids = prefix_token_ids + prefix_resp_token_ids + content_token_ids + full_suffix_token_ids
            else:
                raise ValueError(f"Unsupported end of message format: {tokenizer.decode(self.input_ids[-len(prefix_token_ids) :])}")
            self.input_ids += append_token_ids
            _attention_mask = [1] * len(append_token_ids)
            self.attention_mask += _attention_mask
            _delta_position_ids = compute_position_id_with_mask(torch.tensor(_attention_mask)).tolist()
            last_position_id = self.position_ids[-1]
            _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
            self.loss_mask += [0] * len(append_token_ids)
            self.position_ids += _position_ids
        else:
            raise ValueError(f"Unsupported format: {format}")
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        self.truncate_output_ids(tokenizer)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
