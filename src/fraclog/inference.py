from __future__ import annotations

import copy
from pathlib import Path


def build_mm_messages(record: dict, image_path: str | Path) -> list[dict]:
    messages = []
    image_inserted = False
    has_any_marker = any(message["role"] == "user" and "<image>" in message["content"] for message in record["messages"])
    for message in record["messages"]:
        if message["role"] == "user":
            content = message["content"]
            has_marker = "<image>" in content
            text = content.replace("<image>", "").strip()
            should_insert_image = not image_inserted and (has_marker or not has_any_marker)
            if should_insert_image:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": text},
                        ],
                    }
                )
                image_inserted = True
            else:
                messages.append({"role": "user", "content": text})
        else:
            messages.append(message)
    return messages


def append_final_user_instruction(messages: list[dict], instruction: str | None) -> list[dict]:
    if not instruction:
        return copy.deepcopy(messages)
    updated = copy.deepcopy(messages)
    for index in range(len(updated) - 1, -1, -1):
        message = updated[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = f"{part.get('text', '').rstrip()}\n{instruction}"
                    return updated
            content.append({"type": "text", "text": instruction})
            return updated
        message["content"] = f"{str(content or '').rstrip()}\n{instruction}"
        return updated
    return updated
