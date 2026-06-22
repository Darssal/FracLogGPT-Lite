from __future__ import annotations

import unittest

from fraclog.inference import append_final_user_instruction, build_mm_messages


class InferenceMessageTests(unittest.TestCase):
    def test_build_mm_messages_inserts_single_image_for_multiturn_prompt(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": "<image>first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ]
        }
        messages = build_mm_messages(record, "pic/example.png")

        image_blocks = [
            block
            for message in messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if block.get("type") == "image"
        ]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(messages[0]["content"][0], {"type": "image", "image": "pic/example.png"})
        self.assertEqual(messages[2]["content"], "second question")

    def test_build_mm_messages_inserts_image_for_first_user_when_marker_is_absent(self) -> None:
        record = {"messages": [{"role": "user", "content": "question without marker"}]}
        messages = build_mm_messages(record, "pic/example.png")
        self.assertEqual(messages[0]["content"][0], {"type": "image", "image": "pic/example.png"})

    def test_build_mm_messages_uses_only_first_marker(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": "<image>first question"},
                {"role": "user", "content": "<image>second question"},
            ]
        }
        messages = build_mm_messages(record, "pic/example.png")
        image_counts = [
            sum(1 for block in message["content"] if block.get("type") == "image")
            if isinstance(message.get("content"), list)
            else 0
            for message in messages
        ]
        self.assertEqual(image_counts, [1, 0])

    def test_append_final_user_instruction_updates_last_text_user_message(self) -> None:
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "context"},
            {"role": "user", "content": "final"},
        ]

        updated = append_final_user_instruction(messages, "answer directly")

        self.assertEqual(updated[-1]["content"], "final\nanswer directly")
        self.assertEqual(messages[-1]["content"], "final")

    def test_append_final_user_instruction_updates_multimodal_text_part(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "pic/a.png"},
                    {"type": "text", "text": "final"},
                ],
            }
        ]

        updated = append_final_user_instruction(messages, "structured only")

        self.assertEqual(updated[0]["content"][1]["text"], "final\nstructured only")


if __name__ == "__main__":
    unittest.main()
