import json
import tempfile
import unittest
from pathlib import Path

from fraclog.data import (
    convert_preference_record,
    convert_sft_record,
    load_jsonl,
    normalize_image_ref,
    prompt_without_final_assistant,
    resolve_image_path,
    write_jsonl,
)


class DataConversionTests(unittest.TestCase):
    def test_resolve_image_path_repairs_pic_4_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "pic"
            image_dir.mkdir()
            image = image_dir / "pic_1.png"
            image.write_bytes(b"png")

            resolved = resolve_image_path("pic_4/pic_1.png", root)

            self.assertEqual(resolved, image)

    def test_normalize_image_ref_uses_local_image_dir_and_filename(self):
        self.assertEqual(normalize_image_ref("pic_4/sample_0001.png"), "pic/sample_0001.png")
        self.assertEqual(normalize_image_ref("pic/pic_1.png"), "pic/pic_1.png")

    def test_convert_sft_record_preserves_messages_and_normalizes_image_refs(self):
        record = {
            "messages": [
                {"role": "user", "content": "<image>请分析裂缝数量。"},
                {"role": "assistant", "content": "图中有2条裂缝。"},
            ],
            "images": ["pic_4/pic_1.png"],
        }

        converted = convert_sft_record(record)

        self.assertEqual(converted["messages"], record["messages"])
        self.assertEqual(converted["images"], ["pic/pic_1.png"])

    def test_convert_preference_record_uses_rejected_response_format(self):
        record = {
            "messages": [
                {"role": "system", "content": "你是专家。"},
                {"role": "user", "content": "<image>\n请告诉我裂缝数量。"},
            ],
            "chosen": "图中有2条裂缝。",
            "rejected": "图中有1条裂缝。",
            "images": ["pic_4/pic_2.png"],
        }

        converted = convert_preference_record(record)

        self.assertEqual(converted["images"], ["pic/pic_2.png"])
        self.assertEqual(converted["messages"][-1], {"role": "assistant", "content": "图中有2条裂缝。"})
        self.assertEqual(converted["rejected_response"], "图中有1条裂缝。")
        self.assertNotIn("chosen", converted)
        self.assertNotIn("rejected", converted)

    def test_jsonl_round_trip_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            rows = [{"text": "裂缝数量：2条"}, {"text": "无明显裂缝"}]

            write_jsonl(path, rows)
            loaded = load_jsonl(path)

            self.assertEqual(loaded, rows)
            raw = path.read_text(encoding="utf-8")
            self.assertIn("裂缝数量", raw)
            self.assertEqual(len(raw.splitlines()), 2)

    def test_load_jsonl_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bom.jsonl"
            path.write_text('{"text": "裂缝"}\n', encoding="utf-8-sig")

            self.assertEqual(load_jsonl(path), [{"text": "裂缝"}])

    def test_prompt_without_final_assistant_preserves_prior_context(self):
        messages = [
            {"role": "user", "content": "<image>第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "第二答"},
        ]

        prompt = prompt_without_final_assistant(messages)

        self.assertEqual(
            prompt,
            [
                {"role": "user", "content": "<image>第一问"},
                {"role": "assistant", "content": "第一答"},
                {"role": "user", "content": "第二问"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
