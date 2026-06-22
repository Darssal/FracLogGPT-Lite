from __future__ import annotations

import unittest

try:
    import torch

    from fraclog.vcd import BatchedVCDLogitsProcessor, VCDConfig, append_generated_text_token, visual_contrastive_generate_batched
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class VCDInputUpdateTests(unittest.TestCase):
    def test_batched_vcd_logits_processor_fuses_scores_for_both_branches(self) -> None:
        processor = BatchedVCDLogitsProcessor(alpha=0.5)
        input_ids = torch.tensor([[1, 2], [1, 2]])
        scores = torch.tensor([[1.0, 3.0, 0.0], [0.5, 2.0, 4.0]])

        fused = processor(input_ids, scores)

        expected = scores[0:1] + 0.5 * (scores[0:1] - scores[1:2])
        self.assertTrue(torch.allclose(fused[0:1], expected))
        self.assertTrue(torch.allclose(fused[1:2], expected))
        self.assertFalse(torch.equal(fused, scores))

    def test_batched_vcd_logits_processor_requires_original_and_perturbed_pair(self) -> None:
        processor = BatchedVCDLogitsProcessor(alpha=0.5)

        with self.assertRaises(ValueError):
            processor(torch.tensor([[1], [1], [1]]), torch.zeros((3, 4)))

    def test_append_generated_text_token_keeps_qwen35_mm_fields_aligned(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
        }

        append_generated_text_token(inputs, torch.tensor([[13]]))

        self.assertEqual(inputs["input_ids"].tolist(), [[10, 11, 12, 13]])
        self.assertEqual(inputs["attention_mask"].tolist(), [[1, 1, 1, 1]])
        self.assertEqual(inputs["mm_token_type_ids"].tolist(), [[0, 1, 0, 0]])
        self.assertEqual(inputs["input_ids"].shape[-1], inputs["attention_mask"].shape[-1])
        self.assertEqual(inputs["input_ids"].shape[-1], inputs["mm_token_type_ids"].shape[-1])

    def test_visual_contrastive_generate_batched_uses_one_cache_after_first_step(self) -> None:
        class Output:
            def __init__(self, logits: torch.Tensor, past_key_values: tuple[str, ...]) -> None:
                self.logits = logits
                self.past_key_values = past_key_values

        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs: object) -> dict[str, object]:
                return {"input_ids": input_ids, **kwargs}

            def __call__(self, **kwargs: object) -> Output:
                input_ids = kwargs["input_ids"]
                assert isinstance(input_ids, torch.Tensor)
                self.calls.append(
                    {
                        "input_len": input_ids.shape[-1],
                        "has_past": kwargs.get("past_key_values") is not None,
                        "has_pixels": kwargs.get("pixel_values") is not None,
                        "use_cache": kwargs.get("use_cache"),
                    }
                )
                logits = torch.zeros((input_ids.shape[0], input_ids.shape[-1], 8), dtype=torch.float32)
                logits[:, -1, 3] = 10.0
                return Output(logits, ("cached",))

        paired_inputs = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 0], [0, 1, 0]]),
            "pixel_values": torch.ones((2, 3, 2, 2)),
        }
        model = FakeModel()

        generated = visual_contrastive_generate_batched(
            model,
            processor=object(),
            paired_inputs=paired_inputs,
            config=VCDConfig(max_new_tokens=2, temperature=0.0),
            eos_token_id=99,
        )

        self.assertEqual(generated, [3, 3])
        self.assertEqual([call["input_len"] for call in model.calls], [3, 1])
        self.assertEqual([call["has_past"] for call in model.calls], [False, True])
        self.assertEqual([call["has_pixels"] for call in model.calls], [True, False])
        self.assertTrue(all(call["use_cache"] for call in model.calls))


if __name__ == "__main__":
    unittest.main()
