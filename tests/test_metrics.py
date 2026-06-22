import unittest

from fraclog.metrics import (
    compute_structured_text_metrics,
    count_category,
    compute_count_metrics,
    compute_hallucination_metrics,
    extract_location_labels,
    extract_morphology_labels,
    parse_fracture_count,
    rouge_l_score,
    simple_bleu,
)


class MetricsTests(unittest.TestCase):
    def test_parse_fracture_count_from_chinese_and_english(self):
        self.assertEqual(parse_fracture_count("图像中清晰可见2条主要裂缝"), 2)
        self.assertEqual(parse_fracture_count("未见明显裂缝"), 0)
        self.assertEqual(parse_fracture_count("no visible fractures are observed"), 0)
        self.assertEqual(parse_fracture_count("there are three fractures"), 3)
        self.assertEqual(parse_fracture_count("至少4条裂缝"), 4)

    def test_parse_fracture_count_ignores_fracture_indices(self):
        text = "可以确定图中有2条清晰可见的裂缝。裂缝1的倾角约为20°，裂缝2的倾角约为13°。"

        self.assertEqual(parse_fracture_count(text), 2)

    def test_count_category_maps_three_or_more_to_single_class(self):
        self.assertEqual(count_category(0), "0")
        self.assertEqual(count_category(1), "1")
        self.assertEqual(count_category(2), "2")
        self.assertEqual(count_category(3), ">=3")
        self.assertEqual(count_category(9), ">=3")

    def test_compute_count_metrics_returns_macro_scores(self):
        refs = [0, 1, 2, 3]
        preds = [0, 2, 2, 3]

        metrics = compute_count_metrics(refs, preds)

        self.assertAlmostEqual(metrics["count_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["count_precision"], 0.625)
        self.assertAlmostEqual(metrics["count_recall"], 0.75)
        self.assertAlmostEqual(metrics["count_f1"], 2 / 3)

    def test_compute_hallucination_metrics(self):
        refs = [0, 0, 1, 2, 3]
        preds = [0, 2, 3, 1, 3]

        metrics = compute_hallucination_metrics(refs, preds)

        self.assertAlmostEqual(metrics["zero_fracture_hallucination_rate"], 0.5)
        self.assertAlmostEqual(metrics["over_count_rate"], 0.4)
        self.assertAlmostEqual(metrics["under_count_rate"], 0.2)

    def test_text_similarity_scores_are_bounded(self):
        reference = "图中有2条裂缝，位于上部和下部，呈白色双曲线。"
        prediction = "图中有2条裂缝，呈白色双曲线。"

        bleu = simple_bleu(prediction, reference)
        rouge = rouge_l_score(prediction, reference)

        self.assertGreater(bleu, 0)
        self.assertLessEqual(bleu, 1)
        self.assertGreater(rouge, 0)
        self.assertLessEqual(rouge, 1)

    def test_extract_structured_location_and_morphology_labels(self):
        text = "裂缝位置：第1条裂缝位于图像上部1/3以内位置。空间分布：垂直间距约为图像高度的0.18。形态特征：白色双曲线，S形缝隙，曲线对称性较大。"

        self.assertEqual(extract_location_labels(text), frozenset({"upper_third", "rel_0.18"}))
        self.assertEqual(extract_morphology_labels(text), frozenset({"white", "hyperbola", "s_shape", "symmetric"}))

    def test_compute_structured_text_metrics_returns_automatic_loc_and_morph_acc(self):
        refs = [
            "空间分布：垂直间距约为图像高度的0.18。形态特征：白色双曲线，S形缝隙，曲线对称性较大。",
            "裂缝位置：第1条裂缝位于图像下部1/3以内位置。形态特征：白色双曲线。",
        ]
        preds = [
            "裂缝空间分布：间距约为图像高度的0.18。裂缝形态：呈现白色双曲线、S形缝隙，对称性较大。",
            "裂缝位置：第1条裂缝位于图像上部1/3以内位置。形态特征：没有明确说明双曲线。",
        ]

        metrics = compute_structured_text_metrics(refs, preds)

        self.assertAlmostEqual(metrics["loc_acc"], 0.5)
        self.assertAlmostEqual(metrics["morph_acc"], 0.5)
        self.assertEqual(metrics["loc_evaluable"], 2)
        self.assertEqual(metrics["morph_evaluable"], 2)

    def test_extracts_real_chinese_fracture_labels(self):
        text = (
            "通过分析图像中的白色双曲线特征，可以确定图中有2条清晰可见的裂缝。"
            "第1条裂缝位于图像上部1/3以内位置。第2条裂缝位于图像上部1/3以内位置。"
            "两条裂缝呈S形缝隙，曲线对称性明显，并在水平方向上平行延伸，"
            "垂直间距约为图像高度的0.18。"
        )

        self.assertEqual(parse_fracture_count(text), 2)
        self.assertEqual(extract_location_labels(text), frozenset({"upper_third", "rel_0.18"}))
        self.assertEqual(
            extract_morphology_labels(text),
            frozenset({"white", "hyperbola", "s_shape", "symmetric", "parallel"}),
        )

    def test_real_chinese_count_parser_ignores_ordinal_fracture_indices(self):
        text = "第2条裂缝位于图像上部1/3以内位置，倾角约为18.91°。"

        self.assertIsNone(parse_fracture_count(text))

    def test_real_chinese_count_parser_accepts_hyperbola_count_as_fracture_evidence(self):
        self.assertEqual(parse_fracture_count("图像下部清晰可见2条白色双曲线特征。"), 2)
        self.assertEqual(parse_fracture_count("可以看到图像下部有两条明显的白色双曲线。"), 2)

    def test_parse_fracture_count_accepts_vcd_diagnostic_count_phrasing(self):
        self.assertEqual(parse_fracture_count("\u56fe\u4e2d\u67092\u6761\u6e05\u6670\u53ef\u89c1\u7684\u767d\u8272\u53cc\u66f2\u7ebf\u7279\u5f81\u3002"), 2)
        self.assertEqual(
            parse_fracture_count(
                "\u56fe\u50cf\u4e2d\u6e05\u6670\u53ef\u89c12\u6761\u4e3b\u8981\u767d\u8272\u5782\u76f4\u7ebf\u6761\u3002"
                "\u7b2c1\u6761\u88c2\u7f1d\u4f4d\u4e8e\u4e0a\u90e8\uff0c\u7b2c2\u6761\u88c2\u7f1d\u4f4d\u4e8e\u4e0b\u90e8\u3002"
            ),
            2,
        )
        self.assertEqual(parse_fracture_count("\u6211\u53ea\u770b\u52301\u6761\u6e05\u6670\u7684\u767d\u8272\u53cc\u66f2\u7ebf\u3002"), 1)
        self.assertEqual(parse_fracture_count("\u56fe\u50cf\u4e2d\u6709\u4e00\u4e2a\u660e\u663e\u7684\u767d\u8272\u53cc\u66f2\u7ebf\u7279\u5f81\u3002"), 1)

    def test_parse_fracture_count_prefers_explicit_count_over_no_spacing_phrase(self):
        text = (
            "\u7531\u4e8e\u53ea\u6709\u4e00\u6761\u88c2\u7f1d\uff0c"
            "\u65e0\u6cd5\u8ba1\u7b97\u95f4\u8ddd\uff0c"
            "\u7a7a\u95f4\u5206\u5e03\u7279\u70b9\u63cf\u8ff0\u4e3a\u4ec5\u5206\u5e031\u6761\u88c2\u7f1d\u3002\n"
            "</think>\n\n"
            "\u88c2\u7f1d\u6570\u91cf\uff1a1\n"
            "\u7a7a\u95f4\u5206\u5e03\uff1a\u4ec5\u5206\u5e031\u6761\u88c2\u7f1d"
        )

        self.assertEqual(parse_fracture_count(text), 1)



if __name__ == "__main__":
    unittest.main()
