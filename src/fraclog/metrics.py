from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


COUNT_LABELS = ["0", "1", "2", ">=3"]

_CN_NUMBERS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

_EN_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def parse_fracture_count(text: str) -> int | None:
    normalized = text.strip().lower()
    explicit_count = re.search(
        r"\u88c2\u7f1d(?:\u6570\u91cf|\u6570)?\s*[:\uff1a]\s*(?<!\u7b2c)(\d+)",
        text,
    )
    if explicit_count:
        return int(explicit_count.group(1))
    zero_patterns = [
        r"未见[^。；,\n]*裂缝",
        r"未发现[^。；,\n]*裂缝",
        r"无[^。；,\n]*裂缝",
        r"没有[^。；,\n]*裂缝",
        r"no\s+(visible\s+)?fractures?",
        r"zero\s+fractures?",
    ]
    if any(re.search(pattern, normalized) for pattern in zero_patterns):
        return 0

    numeric_patterns = [
        r"(?:至少|约|大约|共|共有|总共|有|可见|观察到|出现)\s*(?<!第)(\d+)\s*(?:条|处|个)?\s*(?:清晰可见的|明显的|主要)?裂缝",
        r"图(?:像|中)?[^。；,\n]{0,24}?(?<!第)(\d+)\s*(?:条|处|个)?\s*(?:清晰可见的|明显的|主要)?裂缝",
        r"(?:图(?:像|中)?|可以看到|观察到|有|可见)[^。；,\n]{0,24}?(?<!第)(\d+)\s*(?:条|处|个)?[^。；,\n]{0,12}?(?:白色)?(?:双曲线|S形缝隙|s形缝隙)(?:特征)?",
        r"裂缝(?:数量|数)[：:\s]*(?:图像中)?(?:清晰可见)?\s*(\d+)",
        r"裂缝[：:]\s*(\d+)",
        r"(\d+)\s+(?:visible\s+)?fractures?",
    ]
    for pattern in numeric_patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))

    vcd_count_patterns = [
        (
            r"(?:\u56fe\u4e2d|\u56fe\u50cf\u4e2d|\u53ef\u4ee5\u786e\u5b9a\u56fe\u4e2d|\u89c2\u5bdf\u5230|"
            r"\u770b\u5230|\u53ea\u770b\u5230|\u6e05\u6670\u53ef\u89c1)[^\u3002\uff1b,\n]{0,32}"
            r"(?<!\u7b2c)(\d+)\s*\u6761[^\u3002\uff1b,\n]{0,32}"
            r"(?:\u88c2\u7f1d|\u767d\u8272\u53cc\u66f2\u7ebf|\u53cc\u66f2\u7ebf|\u5782\u76f4\u7ebf\u6761|\u7ebf\u6761|\u7279\u5f81)"
        ),
        (
            r"(?:\u88c2\u7f1d\u6570\u91cf|\u8ba1\u6570)[^\u3002\uff1b,\n]{0,16}"
            r"(?<!\u7b2c)(\d+)\s*\u6761[^\u3002\uff1b,\n]{0,32}"
            r"(?:\u88c2\u7f1d|\u767d\u8272\u53cc\u66f2\u7ebf|\u53cc\u66f2\u7ebf|\u5782\u76f4\u7ebf\u6761|\u7ebf\u6761|\u7279\u5f81)"
        ),
    ]
    for pattern in vcd_count_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    single_feature_patterns = [
        (
            r"(?:\u56fe\u4e2d|\u56fe\u50cf\u4e2d|\u770b\u5230|\u53ea\u770b\u5230)[^\u3002\uff1b,\n]{0,24}"
            r"(?:\u4e00\u4e2a|\u4e00\u6761|1\u4e2a|1\u6761)[^\u3002\uff1b,\n]{0,24}"
            r"(?:\u767d\u8272)?\u53cc\u66f2\u7ebf"
        ),
    ]
    if any(re.search(pattern, text) for pattern in single_feature_patterns):
        return 1

    for token, value in _CN_NUMBERS.items():
        if re.search(rf"(?<!第){token}\s*(?:条|处|个)?\s*(?:清晰可见的|明显的|主要)?裂缝", text):
            return value
        if re.search(rf"(?:图(?:像|中)?|可以看到|观察到|有|可见)[^。；,\n]{{0,24}}(?<!第){token}\s*(?:条|处|个)?[^。；,\n]{{0,12}}(?:白色)?(?:双曲线|S形缝隙|s形缝隙)(?:特征)?", text):
            return value
    for token, value in _EN_NUMBERS.items():
        if re.search(rf"\b{token}\s+(?:visible\s+)?fractures?\b", normalized):
            return value

    return None


def count_category(count: int | None) -> str | None:
    if count is None:
        return None
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return ">=3"


def _as_label(value: int | str | None) -> str | None:
    if isinstance(value, int):
        return count_category(value)
    if isinstance(value, str):
        if value in COUNT_LABELS:
            return value
        parsed = parse_fracture_count(value)
        return count_category(parsed)
    return None


def compute_count_metrics(references: Iterable[int | str | None], predictions: Iterable[int | str | None]) -> dict[str, float]:
    ref_labels = [_as_label(value) for value in references]
    pred_labels = [_as_label(value) for value in predictions]
    if len(ref_labels) != len(pred_labels):
        raise ValueError("references and predictions must have the same length")
    if not ref_labels:
        return {"count_precision": 0.0, "count_recall": 0.0, "count_f1": 0.0, "count_accuracy": 0.0}

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in COUNT_LABELS:
        tp = sum(ref == label and pred == label for ref, pred in zip(ref_labels, pred_labels))
        fp = sum(ref != label and pred == label for ref, pred in zip(ref_labels, pred_labels))
        fn = sum(ref == label and pred != label for ref, pred in zip(ref_labels, pred_labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    exact = sum(ref == pred for ref, pred in zip(ref_labels, pred_labels)) / len(ref_labels)
    return {
        "count_precision": sum(precision_values) / len(COUNT_LABELS),
        "count_recall": sum(recall_values) / len(COUNT_LABELS),
        "count_f1": sum(f1_values) / len(COUNT_LABELS),
        "count_accuracy": exact,
    }


def _label_to_rank(value: int | str | None) -> int | None:
    label = _as_label(value)
    if label is None:
        return None
    return {"0": 0, "1": 1, "2": 2, ">=3": 3}[label]


def compute_hallucination_metrics(references: Iterable[int | str | None], predictions: Iterable[int | str | None]) -> dict[str, float]:
    ref_ranks = [_label_to_rank(value) for value in references]
    pred_ranks = [_label_to_rank(value) for value in predictions]
    if len(ref_ranks) != len(pred_ranks):
        raise ValueError("references and predictions must have the same length")
    pairs = [(ref, pred) for ref, pred in zip(ref_ranks, pred_ranks) if ref is not None and pred is not None]
    if not pairs:
        return {"zero_fracture_hallucination_rate": 0.0, "over_count_rate": 0.0, "under_count_rate": 0.0}

    zero_refs = [(ref, pred) for ref, pred in pairs if ref == 0]
    zero_hallucination = sum(pred > 0 for _, pred in zero_refs) / len(zero_refs) if zero_refs else 0.0
    return {
        "zero_fracture_hallucination_rate": zero_hallucination,
        "over_count_rate": sum(pred > ref for ref, pred in pairs) / len(pairs),
        "under_count_rate": sum(pred < ref for ref, pred in pairs) / len(pairs),
    }


def extract_location_labels(text: str) -> frozenset[str]:
    labels: set[str] = set()
    normalized = text.lower()
    zone_patterns = {
        "upper_third": ["上部1/3", "上1/3", "upper third", "upper 1/3"],
        "middle": ["偏中间", "中部", "middle"],
        "lower_third": ["下部1/3", "下1/3", "lower third", "lower 1/3"],
    }
    for label, patterns in zone_patterns.items():
        if any(pattern in normalized for pattern in patterns):
            labels.add(label)

    context_pattern = re.compile(r"(?:位置|间距|间隔|距离|分布|垂直|vertical|spacing|distance|position)[^。；;\n]{0,48}?([01]?\.\d{1,4}|1\.0{1,4})", re.IGNORECASE)
    for match in context_pattern.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if 0 <= value <= 1:
            labels.add(f"rel_{value:.2f}")
    return frozenset(labels)


def _has_positive_term(text: str, terms: Iterable[str]) -> bool:
    normalized = text.lower()
    negations = ("没有", "未", "无", "not", "no ")
    for term in terms:
        start = 0
        term_lower = term.lower()
        while True:
            index = normalized.find(term_lower, start)
            if index < 0:
                break
            window = normalized[max(0, index - 8) : index]
            if not any(negation in window for negation in negations):
                return True
            start = index + len(term_lower)
    return False


def extract_morphology_labels(text: str) -> frozenset[str]:
    labels: set[str] = set()
    feature_terms = {
        "white": ["白色", "white"],
        "hyperbola": ["双曲线", "hyperbola", "hyperbolic"],
        "s_shape": ["s形", "s-shaped", "s shape"],
        "symmetric": ["对称", "symmetry", "symmetric"],
        "parallel": ["平行", "parallel"],
    }
    for label, terms in feature_terms.items():
        if _has_positive_term(text, terms):
            labels.add(label)
    return frozenset(labels)


def _label_subset_accuracy(
    references: Iterable[str],
    predictions: Iterable[str],
    extractor,
) -> tuple[float | None, int]:
    pairs = [(extractor(ref), extractor(pred)) for ref, pred in zip(references, predictions)]
    evaluable = [(ref_labels, pred_labels) for ref_labels, pred_labels in pairs if ref_labels]
    if not evaluable:
        return None, 0
    correct = sum(ref_labels.issubset(pred_labels) for ref_labels, pred_labels in evaluable)
    return correct / len(evaluable), len(evaluable)


def compute_structured_text_metrics(references: Iterable[str], predictions: Iterable[str]) -> dict[str, float | int | None]:
    refs = list(references)
    preds = list(predictions)
    if len(refs) != len(preds):
        raise ValueError("references and predictions must have the same length")
    loc_acc, loc_evaluable = _label_subset_accuracy(refs, preds, extract_location_labels)
    morph_acc, morph_evaluable = _label_subset_accuracy(refs, preds, extract_morphology_labels)
    return {
        "loc_acc": loc_acc,
        "morph_acc": morph_acc,
        "loc_evaluable": loc_evaluable,
        "morph_evaluable": morph_evaluable,
    }


def _tokenize(text: str) -> list[str]:
    if re.search(r"[\u4e00-\u9fff]", text):
        return [char for char in text if not char.isspace()]
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def simple_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
        if not pred_ngrams:
            precisions.append(1e-9)
            continue
        clipped = sum(min(count, ref_ngrams[ngram]) for ngram, count in pred_ngrams.items())
        precisions.append(max(clipped / sum(pred_ngrams.values()), 1e-9))

    brevity = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return brevity * math.exp(sum(math.log(value) for value in precisions) / max_n)


def rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for index, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def average_text_metrics(predictions: Iterable[str], references: Iterable[str]) -> dict[str, float]:
    pairs = list(zip(predictions, references))
    if not pairs:
        return {"avg_bleu": 0.0, "avg_rouge_l": 0.0}
    return {
        "avg_bleu": sum(simple_bleu(pred, ref) for pred, ref in pairs) / len(pairs),
        "avg_rouge_l": sum(rouge_l_score(pred, ref) for pred, ref in pairs) / len(pairs),
    }
