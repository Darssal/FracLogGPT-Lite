from __future__ import annotations

from typing import Any

from fraclog.predictions import prediction_key


class BaselineImportError(ValueError):
    pass


def _prediction_text(row: dict[str, Any], prediction_field: str) -> str:
    value = row.get(prediction_field)
    if value is None and prediction_field != "prediction":
        value = row.get("prediction")
    if value is None:
        value = row.get("output")
    if value is None:
        value = row.get("text")
    if not isinstance(value, str) or not value.strip():
        raise BaselineImportError(f"Baseline prediction row has no text field: key={prediction_key(row)}")
    return value


def _index_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = prediction_key(row)
        if key in indexed:
            raise BaselineImportError(f"Duplicate baseline prediction for {key}")
        indexed[key] = row
    return indexed


def _eval_image(row: dict[str, Any]) -> str | None:
    image = row.get("image")
    if isinstance(image, str):
        return image
    images = row.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    return None


def standardize_baseline_predictions(
    eval_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    prediction_field: str = "prediction",
) -> list[dict[str, Any]]:
    indexed_predictions = _index_predictions(prediction_rows)
    output_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for eval_row in eval_rows:
        key = prediction_key(eval_row)
        pred_row = indexed_predictions.get(key)
        if pred_row is None:
            missing.append(key)
            continue
        output: dict[str, Any] = {
            "id": eval_row.get("id"),
            "image": _eval_image(eval_row),
            "reference": eval_row.get("reference", ""),
            "prediction": _prediction_text(pred_row, prediction_field),
        }
        if eval_row.get("count") is not None:
            output["count"] = eval_row.get("count")
        if pred_row.get("latency_s") is not None:
            output["latency_s"] = pred_row.get("latency_s")
        output_rows.append(output)

    extra = sorted(set(indexed_predictions) - {prediction_key(row) for row in eval_rows})
    if missing:
        preview = ", ".join(missing[:5])
        raise BaselineImportError(f"Missing baseline predictions for {len(missing)} eval rows: {preview}")
    if extra:
        preview = ", ".join(extra[:5])
        raise BaselineImportError(f"Baseline predictions contain {len(extra)} rows outside eval split: {preview}")
    return output_rows
