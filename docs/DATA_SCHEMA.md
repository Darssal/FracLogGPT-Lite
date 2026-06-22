# Data Schema

The public repository does not contain real fracture-log data. Use the synthetic examples under `examples/` only as format references.

## SFT Conversations

Expected local path:

```text
sft/conversations.json
```

Accepted records are inspected dynamically by `scripts/inspect_data.py` and normalized by `scripts/convert_data.py`. A typical multimodal SFT record contains:

```json
{
  "id": "sample-0001",
  "images": ["pic/sample-0001.png"],
  "messages": [
    {"role": "system", "content": "You are a fracture interpretation assistant."},
    {"role": "user", "content": "<image>\nDescribe fracture count, location, and morphology."},
    {"role": "assistant", "content": "Fracture count: 1; location: upper interval; morphology: conductive fracture."}
  ]
}
```

## Preference Triplets

Expected local path:

```text
rlhf_data/pref_triplet.jsonl
```

Each row should preserve the original prompt context and include chosen/rejected answers:

```json
{"id":"pref-0001","images":["pic/sample-0001.png"],"messages":[{"role":"user","content":"<image>\nDescribe fractures."}],"chosen":"Fracture count: 1; location: upper interval.","rejected":"No fracture is visible."}
```

## Evaluation Predictions

Prediction JSONL files should contain one row per evaluation item:

```json
{"id":"sample-0001","image":"pic/sample-0001.png","reference":"Fracture count: 1; location: upper interval.","prediction":"Fracture count: 1; location: upper interval.","latency_s":1.23}
```

## DAPT Corpus

For paper-style DAPT, place domain text files outside Git, for example:

```text
data/dapt_corpus/*.txt
```

Then build the training JSONL:

```powershell
python scripts/prepare_dapt_corpus.py --corpus-dir data/dapt_corpus --out outputs/data/dapt_train.jsonl
```

Do not commit proprietary geological text, well identifiers, image names, annotation comments, or expert-review notes.
