# Prompt format

The model uses the project's frozen zero-shot template **`nl-mql-spider1-v1`** (template sha256 `7401bf14…`), the
same one as the MQL-0006/0008/0010 zero-shot arms, inside Qwen's chat template. It was scored with these prompts.

## 1. Messages
```
[ {"role": "system", "content": <system prompt>},
  {"role": "user",   "content": <user prompt>} ]
```
Rendered with the tokenizer's chat template (`apply_chat_template(..., add_generation_prompt=True)`), i.e. Qwen ChatML:
```
<|im_start|>system
<system prompt><|im_end|>
<|im_start|>user
<user prompt><|im_end|>
<|im_start|>assistant
```

## 2. System prompt (verbatim)
In `inference/system_prompt.txt`. It's identical in training and evaluation.

## 3. User prompt
Full example: `inference/example_user_prompt.txt`. Shape:
```
Database: <db_id>

Collections (field: observed type [declared SQL type]):
- <db>__<Table>
    <field>: <type> [<SQL type>]
    ...

Foreign keys:
- <db>__<A>.<field> -> <db>__<B>.<field>        ("(none)" if there are none)

Sample documents:
- <db>__<Table>: {<one JSON document>}           (2 per collection)

Question: <natural-language question>

Return the JSON object.
```
The builder is `scripts/spider1/prompt_spider1.py` (also `kaggle/nl2mql_spider_dev_1034_bundle_v2/code/prompt_spider1.py`).
For the 716 MQL-0010 questions the finished prompts are in the MQL-0010 package, `data/eval/prompts.jsonl`.

**Training vs evaluation:** the training rows use the same system prompt and layout, but list only the collections
the corpus touched and write `Foreign keys: (none)` / `Sample documents: (none)`. Evaluation prompts include the full
schema, foreign keys and samples. The 72% was measured with the full evaluation prompts.

## 4. Expected output
One JSON object, no prose, no fences, compact:
```json
{"collection":"department_management__head","pipeline":[{"$match":{"age":{"$gt":56}}}, ...]}
```
Run it as `db[collection].aggregate(pipeline)`. Each output document is one answer row.

## 5. Decoding (as scored)
bfloat16, greedy (`do_sample=False`), `max_new_tokens=1024`, batch 8 with left padding, stop at `<|im_end|>`/EOS.
Parse with the scorer's rule: strip ``` fences if present, take the first complete JSON object
(`parse_output` in `inference/generate.py`).
