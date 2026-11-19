# Fine-tuning pilot (Qwen2.5-Coder-3B, QLoRA)

Status: PILOT. Built from `provisional_rag_corpus.jsonl` (3,589 entries, provisional v1, AI-checked
only, not human-reviewed). Not the approved v2.1 corpus. Swap that in when you have it (step 1).

## 1. Build the data (already done once; rerun to change corpus)

    python scripts/finetune_pilot/build_pilot_sft.py --corpus provisional_rag_corpus.jsonl --out data/finetune_pilot/pilot_v0
    # later:  --corpus outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl --out data/finetune_pilot/pilot_v1

Result: 3,346 train / 243 dev. The dev split holds out 12 whole databases, so no database appears
in both. No training database is an evaluation database (checked against the 20 Spider dev DBs).

## 2. Train on Kaggle (free T4, no local GPU needed)

1. Create a Kaggle dataset from `data/finetune_pilot/pilot_v0/` (the two `sft_*.jsonl` files).
2. New notebook, accelerator GPU T4, add that dataset, upload `scripts/finetune_v1/train_qlora.py`.
3. Run:

        !pip install -q "transformers>=4.46" "trl>=0.20" "peft>=0.13" "bitsandbytes>=0.44" "accelerate>=1.0" datasets
        !python train_qlora.py --model Qwen/Qwen2.5-Coder-3B-Instruct \
            --data /kaggle/input/<your-dataset-name> --out /kaggle/working/qlora-3b-pilot
        !pip freeze > /kaggle/working/pip_freeze.txt

   `train_qlora.py` is the repo's existing script, unchanged (r=16, lr 2e-4, 2 epochs, loss on the
   completion only). Estimated 30-60 min per epoch on a T4. That is a guess, not a measurement.
4. Download `/kaggle/working/qlora-3b-pilot` (the LoRA adapter, `run_config.json`, `pip_freeze.txt`).

## 3. What the training run tells you, and what it doesn't

- Training/dev loss going down shows the pipeline works. It is not accuracy.
- The project metric (execution accuracy on Atlas, 892 questions) needs the read-only Atlas
  credential and Smaran's scoring scripts. Without them, report the loss curves and say accuracy
  is pending.
- Do not compare a number from this pilot with the 415/892 zero-shot or 457/892 RAG results
  without saying: provisional corpus, simplified prompt, and no paired test.

## 4. Deviations from the benchmark prompt (read before quoting results)

1. Only collections that some corpus entry touches are listed (about 20% of tables are missing).
2. Foreign keys and sample documents are "(none)" (the corpus has neither).
3. Field types are mapped from the corpus, not read from stored data.
4. About 22% of count-style answers use a long `$facet` wrapper, so the model will learn that style.

Fix for 1-3: rebuild from the Spider training SQLite files (needs the official `spider_data.zip`).
