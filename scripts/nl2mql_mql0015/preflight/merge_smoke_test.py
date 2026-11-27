"""CPU smoke test of mql0015_core.merge_adapter on a TINY random Qwen2 model with a toy LoRA (no download, no GPU).

Exercises the real merge code path (peft PeftModel.from_pretrained -> merge_and_unload, torchao workaround, trained-LoRA
check, changed-tensor count, byte copies of the base decoding files, weights_info) and checks that merging twice gives
byte-identical merged weights (needed for a session-3 resume). It does NOT load the real 3B model or adapter.

  PYTHONPATH=<dir with peft+accelerate+psutil> ~/codegen_data/venv_rag/Scripts/python.exe -B \
      scripts/nl2mql_mql0015/preflight/merge_smoke_test.py <workdir>
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]


def main(work):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM
    sys.path.insert(0, str(HERE.parents[1] / "kaggle/nl2mql_mql0013_test_full/src"))
    sys.path.insert(0, str(HERE / "pkg_src"))
    import mql0015_core as C
    work = Path(work)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
    base = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    base_dir = work / "base"
    base.save_pretrained(base_dir, safe_serialization=True)
    GenerationConfig(eos_token_id=[1, 2], pad_token_id=0, do_sample=False).save_pretrained(base_dir)
    (base_dir / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>"}),
                                                    encoding="utf-8")
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    pm = get_peft_model(Qwen2ForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16),
                        LoraConfig(r=4, lora_alpha=8, target_modules=targets, lora_dropout=0.0))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    ad = work / "adapter"
    pm.save_pretrained(ad)
    sha = C.sha256_file(ad / "adapter_model.safetensors")
    w1 = C.merge_adapter(base_dir, ad, work / "merged1", "bfloat16", expected_sha=sha)
    w2 = C.merge_adapter(base_dir, ad, work / "merged2", "bfloat16", expected_sha=sha)
    ok = {"changed_tensors": w1["changed_tensors"], "deterministic_merge": w1["merged_weights_sha256"] == w2["merged_weights_sha256"],
          "decoding_files_copied": sorted(w1["decoding_files_sha256_equal_to_base"]),
          "generation_config_identical": (work / "merged1/generation_config.json").read_bytes() == (base_dir / "generation_config.json").read_bytes()}
    try:
        C.merge_adapter(base_dir, ad, work / "merged3", "bfloat16")          # real pinned hash -> must refuse a toy adapter
        ok["refuses_wrong_adapter"] = False
    except Exception as e:  # noqa: BLE001
        ok["refuses_wrong_adapter"] = "adapter hash" in str(e)
    print(json.dumps(ok, indent=1))
    assert ok["deterministic_merge"] and ok["generation_config_identical"] and ok["refuses_wrong_adapter"]
    n_changed, n_total = map(int, w1["changed_tensors"].split("/"))
    assert n_changed == n_total == 14
    return ok


if __name__ == "__main__":
    main(sys.argv[1])
