"""Local paths and optional integration settings."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path_setting(name, default=None):
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Settings:
    tables: Path
    training: Path
    examples: Path
    gguf: Path
    approved_corpus: Path | None = None
    bge_embeddings: Path | None = None
    bge_snapshot: Path | None = None
    ft_adapter: Path | None = None
    ft_model: Path | None = None
    mongo_uri: str | None = field(default=None, repr=False)
    port: int = 7860
    gpu_layers: int = 0

    @classmethod
    def from_env(cls):
        spider = path_setting("MQL_DEMO_SPIDER_DIR", Path.home() / "Documents/CodeGen-MongoDB/spider_data")
        return cls(
            tables=path_setting("MQL_DEMO_TABLES", spider / "tables.json"),
            training=path_setting("MQL_DEMO_TRAIN", spider / "train_spider.json"),
            examples=path_setting("MQL_DEMO_EXAMPLES", spider / "week2_validated.json"),
            gguf=path_setting("MQL_DEMO_GGUF", ROOT / "models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
            approved_corpus=path_setting("MQL_DEMO_APPROVED_CORPUS"),
            bge_embeddings=path_setting("MQL_DEMO_BGE_EMBEDDINGS"),
            bge_snapshot=path_setting("MQL_DEMO_BGE_SNAPSHOT"),
            ft_adapter=path_setting("MQL_DEMO_FT_ADAPTER"),
            ft_model=path_setting("MQL_DEMO_FT_MODEL"),
            mongo_uri=os.getenv("MONGODB_URI"),
            port=int(os.getenv("MQL_DEMO_PORT", "7860")),
            gpu_layers=int(os.getenv("MQL_DEMO_GPU_LAYERS", "0")),
        )
