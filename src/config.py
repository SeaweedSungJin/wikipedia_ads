"""Configuration utilities loaded from a simple YAML file."""
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    """Runtime configuration options for the RAG search pipeline."""

    # Paths
    base_path: str  # Folder containing the knowledge base files
    image_path: str  # Query image path or URL
    kb_json_name: str  # Name of the KB JSON file

    # Search parameters
    text_query: str  # Text question for retrieval
    k_value: int  # Number of images to retrieve
    m_value: int  # Number of text sections to return
    alpha: float  # Weight for image/text fusion
    search_expand: int | None = None  # Oversampled FAISS results before filtering

    # Model options
    text_encoder_model: str = "facebook/contriever"  # HuggingFace model name
    segment_level: str = "section"  # "section", "paragraph" or "sentence"
    chunk_size: int = 1024  # Maximum characters per segment when splitting
    bge_max_length: int = 512
    bge_batch_size: int = 512
    bge_model: str = "BAAI/bge-reranker-v2-m3"  # BGE reranker model name
    electra_model: str = "cross-encoder/ms-marco-electra-base"  # Cross-encoder model
    electra_batch_size: int = 32
    mpnet_model: str = "sentence-transformers/all-mpnet-base-v2"  # Bi-encoder model
    bge_conf_threshold: float = 0.5  # Confidence threshold for reranker scores
    # Fusion weights for section ranking (min-max scaled):
    # combined = rank_img_weight * img_score_norm + rank_rerank_weight * rerank_score_norm
    rank_img_weight: float = 0.3
    rank_rerank_weight: float = 0.7
    # Normalization temperatures
    # Image doc score: softmax over retrieved docs with this temperature
    rank_img_softmax_temp: float = 1.0
    # Text reranker score: sigmoid(score / temp)
    rank_text_temp: float = 2.0
    # NLI model names
    deberta_nli_model: str = "tasksource/deberta-base-long-nli"
    roberta_nli_model: str = "FacebookAI/roberta-large-mnli"
    deberta_v3_nli_model: str = "microsoft/deberta-v3-large"
    nli_models: dict = field(default_factory=dict)  # Which NLI model to use
    nli_max_length: int = 512  # Max tokens per NLI input pair
    nli_max_cluster: int = 3  # Max sections per NLI cluster
    nli_e_min: float = 0.5  # Entailment minimum for graph edges
    nli_margin: float = 0.15  # Entailment-contradiction margin for edges
    nli_tau: float = 0.25  # Weight cutoff for retaining edges
    nli_lambda: float = 0.7  # Blend weight between BGE and edge coherence
    # Pairwise consistency weighting (C_ij = max(0, alpha*Pe - beta*Pc))
    nli_alpha: float = 1.0
    nli_beta: float = 1.0
    # NLI selection mode: 'clique' (existing) or 'consistency' (greedy pruning)
    nli_selection: str = "consistency"
    # Hybrid cluster scoring in consistency mode:
    # final = nli_hybrid_lambda * consistency_norm + (1 - nli_hybrid_lambda) * section_score_norm
    nli_hybrid_lambda: float = 0.5
    # NLI edge rule: 'avg' (default) or 'both_dir'
    nli_edge_rule: str = "avg"
    # Minimum (entailment - contradiction) required in each direction when using 'both_dir'
    nli_dir_margin: float = 0.0
    # NLI performance knobs (autocast only; bidirectional/topk removed for stability)
    nli_autocast: bool = True  # Use torch.autocast for faster NLI matmuls
    nli_autocast_dtype: str = "fp16"  # 'fp16' or 'bf16'
    # Which ranking modules to enable
    rerankers: dict = field(default_factory=dict)

    # Image search options
    # Previous options for TF-IDF pre-filtering and restricting FAISS to the first
    # image per document have been removed for simplicity.

    # Dataset evaluation options
    dataset_csv: str | None = None  # Path to EVQA CSV file
    id2name_paths: list[str] | None = None  # Mapping from image ID to name files
    dataset_image_root: str | None = None  # Root directory for inaturalist images
    dataset_google_root: str | None = None  # Root directory for googlelandmark images
    dataset_start: int = 0  # Optional start offset
    dataset_end: int | None = None  # Optional end offset

    # Device placement
    image_device: int | str = 3  # GPU id for image model
    bge_device: int | str = 0  # GPU id or name for BGE reranker
    nli_device: int | str = 0  # GPU id or name for NLI model
    vlm_device: int | str = 0  # GPU id or name for VLM model (separate from reranker to avoid OOM)
    nli_batch_size: int = 32  # Pairwise NLI comparisons per batch

    @property
    def kb_json_path(self) -> str:
        """Return full path to the KB JSONL file."""
        return os.path.join(self.base_path, self.kb_json_name)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Parse a tiny YAML-like config file into a :class:`Config`."""
        import yaml

        print("config.yaml 로딩중...")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Support legacy field name `googlelandmark_root`
        if "googlelandmark_root" in data and "dataset_google_root" not in data:
            data["dataset_google_root"] = data.pop("googlelandmark_root")

        # Backwards compatibility for previous `id2name_json` field
        if "id2name_json" in data and "id2name_paths" not in data:
            value = data.pop("id2name_json")
            if isinstance(value, str):
                data["id2name_paths"] = [value]
            else:
                data["id2name_paths"] = value

        if "rerankers" not in data:
            data["rerankers"] = {}
        if "nli_models" not in data:
            data["nli_models"] = {}

        # Ignore any unexpected keys instead of raising a TypeError
        valid_fields = set(cls.__dataclass_fields__)
        unknown_keys = [k for k in list(data.keys()) if k not in valid_fields]
        if unknown_keys:
            print("Ignoring unknown config keys:", ", ".join(unknown_keys))
            for k in unknown_keys:
                data.pop(k)

        return cls(**data)
