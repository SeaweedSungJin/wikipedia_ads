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
    clip_cache_dir: str | None = None
    clip_cache_kmax: int = 50

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
    # Fusion ratio for section ranking using raw scores:
    # combined = rank_img_ratio * img_score_raw + (1 - rank_img_ratio) * rerank_score_raw
    rank_img_ratio: float = 0.5
    # Q-Former integration
    qformer_ckpt: str = "datasets/reranker.pth"
    qformer_device: int | str = 0
    qformer_section_batch_size: int = 32
    qformer_text_token_count: int = 32
    qformer_max_text_length: int = 256
    # Fusion ratio for Q-Former reranking (raw scores)
    qformer_img_ratio: float = 0.5
    # NLI model names
    deberta_nli_model: str = "tasksource/deberta-base-long-nli"
    roberta_nli_model: str = "FacebookAI/roberta-large-mnli"
    deberta_v3_nli_model: str = "microsoft/deberta-v3-large"
    nli_models: dict = field(default_factory=dict)  # Which NLI model to use
    nli_max_length: int = 512  # Max tokens per NLI input pair
    nli_max_cluster: int = 3  # Max sections per NLI cluster
    nli_tau: float = 0.25  # Weight cutoff for retaining edges
    # Pairwise consistency weighting (C_ij = max(0, alpha*Pe - beta*Pc))
    nli_alpha: float = 1.0
    nli_beta: float = 1.0
    # Hybrid cluster scoring in consistency mode:
    # final = nli_hybrid_lambda * consistency_norm + (1 - nli_hybrid_lambda) * section_score_norm
    nli_hybrid_lambda: float = 0.5
    # NLI edge rule: 'avg' (default) or 'both_dir'
    nli_edge_rule: str = "avg"
    # Minimum (entailment - contradiction) required in each direction when using 'both_dir'
    nli_dir_margin: float = 0.0
    # Graph reduction mode for consistency clustering: 'greedy' or 'recompute'
    nli_reduction_mode: str = "greedy"
    nli_reduction_epsilon: float = 1e-6
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

        # Convert legacy fusion weights to ratio if present
        if "rank_img_ratio" not in data:
            img_w = data.pop("rank_img_weight", None)
            rer_w = data.pop("rank_rerank_weight", None)
            text_temp = data.pop("rank_text_temp", None)
            data.pop("rank_img_softmax_temp", None)
            if img_w is not None or rer_w is not None:
                try:
                    img_w = float(0.0 if img_w is None else img_w)
                    rer_w = float(0.0 if rer_w is None else rer_w)
                    total = img_w + rer_w
                    if total > 0:
                        data["rank_img_ratio"] = img_w / total
                except (TypeError, ValueError):
                    pass
            elif text_temp is not None:
                # No legacy weights provided; nothing to convert, but key removed above
                pass

        if "qformer_img_ratio" not in data:
            doc_w = data.pop("qformer_doc_weight", None)
            txt_w = data.pop("qformer_text_weight", None)
            if doc_w is not None or txt_w is not None:
                try:
                    doc_w = float(0.0 if doc_w is None else doc_w)
                    txt_w = float(0.0 if txt_w is None else txt_w)
                    total = doc_w + txt_w
                    if total > 0:
                        data["qformer_img_ratio"] = doc_w / total
                except (TypeError, ValueError):
                    pass

        # Ignore any unexpected keys instead of raising a TypeError
        valid_fields = set(cls.__dataclass_fields__)
        unknown_keys = [k for k in list(data.keys()) if k not in valid_fields]
        if unknown_keys:
            print("Ignoring unknown config keys:", ", ".join(unknown_keys))
            for k in unknown_keys:
                data.pop(k)

        return cls(**data)
