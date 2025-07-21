"""Configuration utilities loaded from a simple YAML file."""
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    """Runtime configuration options for the RAG search pipeline."""

    # Paths
    base_path: str  # Folder containing the knowledge base files
    image_path: str  # Query image path or URL
    kb_json_name: str  # Name of the KB JSONL file

    # Search parameters
    text_query: str  # Text question for retrieval
    k_value: int  # Number of images to retrieve
    m_value: int  # Number of text sections to return
    alpha: float  # Weight for image/text fusion

    # Model options
    text_encoder_model: str = "facebook/contriever"  # HuggingFace model name
    segment_level: str = "section"  # "section", "paragraph" or "sentence"
    chunk_size: int = 1024  # Maximum characters per segment when splitting
    qformer_model: str = "Salesforce/blip2-flan-t5-xl"  # Q-former model name
    qformer_provider: str = "hf"  # "hf" or "lavis" loader
    qformer_weights: str | None = None  # optional fine-tuned weights path
    # Which ranking modules to enable
    rerankers: dict = field(default_factory=dict)

    # Optional TF-IDF pre-filtering
    use_tfidf_filter: bool = False  # Whether to filter sections with TF-IDF
    tfidf_ratio: float = 0.5  # Ratio of sections to keep when filtering

    # Dataset evaluation options
    dataset_csv: str | None = None  # Path to EVQA CSV file
    id2name_json: str | None = None  # Mapping from image ID to name
    dataset_image_root: str | None = None  # Root directory for inaturalist images
    dataset_google_root: str | None = None  # Root directory for googlelandmark images
    dataset_start: int = 0  # Optional start offset
    dataset_end: int | None = None  # Optional end offset
    
    # Device placement
    image_device: int | str = 3  # GPU id for image model

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

        if "rerankers" not in data:
            data["rerankers"] = {}

        return cls(**data)