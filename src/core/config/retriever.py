from dataclasses import dataclass


@dataclass
class RetrieverConfig:
    similarity_threshold: float = 0.5
    max_results: int = 5
    context_depth: int = 2
