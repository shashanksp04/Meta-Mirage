from typing import List, Dict, Optional, Tuple
import re
import hashlib
from transformers import AutoTokenizer


class ContentUtils:
    """
    Utility class for content processing tasks such as
    hashing, chunking, and deduplication checks.
    """

    def __init__(
        self,
        embed_model: str = "BAAI/bge-base-en-v1.5",
        chunk_config: Dict | None = None,
    ):
        self.embed_model = embed_model
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)

        self.chunk_config = chunk_config or {
            "pdf": {
                "max_tokens": 500,
                "overlap": 90,
            },
            "web": {
                "chunk_if_over": 800,
                "max_tokens": 700,
                "overlap": 80,
            },
        }

    # -------------------------
    # Hashing
    # -------------------------

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Computes a normalized SHA-256 hash for text content."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # -------------------------
    # Deduplication
    # -------------------------

    @staticmethod
    def content_hash_exists(collection, content_hash: str) -> bool:
        """Checks whether a content hash already exists in the collection."""
        result = collection.get(where={"content_hash": content_hash})
        return len(result.get("ids", [])) > 0

    # -------------------------
    # Chunking
    # -------------------------

    def chunk_by_tokens(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap: int,
    ) -> List[str]:
        """Splits text into overlapping chunks based on token count."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: List[str] = []

        start = 0
        total_tokens = len(tokens)

        while start < total_tokens:
            end = min(start + max_tokens, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True
            )
            chunks.append(chunk_text)

            if end == total_tokens:
                break

            start = max(end - overlap, 0)

        return chunks
        
    def retrieve_with_priority_filters(
        self,
        *,
        query: str,
        collection,
        location: Optional[str] = None,
        month_year: Optional[str] = None,
        title: Optional[str] = None,
        k: int = 5,
        min_results: int = 2,
    ) -> Tuple[Optional[Dict], str, List[Dict]]:
        """
        Performs semantic retrieval with progressive metadata filtering.

        Returns:
            used_filter: The metadata filter that succeeded (or None)
            strategy: Name of the retrieval strategy used
            results: List of retrieved chunks with text, metadata, and distance
        """

        location = location.upper() if location else None
        month_year = month_year.strip() if month_year else None
        title = title.strip() if title else None

        filter_attempts: List[Tuple[str, Optional[Dict]]] = []

        if location and month_year and title:
            filter_attempts.append((
                "location+month_year+title",
                {"location": location, "month_year": month_year, "title": title}
            ))

        if location and month_year:
            filter_attempts.append((
                "location+month_year",
                {"location": location, "month_year": month_year}
            ))

        if location:
            filter_attempts.append(("location", {"location": location}))

        if title:
            filter_attempts.append(("title", {"title": title}))

        filter_attempts.append(("semantic_only", None))

        for strategy_name, where_filter in filter_attempts:

            query_args = {
                "query_texts": [query],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"],
            }

            if where_filter:
                query_args["where"] = where_filter

            results = collection.query(**query_args)

            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if len(docs) >= min_results:
                return (
                    where_filter,
                    strategy_name,
                    [
                        {
                            "text": doc,
                            "metadata": metadata,
                            "distance": distance,
                        }
                        for doc, metadata, distance in zip(docs, metadatas, distances)
                    ],
                )

        return None, "no_results", []
