import chromadb
from .tools.pdf_addition import PDFAddition
from .tools.web_search import WebSearch
from .tools.web_addition import WebAddition
from .tools.confidence_evaluator import ConfidenceEvaluator
from .tools.keyword_extractor import KeywordExtractor
from .utils.ContentUtils import ContentUtils
from .utils.Embedding import SentenceTransformerEmbeddingFunction
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from typing import Optional, Dict, List, Any


class MainAgent:
    def __init__(self, test_model: str = "Qwen2.5-VL-3B-Instruct", embed_model_name: str = "BAAI/bge-base-en-v1.5", device: str = "None", api_base: str = "http://127.0.0.1:11434/v1"):
        self.test_model = test_model
        self.api_base = api_base
        self.embedding_function = SentenceTransformerEmbeddingFunction(embed_model_name, device)
        self.client = chromadb.PersistentClient(path="./chroma_database/chroma_db") # path has to be a valid path to a directory
        self.collection = self.client.get_or_create_collection(name="meta-mirage_collection", embedding_function=self.embedding_function)
        self.null_str = "__null__"
        self.null_int = -1
        self.content_utils = ContentUtils(embed_model=embed_model_name)
        self.pdf_addition = PDFAddition(self.collection, self.content_utils, self.null_str)
        self.web_search = WebSearch()
        self.web_addition = WebAddition(self.collection, self.content_utils, self.null_str, self.null_int)
        self.confidence_evaluator = ConfidenceEvaluator(self.collection, self.content_utils)
        self.keyword_extractor = KeywordExtractor(model_name=test_model, openai_api_base=api_base)
        # Track web search calls and results
        self.web_search_calls = []
    
    def _tracked_web_search(self, query: str, results_to_extract_count: int = 10) -> Dict:
        """Wrapper around web_search that tracks calls and results"""
        result = self.web_search.web_search(query, results_to_extract_count)
        # Store the call information
        self.web_search_calls.append({
            "query": query,
            "status": result.get("status"),
            "results_count": len(result.get("results", [])) if result.get("status") == "success" else 0,
            "error": result.get("error_message") if result.get("status") == "error" else None,
            "results": result.get("results", [])[:3] if result.get("status") == "success" else []  # Store first 3 results
        })
        return result

    def retrieve_content(self,
            *,
            query: str,
            location: str | None = None,
            month_year: str | None = None,
            title: str | None = None,
        ) -> dict:
        """Retrieves relevant content using progressive metadata filtering."""

        if not query or not query.strip():
            return {
                "status": "error",
                "error_message": "Empty query provided",
                "results": [],
            }

        used_filter, strategy, results = self.content_utils.retrieve_with_priority_filters(
            query=query,
            collection=self.collection,
            location=location,
            month_year=month_year,
            title=title,
        )

        if not results:
            return {
                "status": "error",
                "error_message": "No results found",
                "results": [],
            }

        return {
            "status": "success",
            "used_filter": used_filter,
            "strategy": strategy,
            "results": results,
        }

    def main(self):
        import os
        # Set API base URL for OpenAI-compatible endpoints (vLLM)
        # google-adk uses OPENAI_API_BASE environment variable
        os.environ["OPENAI_API_BASE"] = self.api_base
        os.environ["OPENAI_API_KEY"] = "EMPTY"  # vLLM ignores this
        
        # Format model name for google-adk: "openai/model_name" for OpenAI-compatible APIs
        model_name = f"openai/{self.test_model}"
        
        rag_agent = LlmAgent(
            name="Rag_Agent",
            model=model_name,
            description="An agent that retrieves, evaluates, and ingests knowledge.",
            instruction=
            """You are a retrieval-augmented evidence runner. Your job is NOT to answer the user’s question.
            Your only job is to run the retrieval pipeline and return the exact retrieved text passages (verbatim)
            that are relevant to the user query, so they can be appended to the user query and sent to another model.

            You have access to tools for:
            - Extracting search-optimized keywords from a user query (extract_keywords)
            - Retrieving information from a vector database (retrieve)
            - Evaluating confidence in retrieved evidence (evaluate_retrieval_confidence)
            - Searching the web (web_search)
            - Ingesting new web content into the database (ingest_web_content)

            ====================
            CORE RULES (MANDATORY)
            ====================

            1. NEVER answer the user’s question.
            2. ALWAYS attempt retrieval first (vector database).
            3. AFTER each retrieval attempt, you MUST call evaluate_retrieval_confidence.
            4. You MUST follow the confidence-based decision rules below.
            5. Output MUST contain only retrieved text (verbatim) or nothing. No paraphrases, no summaries, no extra facts.
            6. If evidence is insufficient or not found, explicitly admit it and return no evidence.

            ===========================
            CONFIDENCE-BASED DECISIONS
            ===========================

            After calling evaluate_retrieval_confidence, follow these rules:

            - If confidence_level is "high":
            - Do NOT perform web search.
            - Return the retrieved passages exactly as-is (verbatim), with minimal structure (see Output Format).
            - Do NOT add analysis, explanation, or answers.

            - If confidence_level is "medium":
            - Do NOT answer the question.
            - Return the retrieved passages exactly as-is (verbatim).
            - Include a brief note: "Confidence: medium" (and nothing else besides the evidence).

            - If confidence_level is "low":
            - Do NOT return evidence yet (unless your pipeline requires showing low-confidence results; default is NO).
            - Prepare for web search by calling extract_keywords ONCE.
            - Join extracted keywords into a single query string.
            - Perform web_search.
            - Ingest relevant web content into the database (ingest_web_content) ONLY from web_search results.
            - Retrieve again from the vector database.
            - Evaluate retrieval confidence again.

            - If confidence remains "low" after ingestion:
            - Do NOT guess.
            - Respond exactly with:
                "No sufficient reliable information available to return."
            - Return no evidence (empty).

            ===================
            TOOL USAGE RULES
            ===================

            - Use tools only when needed.
            - Do not call the same tool repeatedly with the same arguments.
            - Do not perform web search unless confidence is low.
            - Do not ingest content unless it comes from web_search results.
            - Do not call one tool from inside another tool.
            - Do not fabricate sources, passages, titles, URLs, or citations.

            ===================
            KEYWORD EXTRACTION
            ===================

            Use extract_keywords ONLY when:
            - confidence_level is "low", AND
            - you are preparing a query for web_search.

            Rules:
            - Do NOT use extract_keywords if confidence is "high" or "medium".
            - Call extract_keywords at most once per user query.
            - If keyword extraction fails, fall back to the original user query for web_search.

            ================
            OUTPUT FORMAT
            ================

            Your output must be structured and STRICT.

            If confidence is high or medium and you have relevant evidence:

            Return:

            CONFIDENCE: <high|medium>
            EVIDENCE:
            <verbatim retrieved text passage 1>
            ...

            Rules:
            - Only include passages that were actually retrieved.
            - Do not edit, paraphrase, or “clean up” the text.
            - Preserve original punctuation, casing, line breaks, and any citations included in the retrieved text.
            - Do not add your own citations or commentary.
            - Do not include anything outside the template.

            If no relevant information is found OR confidence remains low after web ingestion:

            Return exactly:

            "No sufficient reliable information available to return."

            And DO NOT include an EVIDENCE section (i.e., return nothing else).

            ===================
            FINAL REMINDER
            ===================

            Accuracy is more important than completeness.
            It is always acceptable to return no evidence.
            It is never acceptable to hallucinate.
            """,
            tools=[
                self.retrieve_content,
                self.confidence_evaluator.evaluate_retrieval_confidence,
                self._tracked_web_search,  # Use tracked version to monitor web search calls
                self.web_addition.add_web_content,
                self.pdf_addition.add_pdf_content,
                self.keyword_extractor.extract_keywords,
            ],
        )

        runner = InMemoryRunner(agent=rag_agent)

        return runner

if __name__ == "__main__":
    main_agent = MainAgent()
    runner = main_agent.main()
    response = runner.run_debug(
        "What is Agent Development Kit from Google? What languages is the SDK available in?"
    )
    print(response)