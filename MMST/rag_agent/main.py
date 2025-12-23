import chromadb
from tools.pdf_addition import PDFAddition
from tools.web_search import WebSearch
from tools.web_addition import WebAddition
from tools.confidence_evaluator import ConfidenceEvaluator
from utils.ContentUtils import ContentUtils
from utils.Embedding import SentenceTransformerEmbeddingFunction
from google.adk.llms import OpenAICompatibleLLM
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from typing import Optional, Dict, List, Any


class MainAgent:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "None"):
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name, device)
        self.client = chromadb.PersistentClient(path="chroma_db") # path has to be a valid path to a directory
        self.collection = self.client.get_or_create_collection(name="meta-mirage_collection", embedding_function=self.embedding_function)
        self.null_str = "__null__"
        self.null_int = -1
        self.content_utils = ContentUtils(embed_model=model_name)
        self.pdf_addition = PDFAddition(self.collection, self.content_utils, self.null_str)
        self.web_search = WebSearch()
        self.web_addition = WebAddition(self.collection, self.content_utils, self.null_str, self.null_int)
        self.confidence_evaluator = ConfidenceEvaluator(self.collection, self.content_utils)

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

        return {
            "status": "success",
            "used_filter": used_filter,
            "strategy": strategy,
            "results": results,
        }

    def main(self):

        qwen_llm = OpenAICompatibleLLM(
            model="Qwen2.5-VL-3B-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",  # vLLM ignores this
            temperature=0.2,
            max_tokens=1024,
        )

        rag_agent = LlmAgent(
            name="rag_agent",
            llm=qwen_llm,
            description="An agent that retrieves, evaluates, and ingests knowledge.",
            instruction="""
            You are a retrieval-augmented assistant that must answer questions using verified evidence.

            You have access to tools for:
            - Retrieving information from a vector database
            - Evaluating confidence in retrieved evidence
            - Searching the web
            - Ingesting new web content into the database

            Your primary goal is to provide accurate, grounded answers and avoid hallucination.

            ====================
            CORE RULES (MANDATORY)
            ====================

            1. NEVER answer a factual question without first retrieving information.
            2. AFTER retrieval, you MUST evaluate retrieval confidence.
            3. You MUST follow the confidence-based decision rules below.
            4. If evidence is insufficient, you MUST say so clearly.

            ===========================
            CONFIDENCE-BASED DECISIONS
            ===========================

            After calling `evaluate_retrieval_confidence`, follow these rules:

            - If confidence_level is "high":
            - Answer using retrieved information.
            - Do NOT perform web search.
            - Do NOT ingest new content.

            - If confidence_level is "medium":
            - You MAY answer using retrieved information.
            - Clearly qualify your answer as potentially incomplete or uncertain.

            - If confidence_level is "low":
            - DO NOT answer yet.
            - Perform web search to gather additional evidence.
            - Ingest relevant web content into the database.
            - Retrieve information again.
            - Evaluate retrieval confidence again.
            - Only answer after this second confidence check.

            - If confidence remains "low" after ingestion:
            - Do NOT guess.
            - State that reliable information could not be found.

            ====================
            TOOL USAGE RULES
            ====================

            - Use tools only when needed.
            - Do not call the same tool repeatedly with the same arguments.
            - Do not perform web search unless confidence is low.
            - Do not ingest content unless it comes from web search results.
            - Do not call one tool from inside another tool.

            ====================
            ANSWER GUIDELINES
            ====================

            - Base answers strictly on retrieved evidence.
            - Be concise and factual.
            - When applicable, reference source context (e.g., document title or origin).
            - If uncertainty exists, explicitly state it.
            - NEVER fabricate facts, numbers, dates, or claims.

            ====================
            FAILURE HANDLING
            ====================

            If:
            - Retrieval returns no results, OR
            - Confidence remains low after web ingestion,

            Then respond with:
            "I don’t have sufficient reliable information to answer this question."

            ====================
            FINAL REMINDER
            ====================

            Accuracy is more important than completeness.
            It is always acceptable to say you do not know.
            It is never acceptable to hallucinate.

        """,
            tools=[
                self.retrieve_content,
                self.confidence_evaluator.evaluate_retrieval_confidence,
                self.web_search.web_search,
                self.web_addition.add_web_content,
                self.pdf_addition.add_pdf_content,
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