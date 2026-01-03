import json
import re
from typing import List, Dict
from chat_models.Client import Client


class KeywordExtractor:
    """
    Agent tool for extracting search-optimized keywords from a user query.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", openai_api_base: str = "http://127.0.0.1:11434/v1"):
        """
        Args:
            model_name: LLM used for keyword extraction
            openai_api_base: Base URL of the OpenAI-compatible API
        """
        self.client = Client(model_name=model_name, openai_api_base=openai_api_base)

    def extract_keywords(self, *, query: str) -> Dict:
        """Extracts search-optimized keywords from a query.

        Use this tool when:
        - Preparing a query for web search
        - Improving recall for SerpAPI / Google-style search

        The tool:
        - Extracts important keywords and phrases
        - Combines related terms
        - Quotes multi-word concepts
        - Orders keywords from general to specific

        Args:
            query: User query to extract keywords from

        Returns:
            Success:
            {
              "status": "success",
              "keywords": [str, ...]
            }

            Error:
            {
              "status": "error",
              "error_message": str
            }
        """

        if not query or not query.strip():
            return {
                "status": "error",
                "error_message": "Empty query provided",
            }

        prompt = f"""
            You are an intelligent keyword extraction assistant.

            Your task is to extract the most important and relevant keywords or short phrases
            from the user query below, and organize them in a logical order suitable for
            Google-style web search.

            Rules:
            - Combine related words into phrases when appropriate.
            - Include entities such as crops, pests, locations, years, organizations, or events.
            - If a keyword or phrase contains multiple words forming a fixed concept,
            enclose it in double quotes (" ").
            - Do NOT quote single words.
            - Output MUST be a valid JSON list of strings.
            - Order keywords from general to specific.

            Example:
            Input: "impact of drought on corn and soybean pests in Maryland 2022"
            Output: ["drought impact", "corn", "soybean pests", "Maryland", "2022"]

            User Query:
            {query}
            """

        try:
            raw_text = self.client.chat(prompt=prompt)

            # ---- Clean formatting ----
            cleaned = re.sub(r"```(?:json)?", "", raw_text)
            cleaned = cleaned.replace("```", "").strip()
            cleaned = cleaned.replace('\\"', '"')
            cleaned = re.sub(r'""', '"', cleaned)

            # ---- Parse JSON safely ----
            try:
                keywords = json.loads(cleaned)
                if not isinstance(keywords, list):
                    return {
                        "status": "error",
                        "error_message": "Model did not return a valid JSON list",
                    }
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", cleaned, re.DOTALL)
                if not match:
                    return {
                        "status": "error",
                        "error_message": "Model did not return a valid JSON list",
                    }
                try:
                    keywords = json.loads(match.group(0))
                    if not isinstance(keywords, list):
                        return {
                            "status": "error",
                            "error_message": "Model did not return a valid JSON list",
                        }
                except json.JSONDecodeError:
                    return {
                        "status": "error",
                        "error_message": "Model did not return a valid JSON list",
                    }

            # ---- Normalize ----
            keywords = [
                kw.strip()
                for kw in keywords
                if isinstance(kw, str) and kw.strip()
            ]

            if not keywords:
                return {
                    "status": "error",
                    "error_message": "No keywords extracted",
                }

            return {
                "status": "success",
                "keywords": keywords,
            }

        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Keyword extraction failed: {str(e)}",
            }
