#!/usr/bin/env python3
"""
Gemini Search Tool
Provides search and information retrieval using Gemini
"""

import os
import google.generativeai as genai
from typing import List, Dict, Any


class GeminiSearch:
    """Search tool using Gemini for context retrieval"""

    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def search(self, query: str, context: str = "") -> str:
        """Perform search query with optional context"""
        prompt = f"""Search Query: {query}

Context: {context if context else 'No additional context provided'}

Please provide relevant information, insights, and references for this query.
Focus on technical accuracy and cite sources where applicable.
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Search error: {str(e)}"

    def multi_query_search(self, queries: List[str]) -> Dict[str, str]:
        """Perform multiple searches"""
        results = {}
        for query in queries:
            results[query] = self.search(query)
        return results

    def comparative_search(self, topic_a: str, topic_b: str) -> str:
        """Compare two topics"""
        prompt = f"""Compare and contrast:
A: {topic_a}
B: {topic_b}

Provide:
1. Key similarities
2. Key differences
3. Strengths and weaknesses of each
4. Use cases and recommendations
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Comparison error: {str(e)}"
