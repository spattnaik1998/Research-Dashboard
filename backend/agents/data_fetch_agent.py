"""
Data Fetch Agent - Responsible for fetching publication data from Google Scholar
using the Tavily API for web scraping.
"""

import os
import asyncio
from typing import Dict, List, Any
from tavily import TavilyClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetchAgent:
    """
    Agent responsible for fetching publication data from Google Scholar.
    Uses Tavily API to scrape and extract publication information.
    """

    def __init__(self, tavily_api_key: str):
        """
        Initialize the Data Fetch Agent.

        Args:
            tavily_api_key: API key for Tavily service
        """
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.agent_name = "DataFetchAgent"
        logger.info(f"{self.agent_name} initialized successfully")

    async def fetch_scholar_data(self, scholar_url: str) -> Dict[str, Any]:
        """
        Fetch publication data from Google Scholar profile.

        Args:
            scholar_url: URL of the Google Scholar profile

        Returns:
            Dictionary containing fetched data with status and results
        """
        logger.info(f"{self.agent_name}: Starting to fetch data from {scholar_url}")

        try:
            # Use Tavily to search for information about the scholar profile
            search_query = f"Google Scholar profile publications citations {scholar_url}"

            # Perform search using Tavily
            search_results = await asyncio.to_thread(
                self.tavily_client.search,
                query=search_query,
                search_depth="advanced",
                max_results=10
            )

            # Extract relevant information
            publications = await self._parse_scholar_results(search_results, scholar_url)

            logger.info(f"{self.agent_name}: Successfully fetched {len(publications)} publications")

            return {
                "status": "success",
                "agent": self.agent_name,
                "scholar_url": scholar_url,
                "publications_count": len(publications),
                "publications": publications,
                "raw_search_results": search_results
            }

        except Exception as e:
            logger.error(f"{self.agent_name}: Error fetching data - {str(e)}")
            return {
                "status": "error",
                "agent": self.agent_name,
                "error": str(e),
                "publications": []
            }

    async def _parse_scholar_results(self, search_results: Dict, scholar_url: str) -> List[Dict[str, Any]]:
        """
        Parse Tavily search results to extract publication information.

        Args:
            search_results: Raw results from Tavily search
            scholar_url: Original scholar URL for reference

        Returns:
            List of publication dictionaries
        """
        publications = []

        # Extract publications from search results
        # This is a placeholder implementation - in production, you'd use more sophisticated parsing
        if "results" in search_results:
            for idx, result in enumerate(search_results["results"][:10]):  # Limit to top 10
                publication = {
                    "id": f"pub_{idx + 1}",
                    "title": result.get("title", "Unknown Title"),
                    "url": result.get("url", ""),
                    "content_snippet": result.get("content", "")[:500],  # First 500 chars
                    "score": result.get("score", 0.0),
                    "raw_content": result.get("raw_content", ""),
                    # These will be enriched by further processing
                    "authors": [],
                    "year": None,
                    "venue": None,
                    "citations": 0,
                    "doi": None,
                    "pdf_url": None,
                    "abstract": None,
                    "keywords": [],
                    "coauthors": [],
                    "scholar_link": scholar_url,
                    "scrape_timestamp": search_results.get("timestamp", "")
                }
                publications.append(publication)

        return publications

    async def fetch_publication_details(self, publication_url: str) -> Dict[str, Any]:
        """
        Fetch detailed information about a specific publication.

        Args:
            publication_url: URL of the publication

        Returns:
            Dictionary with detailed publication information
        """
        logger.info(f"{self.agent_name}: Fetching details for {publication_url}")

        try:
            # Use Tavily extract for detailed content
            details = await asyncio.to_thread(
                self.tavily_client.extract,
                urls=[publication_url]
            )

            return {
                "status": "success",
                "url": publication_url,
                "details": details
            }

        except Exception as e:
            logger.error(f"{self.agent_name}: Error fetching publication details - {str(e)}")
            return {
                "status": "error",
                "url": publication_url,
                "error": str(e)
            }

    def get_agent_status(self) -> Dict[str, str]:
        """
        Get current status of the agent.

        Returns:
            Dictionary with agent status information
        """
        return {
            "agent_name": self.agent_name,
            "status": "ready",
            "capabilities": [
                "fetch_scholar_data",
                "fetch_publication_details"
            ]
        }
