"""
Data Fetch Agent - Responsible for fetching publication data from Google Scholar
using the Tavily API for web scraping.
"""

import os
import asyncio
import re
from typing import Dict, List, Any, Optional
from tavily import TavilyClient
from urllib.parse import urlparse, parse_qs
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Custom exception for rate limit errors."""
    pass


class InvalidScholarURLError(Exception):
    """Custom exception for invalid Google Scholar URLs."""
    pass


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
        if not tavily_api_key:
            raise ValueError("Tavily API key cannot be empty")

        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.agent_name = "DataFetchAgent"
        self.request_count = 0
        self.last_request_time = None
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
            # Validate the Google Scholar URL
            self._validate_scholar_url(scholar_url)

            # Check rate limiting
            await self._check_rate_limit()

            # Use Tavily to search for information about the scholar profile
            search_query = f"Google Scholar profile publications citations {scholar_url}"

            # Perform search using Tavily
            search_results = await asyncio.to_thread(
                self.tavily_client.search,
                query=search_query,
                search_depth="advanced",
                max_results=10
            )

            # Update request tracking
            self.request_count += 1
            self.last_request_time = datetime.utcnow()

            # Extract relevant information
            publications = await self._parse_scholar_results(search_results, scholar_url)

            logger.info(f"{self.agent_name}: Successfully fetched {len(publications)} publications")

            return {
                "status": "success",
                "agent": self.agent_name,
                "scholar_url": scholar_url,
                "publications_count": len(publications),
                "publications": publications,
                "timestamp": datetime.utcnow().isoformat()
            }

        except InvalidScholarURLError as e:
            logger.error(f"{self.agent_name}: Invalid Scholar URL - {str(e)}")
            return {
                "status": "error",
                "agent": self.agent_name,
                "error_type": "invalid_url",
                "error": str(e),
                "publications": []
            }

        except RateLimitError as e:
            logger.error(f"{self.agent_name}: Rate limit exceeded - {str(e)}")
            return {
                "status": "error",
                "agent": self.agent_name,
                "error_type": "rate_limit",
                "error": str(e),
                "publications": []
            }

        except Exception as e:
            error_message = str(e)
            error_type = "unknown"

            # Check if it's a Tavily-specific error
            if "429" in error_message or "rate limit" in error_message.lower():
                error_type = "rate_limit"
                logger.error(f"{self.agent_name}: API rate limit hit")
            elif "401" in error_message or "unauthorized" in error_message.lower():
                error_type = "authentication"
                logger.error(f"{self.agent_name}: Authentication failed - check API key")
            elif "timeout" in error_message.lower():
                error_type = "timeout"
                logger.error(f"{self.agent_name}: Request timeout")
            else:
                logger.error(f"{self.agent_name}: Unexpected error - {error_message}")

            return {
                "status": "error",
                "agent": self.agent_name,
                "error_type": error_type,
                "error": error_message,
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

    def _validate_scholar_url(self, url: str) -> None:
        """
        Validate that the provided URL is a valid Google Scholar profile URL.

        Args:
            url: URL to validate

        Raises:
            InvalidScholarURLError: If URL is invalid or malformed
        """
        if not url or not isinstance(url, str):
            raise InvalidScholarURLError("URL cannot be empty or non-string")

        # Parse the URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise InvalidScholarURLError(f"Malformed URL: {str(e)}")

        # Check if it's a Google Scholar domain
        valid_domains = ["scholar.google.com", "scholar.google.co.in"]
        if parsed.netloc not in valid_domains:
            raise InvalidScholarURLError(
                f"URL must be from Google Scholar domain (e.g., scholar.google.com). Got: {parsed.netloc}"
            )

        # Check if it's a citations/profile page
        if not parsed.path.startswith("/citations"):
            raise InvalidScholarURLError(
                "URL must be a Google Scholar citations/profile page (path should start with /citations)"
            )

        # Check if user parameter exists
        query_params = parse_qs(parsed.query)
        if "user" not in query_params:
            raise InvalidScholarURLError(
                "URL must contain a 'user' parameter (e.g., ?user=USER_ID)"
            )

        logger.info(f"{self.agent_name}: URL validation passed for {url}")

    async def _check_rate_limit(self, requests_per_minute: int = 10) -> None:
        """
        Check and enforce rate limiting to avoid API abuse.

        Args:
            requests_per_minute: Maximum allowed requests per minute

        Raises:
            RateLimitError: If rate limit would be exceeded
        """
        if self.last_request_time:
            time_since_last = (datetime.utcnow() - self.last_request_time).total_seconds()

            # If less than minimum interval, wait
            min_interval = 60.0 / requests_per_minute  # seconds between requests
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                logger.info(f"{self.agent_name}: Rate limiting - waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

    def get_agent_status(self) -> Dict[str, str]:
        """
        Get current status of the agent.

        Returns:
            Dictionary with agent status information
        """
        return {
            "agent_name": self.agent_name,
            "status": "ready",
            "request_count": self.request_count,
            "last_request": self.last_request_time.isoformat() if self.last_request_time else None,
            "capabilities": [
                "fetch_scholar_data",
                "fetch_publication_details",
                "validate_scholar_url",
                "rate_limiting"
            ]
        }
