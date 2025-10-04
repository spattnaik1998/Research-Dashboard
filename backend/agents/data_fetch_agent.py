"""
Data Fetch Agent - Responsible for fetching publication data from Google Scholar
using the scholarly library for direct Scholar scraping.
"""

import os
import asyncio
import re
from typing import Dict, List, Any, Optional
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
    Uses scholarly library to scrape and extract publication information.
    """

    def __init__(self, tavily_api_key: str = None):
        """
        Initialize the Data Fetch Agent.

        Args:
            tavily_api_key: API key for Tavily service (optional, not used in this version)
        """
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

            # Extract user ID from URL
            user_id = self._extract_user_id(scholar_url)
            logger.info(f"{self.agent_name}: Extracted user ID: {user_id}")

            # Check rate limiting
            await self._check_rate_limit()

            # Fetch publications using scholarly library
            publications = await self._fetch_publications_with_scholarly(user_id, scholar_url)

            # Update request tracking
            self.request_count += 1
            self.last_request_time = datetime.utcnow()

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

            logger.error(f"{self.agent_name}: Unexpected error - {error_message}")

            return {
                "status": "error",
                "agent": self.agent_name,
                "error_type": error_type,
                "error": error_message,
                "publications": []
            }

    async def _fetch_publications_with_scholarly(self, user_id: str, scholar_url: str) -> List[Dict[str, Any]]:
        """
        Fetch publications using the scholarly library.

        Args:
            user_id: Google Scholar user ID
            scholar_url: Original Scholar URL

        Returns:
            List of publication dictionaries
        """
        from scholarly import scholarly

        publications = []

        try:
            logger.info(f"{self.agent_name}: Searching for author with ID: {user_id}")

            # Search for author by ID
            # Note: scholarly doesn't have direct ID lookup, so we'll use search_author
            search_query = scholarly.search_author_id(user_id)
            author = scholarly.fill(search_query)

            logger.info(f"{self.agent_name}: Found author: {author.get('name', 'Unknown')}")

            # Get all publications
            author_pubs = author.get('publications', [])
            logger.info(f"{self.agent_name}: Found {len(author_pubs)} publications")

            for idx, pub in enumerate(author_pubs, 1):
                # Fill publication details (this makes additional requests)
                try:
                    filled_pub = scholarly.fill(pub)
                except Exception as e:
                    logger.warning(f"{self.agent_name}: Could not fill details for publication {idx}: {e}")
                    filled_pub = pub

                # Extract publication data
                publication = {
                    "id": f"pub_{idx}",
                    "title": filled_pub.get('bib', {}).get('title', 'Unknown Title'),
                    "url": filled_pub.get('pub_url', filled_pub.get('eprint_url', '')),
                    "content_snippet": filled_pub.get('bib', {}).get('abstract', ''),
                    "authors": self._extract_authors_from_bib(filled_pub.get('bib', {})),
                    "year": self._extract_year(filled_pub.get('bib', {})),
                    "venue": filled_pub.get('bib', {}).get('journal', filled_pub.get('bib', {}).get('venue', '')),
                    "citations": filled_pub.get('num_citations', 0),
                    "doi": None,  # scholarly doesn't always provide DOI
                    "pdf_url": filled_pub.get('eprint_url', ''),
                    "abstract": filled_pub.get('bib', {}).get('abstract', ''),
                    "keywords": [],  # We'll extract this later
                    "coauthors": self._extract_coauthors(filled_pub.get('bib', {})),
                    "scholar_link": scholar_url,
                    "scrape_timestamp": datetime.utcnow().isoformat()
                }

                publications.append(publication)

                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"{self.agent_name}: Error fetching publications with scholarly: {e}")
            raise

        return publications

    def _extract_user_id(self, scholar_url: str) -> str:
        """
        Extract user ID from Google Scholar URL.

        Args:
            scholar_url: Google Scholar profile URL

        Returns:
            User ID string

        Raises:
            InvalidScholarURLError: If user ID cannot be extracted
        """
        parsed = urlparse(scholar_url)
        query_params = parse_qs(parsed.query)

        user_id = query_params.get('user', [None])[0]

        if not user_id:
            raise InvalidScholarURLError("Could not extract user ID from URL")

        return user_id

    def _extract_authors_from_bib(self, bib: Dict) -> List[str]:
        """Extract authors list from bib data."""
        authors = []

        # Try different author field names
        author_str = bib.get('author', '')

        if isinstance(author_str, list):
            return author_str[:5]  # Limit to 5
        elif isinstance(author_str, str):
            # Split by ' and ' or ','
            if ' and ' in author_str:
                authors = [a.strip() for a in author_str.split(' and ')]
            elif ',' in author_str:
                # Handle "Last, First and Last, First" format
                parts = author_str.split(',')
                authors = [p.strip() for p in parts if p.strip() and ' and ' not in p]
            else:
                authors = [author_str]

        return authors[:5]  # Limit to 5

    def _extract_coauthors(self, bib: Dict) -> List[str]:
        """Extract co-authors (all authors except first)."""
        authors = self._extract_authors_from_bib(bib)
        return authors[1:] if len(authors) > 1 else []

    def _extract_year(self, bib: Dict) -> Optional[int]:
        """Extract publication year from bib data."""
        # Try different year field names - prioritize pub_year as it's more accurate
        year = bib.get('pub_year', bib.get('year', None))

        if year:
            try:
                return int(year)
            except (ValueError, TypeError):
                pass

        return None

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
                "validate_scholar_url",
                "rate_limiting"
            ]
        }
