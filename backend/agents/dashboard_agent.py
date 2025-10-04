"""
Dashboard Agent - Responsible for processing and preparing publication data
for visualization and dashboard presentation.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardAgent:
    """
    Agent responsible for processing fetched publication data,
    normalizing it, and preparing analytics for the dashboard.
    """

    def __init__(self):
        """Initialize the Dashboard Agent."""
        self.agent_name = "DashboardAgent"
        logger.info(f"{self.agent_name} initialized successfully")

    async def process_publications(self, raw_publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and enrich publication data for dashboard display.

        Args:
            raw_publications: List of raw publication data from Data Fetch Agent

        Returns:
            Dictionary containing processed publications and analytics
        """
        logger.info(f"{self.agent_name}: Processing {len(raw_publications)} publications")

        try:
            # Normalize and enrich each publication
            processed_pubs = []
            for pub in raw_publications:
                enriched_pub = await self._enrich_publication(pub)
                processed_pubs.append(enriched_pub)

            # Deduplicate publications
            deduplicated_pubs = await self._deduplicate_publications(processed_pubs)

            # Generate comprehensive analytics
            analytics = await self._generate_analytics(deduplicated_pubs)

            # Calculate advanced metrics
            metrics = await self._calculate_advanced_metrics(deduplicated_pubs)

            logger.info(f"{self.agent_name}: Successfully processed {len(deduplicated_pubs)} unique publications")

            return {
                "status": "success",
                "agent": self.agent_name,
                "publications": deduplicated_pubs,
                "analytics": analytics,
                "metrics": metrics,
                "total_publications": len(deduplicated_pubs),
                "data_completeness": self._calculate_completeness(deduplicated_pubs),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"{self.agent_name}: Error processing publications - {str(e)}")
            return {
                "status": "error",
                "agent": self.agent_name,
                "error": str(e),
                "publications": [],
                "analytics": {},
                "metrics": {}
            }

    async def _enrich_publication(self, publication: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single publication with extracted metadata.

        Args:
            publication: Raw publication data

        Returns:
            Enriched publication dictionary
        """
        # Extract year from title or content
        year = self._extract_year(publication.get("title", "") + " " + publication.get("content_snippet", ""))

        # Extract authors from content
        authors = self._extract_authors(publication.get("content_snippet", ""))

        # Extract venue information
        venue = self._extract_venue(publication.get("content_snippet", ""))

        # Extract keywords
        keywords = self._extract_keywords(publication.get("content_snippet", ""))

        # Extract or estimate citations
        citations = self._extract_citations(publication.get("content_snippet", ""))

        # Update publication with enriched data
        enriched = publication.copy()
        enriched.update({
            "year": year or publication.get("year"),
            "authors": authors if authors else publication.get("authors", []),
            "venue": venue or publication.get("venue"),
            "keywords": keywords if keywords else publication.get("keywords", []),
            "citations": citations if citations is not None else publication.get("citations", 0),
            "coauthors": authors[1:] if len(authors) > 1 else [],  # All authors except first
        })

        return enriched

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year from text."""
        # Look for 4-digit years between 1990 and 2030
        year_pattern = r'\b(19[9]\d|20[0-3]\d)\b'
        matches = re.findall(year_pattern, text)
        if matches:
            return int(matches[0])
        return None

    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names from text (simplified extraction)."""
        authors = []

        # Look for patterns like "FirstName LastName, FirstName LastName"
        author_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        potential_authors = re.findall(author_pattern, text)

        # Take up to 5 unique potential authors
        seen = set()
        for author in potential_authors:
            if author not in seen and len(authors) < 5:
                authors.append(author)
                seen.add(author)

        return authors

    def _extract_venue(self, text: str) -> Optional[str]:
        """Extract venue/conference/journal name from text."""
        venue_patterns = [
            r'(?:Conference|Journal|Proceedings|Workshop|Symposium)\s+(?:on|of)\s+([^.,]+)',
            r'(?:published in|appeared in)\s+([^.,]+)',
        ]

        for pattern in venue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from text."""
        common_cs_keywords = [
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
            'nlp', 'natural language processing', 'computer vision', 'data mining',
            'security', 'privacy', 'blockchain', 'cloud computing', 'distributed systems',
            'algorithms', 'optimization', 'data science', 'big data', 'iot',
            'reinforcement learning', 'database', 'software engineering'
        ]

        text_lower = text.lower()
        found_keywords = [kw for kw in common_cs_keywords if kw in text_lower]

        return found_keywords[:5]  # Limit to top 5

    def _extract_citations(self, text: str) -> Optional[int]:
        """Extract citation count from text."""
        # Look for patterns like "cited by 123", "123 citations"
        citation_patterns = [
            r'cited by (\d+)',
            r'(\d+)\s+citations?',
            r'citations?:\s*(\d+)'
        ]

        for pattern in citation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    async def _deduplicate_publications(self, publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate publications based on title similarity.

        Args:
            publications: List of publications

        Returns:
            Deduplicated list of publications
        """
        seen_titles = set()
        unique_pubs = []

        for pub in publications:
            # Normalize title for comparison
            title = pub.get("title", "").lower().strip()

            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_pubs.append(pub)

        logger.info(f"{self.agent_name}: Removed {len(publications) - len(unique_pubs)} duplicates")
        return unique_pubs

    async def _generate_analytics(self, publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate analytics and statistics from publications.

        Args:
            publications: List of processed publications

        Returns:
            Dictionary containing various analytics
        """
        # Year distribution
        years = [pub.get("year") for pub in publications if pub.get("year")]
        year_distribution = dict(Counter(years))

        # Venue distribution
        venues = [pub.get("venue") for pub in publications if pub.get("venue")]
        venue_distribution = dict(Counter(venues).most_common(10))

        # Co-author analysis
        all_coauthors = []
        for pub in publications:
            all_coauthors.extend(pub.get("coauthors", []))
        coauthor_distribution = dict(Counter(all_coauthors).most_common(10))

        # Keyword analysis
        all_keywords = []
        for pub in publications:
            all_keywords.extend(pub.get("keywords", []))
        keyword_distribution = dict(Counter(all_keywords).most_common(10))

        # Citation statistics
        citations = [pub.get("citations", 0) for pub in publications]
        total_citations = sum(citations)
        avg_citations = total_citations / len(citations) if citations else 0

        analytics = {
            "year_distribution": year_distribution,
            "venue_distribution": venue_distribution,
            "top_coauthors": coauthor_distribution,
            "top_keywords": keyword_distribution,
            "citation_stats": {
                "total_citations": total_citations,
                "average_citations": round(avg_citations, 2),
                "max_citations": max(citations) if citations else 0,
                "min_citations": min(citations) if citations else 0
            },
            "timeline": self._generate_timeline(publications),
            "publication_growth": self._calculate_publication_growth(year_distribution)
        }

        return analytics

    async def _calculate_advanced_metrics(self, publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate advanced metrics for the researcher.

        Args:
            publications: List of publications

        Returns:
            Dictionary with advanced metrics including h-index, i10-index, etc.
        """
        # Calculate h-index
        h_index = self._calculate_h_index(publications)

        # Calculate i10-index (papers with >= 10 citations)
        i10_index = sum(1 for pub in publications if pub.get("citations", 0) >= 10)

        # Get top 5 co-authors
        all_coauthors = []
        for pub in publications:
            all_coauthors.extend(pub.get("coauthors", []))
        top_coauthors = Counter(all_coauthors).most_common(5)

        # Get most cited paper
        most_cited_paper = self._get_most_cited_paper(publications)

        # Citation trends by year
        citation_trends = self._calculate_citation_trends(publications)

        # Total publications
        total_publications = len(publications)

        # Publications by year
        publications_by_year = self._get_publications_by_year(publications)

        metrics = {
            "h_index": h_index,
            "i10_index": i10_index,
            "total_publications": total_publications,
            "top_5_coauthors": [
                {"name": name, "collaborations": count}
                for name, count in top_coauthors
            ],
            "most_cited_paper": most_cited_paper,
            "citation_trends_by_year": citation_trends,
            "publications_by_year": publications_by_year,
            "total_citations": sum(pub.get("citations", 0) for pub in publications),
            "average_citations_per_paper": round(
                sum(pub.get("citations", 0) for pub in publications) / len(publications)
                if publications else 0,
                2
            )
        }

        return metrics

    def _calculate_h_index(self, publications: List[Dict[str, Any]]) -> int:
        """
        Calculate h-index: the largest number h such that h publications
        have at least h citations each.

        Args:
            publications: List of publications

        Returns:
            h-index value
        """
        # Get citation counts and sort in descending order
        citations = sorted(
            [pub.get("citations", 0) for pub in publications],
            reverse=True
        )

        h_index = 0
        for i, citation_count in enumerate(citations, start=1):
            if citation_count >= i:
                h_index = i
            else:
                break

        return h_index

    def _get_most_cited_paper(self, publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get the most cited paper.

        Args:
            publications: List of publications

        Returns:
            Dictionary with most cited paper details
        """
        if not publications:
            return {
                "title": "N/A",
                "citations": 0,
                "year": None,
                "url": None
            }

        most_cited = max(publications, key=lambda p: p.get("citations", 0))

        return {
            "title": most_cited.get("title", "Unknown"),
            "citations": most_cited.get("citations", 0),
            "year": most_cited.get("year"),
            "url": most_cited.get("url"),
            "authors": most_cited.get("authors", []),
            "venue": most_cited.get("venue")
        }

    def _calculate_citation_trends(self, publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate citation trends by year.

        Args:
            publications: List of publications

        Returns:
            List of citation counts by year
        """
        # Group citations by publication year
        year_citations = {}
        for pub in publications:
            year = pub.get("year")
            citations = pub.get("citations", 0)
            if year:
                if year not in year_citations:
                    year_citations[year] = 0
                year_citations[year] += citations

        # Convert to sorted list
        trends = [
            {"year": year, "citations": citations}
            for year, citations in sorted(year_citations.items())
        ]

        return trends

    def _get_publications_by_year(self, publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get publication count by year.

        Args:
            publications: List of publications

        Returns:
            List of publication counts by year
        """
        years = [pub.get("year") for pub in publications if pub.get("year")]
        year_counts = Counter(years)

        by_year = [
            {"year": year, "count": count}
            for year, count in sorted(year_counts.items())
        ]

        return by_year

    def _generate_timeline(self, publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate publication timeline data."""
        years = [pub.get("year") for pub in publications if pub.get("year")]
        if not years:
            return []

        year_counts = Counter(years)
        min_year = min(years)
        max_year = max(years)

        timeline = []
        for year in range(min_year, max_year + 1):
            timeline.append({
                "year": year,
                "count": year_counts.get(year, 0)
            })

        return timeline

    def _calculate_publication_growth(self, year_distribution: Dict[int, int]) -> Dict[str, Any]:
        """Calculate year-over-year publication growth."""
        if not year_distribution:
            return {"trend": "no_data", "growth_rate": 0}

        sorted_years = sorted(year_distribution.items())
        if len(sorted_years) < 2:
            return {"trend": "insufficient_data", "growth_rate": 0}

        recent_year_count = sorted_years[-1][1]
        previous_year_count = sorted_years[-2][1]

        growth_rate = ((recent_year_count - previous_year_count) / previous_year_count * 100) if previous_year_count > 0 else 0

        return {
            "trend": "growing" if growth_rate > 0 else "declining" if growth_rate < 0 else "stable",
            "growth_rate": round(growth_rate, 2),
            "recent_year": sorted_years[-1][0],
            "recent_year_count": recent_year_count
        }

    def _calculate_completeness(self, publications: List[Dict[str, Any]]) -> float:
        """
        Calculate data completeness percentage.

        Args:
            publications: List of publications

        Returns:
            Completeness percentage (0-100)
        """
        if not publications:
            return 0.0

        complete_count = 0
        for pub in publications:
            has_title = bool(pub.get("title"))
            has_year = bool(pub.get("year"))
            has_authors = bool(pub.get("authors"))

            if has_title and has_year and has_authors:
                complete_count += 1

        completeness = (complete_count / len(publications)) * 100
        return round(completeness, 2)

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
                "process_publications",
                "deduplicate_publications",
                "generate_analytics",
                "calculate_h_index",
                "calculate_i10_index",
                "calculate_citation_trends",
                "identify_top_coauthors",
                "find_most_cited_paper",
                "calculate_completeness"
            ]
        }
