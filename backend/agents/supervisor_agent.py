"""
Supervisor Agent - Orchestrates the workflow between Data Fetch Agent and Dashboard Agent.
Handles validation, retries, and coordination of the multi-agent pipeline.
"""

import asyncio
from typing import Dict, List, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Supervisor Agent that orchestrates the entire publication fetching
    and processing pipeline. Coordinates between DataFetchAgent and DashboardAgent.
    """

    def __init__(self, data_fetch_agent, dashboard_agent, max_retries: int = 3):
        """
        Initialize the Supervisor Agent.

        Args:
            data_fetch_agent: Instance of DataFetchAgent
            dashboard_agent: Instance of DashboardAgent
            max_retries: Maximum number of retry attempts for failed operations
        """
        self.data_fetch_agent = data_fetch_agent
        self.dashboard_agent = dashboard_agent
        self.max_retries = max_retries
        self.agent_name = "SupervisorAgent"
        self.execution_log = []
        logger.info(f"{self.agent_name} initialized successfully")

    async def execute_pipeline(self, scholar_url: str) -> Dict[str, Any]:
        """
        Execute the complete pipeline: fetch -> process -> prepare dashboard data.

        Args:
            scholar_url: Google Scholar profile URL

        Returns:
            Dictionary containing the complete pipeline results
        """
        pipeline_start = datetime.utcnow()
        logger.info(f"{self.agent_name}: Starting pipeline execution for {scholar_url}")

        pipeline_result = {
            "status": "started",
            "scholar_url": scholar_url,
            "pipeline_start": pipeline_start.isoformat(),
            "stages": {}
        }

        try:
            # Stage 1: Fetch data from Google Scholar
            fetch_result = await self._execute_with_retry(
                self._fetch_stage,
                scholar_url,
                stage_name="data_fetch"
            )

            pipeline_result["stages"]["fetch"] = {
                "status": fetch_result.get("status"),
                "publications_count": fetch_result.get("publications_count", 0)
            }

            if fetch_result.get("status") != "success":
                raise Exception(f"Data fetch stage failed: {fetch_result.get('error')}")

            # Stage 2: Validate fetched data
            validation_result = await self._validate_fetched_data(fetch_result)
            pipeline_result["stages"]["validation"] = validation_result

            if not validation_result.get("is_valid"):
                logger.warning(f"{self.agent_name}: Validation warnings: {validation_result.get('warnings')}")

            # Stage 3: Process and prepare dashboard data
            dashboard_result = await self._execute_with_retry(
                self._dashboard_stage,
                fetch_result.get("publications", []),
                stage_name="dashboard_processing"
            )

            pipeline_result["stages"]["dashboard"] = {
                "status": dashboard_result.get("status"),
                "total_publications": dashboard_result.get("total_publications", 0),
                "data_completeness": dashboard_result.get("data_completeness", 0)
            }

            if dashboard_result.get("status") != "success":
                raise Exception(f"Dashboard stage failed: {dashboard_result.get('error')}")

            # Final assembly
            pipeline_end = datetime.utcnow()
            pipeline_duration = (pipeline_end - pipeline_start).total_seconds()

            final_result = {
                "status": "success",
                "agent": self.agent_name,
                "scholar_url": scholar_url,
                "pipeline_start": pipeline_start.isoformat(),
                "pipeline_end": pipeline_end.isoformat(),
                "duration_seconds": round(pipeline_duration, 2),
                "publications": dashboard_result.get("publications", []),
                "analytics": dashboard_result.get("analytics", {}),
                "metadata": {
                    "total_publications": dashboard_result.get("total_publications", 0),
                    "data_completeness": dashboard_result.get("data_completeness", 0),
                    "validation_warnings": validation_result.get("warnings", [])
                },
                "stages": pipeline_result["stages"]
            }

            logger.info(f"{self.agent_name}: Pipeline completed successfully in {pipeline_duration:.2f}s")
            self._log_execution(final_result)

            return final_result

        except Exception as e:
            logger.error(f"{self.agent_name}: Pipeline failed - {str(e)}")
            pipeline_end = datetime.utcnow()
            error_result = {
                "status": "error",
                "agent": self.agent_name,
                "error": str(e),
                "scholar_url": scholar_url,
                "pipeline_start": pipeline_start.isoformat(),
                "pipeline_end": pipeline_end.isoformat(),
                "stages": pipeline_result.get("stages", {})
            }
            self._log_execution(error_result)
            return error_result

    async def _fetch_stage(self, scholar_url: str) -> Dict[str, Any]:
        """
        Execute the data fetch stage.

        Args:
            scholar_url: Google Scholar profile URL

        Returns:
            Result from DataFetchAgent
        """
        logger.info(f"{self.agent_name}: Executing fetch stage")
        return await self.data_fetch_agent.fetch_scholar_data(scholar_url)

    async def _dashboard_stage(self, publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the dashboard processing stage.

        Args:
            publications: List of publications from fetch stage

        Returns:
            Result from DashboardAgent
        """
        logger.info(f"{self.agent_name}: Executing dashboard stage")
        return await self.dashboard_agent.process_publications(publications)

    async def _execute_with_retry(
        self,
        async_func,
        *args,
        stage_name: str = "unknown",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute an async function with retry logic.

        Args:
            async_func: Async function to execute
            *args: Positional arguments for the function
            stage_name: Name of the stage for logging
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"{self.agent_name}: {stage_name} - Attempt {attempt}/{self.max_retries}")
                result = await async_func(*args, **kwargs)

                if result.get("status") == "success":
                    logger.info(f"{self.agent_name}: {stage_name} succeeded on attempt {attempt}")
                    return result
                else:
                    last_error = result.get("error", "Unknown error")
                    logger.warning(f"{self.agent_name}: {stage_name} failed on attempt {attempt}: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"{self.agent_name}: {stage_name} exception on attempt {attempt}: {last_error}")

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                wait_time = 2 ** attempt  # 2, 4, 8 seconds
                logger.info(f"{self.agent_name}: Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

        # All retries exhausted
        logger.error(f"{self.agent_name}: {stage_name} failed after {self.max_retries} attempts")
        return {
            "status": "error",
            "error": f"Failed after {self.max_retries} attempts. Last error: {last_error}",
            "stage": stage_name
        }

    async def _validate_fetched_data(self, fetch_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the fetched data quality.

        Args:
            fetch_result: Result from data fetch stage

        Returns:
            Validation result dictionary
        """
        logger.info(f"{self.agent_name}: Validating fetched data")

        publications = fetch_result.get("publications", [])
        warnings = []
        is_valid = True

        # Check if any publications were fetched
        if len(publications) == 0:
            warnings.append("No publications fetched")
            is_valid = False

        # Check data quality
        pubs_with_title = sum(1 for pub in publications if pub.get("title"))
        pubs_with_year = sum(1 for pub in publications if pub.get("year"))
        pubs_with_authors = sum(1 for pub in publications if pub.get("authors"))

        if len(publications) > 0:
            title_coverage = (pubs_with_title / len(publications)) * 100
            year_coverage = (pubs_with_year / len(publications)) * 100
            author_coverage = (pubs_with_authors / len(publications)) * 100

            if title_coverage < 90:
                warnings.append(f"Low title coverage: {title_coverage:.1f}%")

            if year_coverage < 50:
                warnings.append(f"Low year coverage: {year_coverage:.1f}%")

            if author_coverage < 50:
                warnings.append(f"Low author coverage: {author_coverage:.1f}%")

        validation_result = {
            "is_valid": is_valid,
            "publications_count": len(publications),
            "coverage": {
                "titles": pubs_with_title,
                "years": pubs_with_year,
                "authors": pubs_with_authors
            },
            "warnings": warnings
        }

        logger.info(f"{self.agent_name}: Validation result - Valid: {is_valid}, Warnings: {len(warnings)}")
        return validation_result

    def _log_execution(self, result: Dict[str, Any]) -> None:
        """
        Log pipeline execution for observability.

        Args:
            result: Pipeline execution result
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": result.get("status"),
            "scholar_url": result.get("scholar_url"),
            "duration": result.get("duration_seconds"),
            "publications_count": result.get("metadata", {}).get("total_publications", 0)
        }

        self.execution_log.append(log_entry)

        # Keep only last 100 executions
        if len(self.execution_log) > 100:
            self.execution_log = self.execution_log[-100:]

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get pipeline execution history.

        Returns:
            List of execution log entries
        """
        return self.execution_log

    def get_agent_status(self) -> Dict[str, str]:
        """
        Get current status of the supervisor agent.

        Returns:
            Dictionary with agent status information
        """
        return {
            "agent_name": self.agent_name,
            "status": "ready",
            "max_retries": self.max_retries,
            "execution_count": len(self.execution_log),
            "capabilities": [
                "execute_pipeline",
                "validate_fetched_data",
                "retry_failed_operations",
                "get_execution_history"
            ]
        }
