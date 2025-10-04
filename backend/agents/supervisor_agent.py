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
            Dictionary containing the complete pipeline results with metrics and analytics
        """
        pipeline_start = datetime.utcnow()
        logger.info("=" * 80)
        logger.info(f"{self.agent_name}: STARTING WORKFLOW EXECUTION")
        logger.info(f"{self.agent_name}: Scholar URL: {scholar_url}")
        logger.info(f"{self.agent_name}: Start Time: {pipeline_start.isoformat()}")
        logger.info("=" * 80)

        pipeline_result = {
            "status": "started",
            "scholar_url": scholar_url,
            "pipeline_start": pipeline_start.isoformat(),
            "stages": {},
            "execution_log": []
        }

        try:
            # Stage 1: Fetch data from Google Scholar
            logger.info(f"\n{self.agent_name}: STAGE 1 - DATA FETCH")
            logger.info(f"{self.agent_name}: Delegating to Data Fetch Agent...")

            self._log_step(pipeline_result, "Starting data fetch from Google Scholar")

            fetch_result = await self._execute_with_retry(
                self._fetch_stage,
                scholar_url,
                stage_name="data_fetch"
            )

            pipeline_result["stages"]["fetch"] = {
                "status": fetch_result.get("status"),
                "publications_count": fetch_result.get("publications_count", 0),
                "timestamp": fetch_result.get("timestamp")
            }

            if fetch_result.get("status") != "success":
                error_msg = f"Data fetch stage failed: {fetch_result.get('error')}"
                logger.error(f"{self.agent_name}: {error_msg}")
                self._log_step(pipeline_result, error_msg, level="error")
                raise Exception(error_msg)

            logger.info(f"{self.agent_name}: ✓ Data Fetch Complete - {fetch_result.get('publications_count', 0)} publications fetched")
            self._log_step(pipeline_result, f"Data fetch successful: {fetch_result.get('publications_count', 0)} publications")

            # Stage 2: Validate fetched data
            logger.info(f"\n{self.agent_name}: STAGE 2 - DATA VALIDATION")
            logger.info(f"{self.agent_name}: Validating fetched publications...")

            self._log_step(pipeline_result, "Validating fetched data quality")

            validation_result = await self._validate_fetched_data(fetch_result)
            pipeline_result["stages"]["validation"] = validation_result

            if not validation_result.get("is_valid"):
                warnings = validation_result.get("warnings", [])
                logger.warning(f"{self.agent_name}: ⚠ Validation warnings detected:")
                for warning in warnings:
                    logger.warning(f"{self.agent_name}:   - {warning}")
                self._log_step(pipeline_result, f"Validation warnings: {', '.join(warnings)}", level="warning")
            else:
                logger.info(f"{self.agent_name}: ✓ Validation Complete - Data quality acceptable")
                self._log_step(pipeline_result, "Data validation passed")

            # Stage 3: Process and prepare dashboard data
            logger.info(f"\n{self.agent_name}: STAGE 3 - DASHBOARD PROCESSING")
            logger.info(f"{self.agent_name}: Delegating to Dashboard Agent...")

            self._log_step(pipeline_result, "Processing publications for dashboard metrics")

            dashboard_result = await self._execute_with_retry(
                self._dashboard_stage,
                fetch_result.get("publications", []),
                stage_name="dashboard_processing"
            )

            pipeline_result["stages"]["dashboard"] = {
                "status": dashboard_result.get("status"),
                "total_publications": dashboard_result.get("total_publications", 0),
                "data_completeness": dashboard_result.get("data_completeness", 0),
                "h_index": dashboard_result.get("metrics", {}).get("h_index", 0),
                "total_citations": dashboard_result.get("metrics", {}).get("total_citations", 0)
            }

            if dashboard_result.get("status") != "success":
                error_msg = f"Dashboard stage failed: {dashboard_result.get('error')}"
                logger.error(f"{self.agent_name}: {error_msg}")
                self._log_step(pipeline_result, error_msg, level="error")
                raise Exception(error_msg)

            logger.info(f"{self.agent_name}: ✓ Dashboard Processing Complete")
            logger.info(f"{self.agent_name}:   - h-index: {dashboard_result.get('metrics', {}).get('h_index', 0)}")
            logger.info(f"{self.agent_name}:   - Total Citations: {dashboard_result.get('metrics', {}).get('total_citations', 0)}")
            logger.info(f"{self.agent_name}:   - Data Completeness: {dashboard_result.get('data_completeness', 0)}%")

            self._log_step(pipeline_result, f"Dashboard processing successful - h-index: {dashboard_result.get('metrics', {}).get('h_index', 0)}")

            # Stage 4: Final assembly and result preparation
            pipeline_end = datetime.utcnow()
            pipeline_duration = (pipeline_end - pipeline_start).total_seconds()

            logger.info(f"\n{self.agent_name}: STAGE 4 - FINAL ASSEMBLY")
            logger.info(f"{self.agent_name}: Preparing final response...")

            self._log_step(pipeline_result, "Assembling final results")

            final_result = {
                "status": "success",
                "agent": self.agent_name,
                "scholar_url": scholar_url,
                "pipeline_start": pipeline_start.isoformat(),
                "pipeline_end": pipeline_end.isoformat(),
                "duration_seconds": round(pipeline_duration, 2),
                "publications": dashboard_result.get("publications", []),
                "metrics": dashboard_result.get("metrics", {}),
                "analytics": dashboard_result.get("analytics", {}),
                "summary": {
                    "total_publications": dashboard_result.get("total_publications", 0),
                    "data_completeness": dashboard_result.get("data_completeness", 0),
                    "h_index": dashboard_result.get("metrics", {}).get("h_index", 0),
                    "i10_index": dashboard_result.get("metrics", {}).get("i10_index", 0),
                    "total_citations": dashboard_result.get("metrics", {}).get("total_citations", 0),
                    "validation_warnings": validation_result.get("warnings", [])
                },
                "stages": pipeline_result["stages"],
                "execution_log": pipeline_result["execution_log"]
            }

            logger.info("=" * 80)
            logger.info(f"{self.agent_name}: ✓ WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info(f"{self.agent_name}: Duration: {pipeline_duration:.2f}s")
            logger.info(f"{self.agent_name}: Publications: {final_result['summary']['total_publications']}")
            logger.info(f"{self.agent_name}: h-index: {final_result['summary']['h_index']}")
            logger.info(f"{self.agent_name}: Total Citations: {final_result['summary']['total_citations']}")
            logger.info("=" * 80)

            self._log_step(pipeline_result, f"Pipeline completed successfully in {pipeline_duration:.2f}s", level="success")
            self._log_execution(final_result)

            return final_result

        except Exception as e:
            pipeline_end = datetime.utcnow()
            pipeline_duration = (pipeline_end - pipeline_start).total_seconds()

            logger.error("=" * 80)
            logger.error(f"{self.agent_name}: ✗ WORKFLOW FAILED")
            logger.error(f"{self.agent_name}: Error: {str(e)}")
            logger.error(f"{self.agent_name}: Duration before failure: {pipeline_duration:.2f}s")
            logger.error("=" * 80)

            self._log_step(pipeline_result, f"Pipeline failed: {str(e)}", level="error")

            error_result = {
                "status": "error",
                "agent": self.agent_name,
                "error": str(e),
                "error_type": self._classify_error(str(e)),
                "scholar_url": scholar_url,
                "pipeline_start": pipeline_start.isoformat(),
                "pipeline_end": pipeline_end.isoformat(),
                "duration_seconds": round(pipeline_duration, 2),
                "stages": pipeline_result.get("stages", {}),
                "execution_log": pipeline_result.get("execution_log", [])
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

    def _log_step(self, pipeline_result: Dict[str, Any], message: str, level: str = "info") -> None:
        """
        Log a step in the pipeline execution.

        Args:
            pipeline_result: The pipeline result dictionary to update
            message: Log message
            level: Log level (info, warning, error, success)
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message
        }
        pipeline_result["execution_log"].append(log_entry)

    def _classify_error(self, error_message: str) -> str:
        """
        Classify error type based on error message.

        Args:
            error_message: The error message string

        Returns:
            Error classification
        """
        error_lower = error_message.lower()

        if "data fetch" in error_lower:
            return "data_fetch_error"
        elif "dashboard" in error_lower:
            return "dashboard_processing_error"
        elif "validation" in error_lower:
            return "validation_error"
        elif "rate limit" in error_lower or "429" in error_lower:
            return "rate_limit_error"
        elif "authentication" in error_lower or "401" in error_lower:
            return "authentication_error"
        elif "timeout" in error_lower:
            return "timeout_error"
        else:
            return "unknown_error"

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
            "publications_count": result.get("summary", {}).get("total_publications", 0),
            "h_index": result.get("summary", {}).get("h_index", 0)
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
