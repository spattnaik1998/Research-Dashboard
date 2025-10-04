"""
FastAPI Backend - Main application entry point for the Multi-Agent Research Portfolio Dashboard.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional
from dotenv import load_dotenv
import logging

from agents import DataFetchAgent, DashboardAgent, SupervisorAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Research Portfolio Dashboard API",
    description="API for fetching and processing Google Scholar publication data using multi-agent architecture",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_SCHOLAR_URL = os.getenv("GOOGLE_SCHOLAR_URL")

if not TAVILY_API_KEY:
    logger.error("TAVILY_API_KEY not found in environment variables")
    raise ValueError("TAVILY_API_KEY must be set in .env file")

# Initialize agents
logger.info("Initializing agents...")
data_fetch_agent = DataFetchAgent(tavily_api_key=TAVILY_API_KEY)
dashboard_agent = DashboardAgent()
supervisor_agent = SupervisorAgent(
    data_fetch_agent=data_fetch_agent,
    dashboard_agent=dashboard_agent,
    max_retries=3
)
logger.info("All agents initialized successfully")


# Pydantic models for request/response
class FetchPublicationsRequest(BaseModel):
    """Request model for fetching publications."""
    scholar_url: Optional[HttpUrl] = None

    class Config:
        json_schema_extra = {
            "example": {
                "scholar_url": "https://scholar.google.com/citations?user=zHT6Ok0AAAAJ&hl=en"
            }
        }


class ProcessDashboardRequest(BaseModel):
    """Request model for processing dashboard data."""
    publications: list

    class Config:
        json_schema_extra = {
            "example": {
                "publications": [
                    {
                        "title": "Sample Paper",
                        "authors": ["Author One", "Author Two"],
                        "year": 2023,
                        "citations": 10,
                        "venue": "Sample Conference"
                    }
                ]
            }
        }


class FetchPublicationsResponse(BaseModel):
    """Response model for fetch publications endpoint."""
    status: str
    scholar_url: str
    publications: list
    analytics: dict
    metadata: dict


@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "Multi-Agent Research Portfolio Dashboard API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "fetch_publications": "/fetch_publications",
            "process_dashboard": "/process_dashboard",
            "agent_status": "/agent_status",
            "execution_history": "/execution_history"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agents": {
            "supervisor": supervisor_agent.get_agent_status(),
            "data_fetch": data_fetch_agent.get_agent_status(),
            "dashboard": dashboard_agent.get_agent_status()
        }
    }


@app.post("/fetch_publications")
async def fetch_publications(request: FetchPublicationsRequest = None):
    """
    Main endpoint to trigger the publication fetching pipeline.

    This endpoint:
    1. Uses the Supervisor Agent to orchestrate the workflow
    2. Delegates to Data Fetch Agent to query Google Scholar via Tavily API
    3. Delegates to Dashboard Agent to process and prepare the data
    4. Returns structured JSON with publications and analytics

    Args:
        request: Optional request body with scholar_url. If not provided, uses env variable.

    Returns:
        JSON response with publications, analytics, and metadata
    """
    try:
        # Determine which scholar URL to use
        if request and request.scholar_url:
            scholar_url = str(request.scholar_url)
            logger.info(f"Using scholar URL from request: {scholar_url}")
        elif GOOGLE_SCHOLAR_URL:
            scholar_url = GOOGLE_SCHOLAR_URL
            logger.info(f"Using scholar URL from environment: {scholar_url}")
        else:
            raise HTTPException(
                status_code=400,
                detail="No scholar_url provided in request or environment variables"
            )

        logger.info(f"Starting publication fetch pipeline for: {scholar_url}")

        # Execute the multi-agent pipeline
        result = await supervisor_agent.execute_pipeline(scholar_url)

        if result.get("status") == "error":
            logger.error(f"Pipeline failed: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline execution failed: {result.get('error')}"
            )

        logger.info(f"Pipeline completed successfully. Found {len(result.get('publications', []))} publications")

        return {
            "status": result.get("status"),
            "scholar_url": result.get("scholar_url"),
            "publications": result.get("publications", []),
            "analytics": result.get("analytics", {}),
            "metadata": result.get("metadata", {}),
            "pipeline_info": {
                "duration_seconds": result.get("duration_seconds"),
                "pipeline_start": result.get("pipeline_start"),
                "pipeline_end": result.get("pipeline_end"),
                "stages": result.get("stages", {})
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in fetch_publications: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/process_dashboard")
async def process_dashboard(request: ProcessDashboardRequest):
    """
    Process publication data and generate dashboard metrics.

    This endpoint takes raw publication data and returns:
    - Total publications count
    - h-index and i10-index
    - Citation trends by year
    - Top 5 co-authors
    - Most cited paper
    - Publications by year
    - Various analytics ready for frontend visualization

    Args:
        request: Request body containing list of publications

    Returns:
        JSON with comprehensive metrics and analytics
    """
    try:
        logger.info(f"Processing {len(request.publications)} publications for dashboard")

        if not request.publications:
            raise HTTPException(
                status_code=400,
                detail="Publications list cannot be empty"
            )

        # Process publications using Dashboard Agent
        result = await dashboard_agent.process_publications(request.publications)

        if result.get("status") == "error":
            logger.error(f"Dashboard processing failed: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Dashboard processing failed: {result.get('error')}"
            )

        logger.info(f"Dashboard processing completed successfully. Metrics generated.")

        # Return structured response for frontend
        return {
            "status": "success",
            "metrics": result.get("metrics", {}),
            "analytics": result.get("analytics", {}),
            "summary": {
                "total_publications": result.get("total_publications", 0),
                "data_completeness": result.get("data_completeness", 0),
                "h_index": result.get("metrics", {}).get("h_index", 0),
                "i10_index": result.get("metrics", {}).get("i10_index", 0),
                "total_citations": result.get("metrics", {}).get("total_citations", 0)
            },
            "publications": result.get("publications", []),
            "timestamp": result.get("timestamp")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_dashboard: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/agent_status")
async def get_agent_status():
    """
    Get current status of all agents.

    Returns:
        Status information for Supervisor, Data Fetch, and Dashboard agents
    """
    return {
        "supervisor": supervisor_agent.get_agent_status(),
        "data_fetch": data_fetch_agent.get_agent_status(),
        "dashboard": dashboard_agent.get_agent_status()
    }


@app.get("/execution_history")
async def get_execution_history():
    """
    Get pipeline execution history.

    Returns:
        List of recent pipeline executions with timestamps and results
    """
    history = supervisor_agent.get_execution_history()
    return {
        "total_executions": len(history),
        "history": history
    }


@app.get("/config")
async def get_config():
    """
    Get current configuration (without exposing sensitive data).

    Returns:
        Configuration information
    """
    return {
        "tavily_api_configured": bool(TAVILY_API_KEY),
        "default_scholar_url_configured": bool(GOOGLE_SCHOLAR_URL),
        "max_retries": supervisor_agent.max_retries
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return {
        "status": "error",
        "error": exc.detail,
        "status_code": exc.status_code
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
