# Multi-Agent Research Portfolio Dashboard - Backend

Python FastAPI backend with a three-agent architecture for fetching and processing Google Scholar publication data.

## Architecture

### Three-Agent System:

1. **Supervisor Agent** - Orchestrates workflow, validation, and retries
2. **Data Fetch Agent** - Queries Google Scholar using Tavily API
3. **Dashboard Agent** - Processes and prepares data for visualization

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Your `.env` file should contain:
```
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_SCHOLAR_URL=https://scholar.google.com/citations?user=YOUR_USER_ID
```

### 3. Run the Server

```bash
cd backend
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### `GET /` - API Information
Returns basic API information and available endpoints.

### `GET /health` - Health Check
Returns status of all agents.

### `POST /fetch_publications` - Main Pipeline Endpoint
Triggers the multi-agent pipeline to fetch and process publications.

**Request Body (optional):**
```json
{
  "scholar_url": "https://scholar.google.com/citations?user=YOUR_USER_ID"
}
```

If not provided, uses `GOOGLE_SCHOLAR_URL` from environment.

**Response:**
```json
{
  "status": "success",
  "scholar_url": "...",
  "publications": [...],
  "analytics": {...},
  "metadata": {
    "total_publications": 10,
    "data_completeness": 85.5
  }
}
```

### `GET /agent_status` - Agent Status
Returns current status of all agents.

### `GET /execution_history` - Execution History
Returns recent pipeline execution logs.

## Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Fetch publications (using env variable)
curl -X POST http://localhost:8000/fetch_publications

# Fetch publications (with custom URL)
curl -X POST http://localhost:8000/fetch_publications \
  -H "Content-Type: application/json" \
  -d '{"scholar_url": "https://scholar.google.com/citations?user=YOUR_USER_ID"}'
```

## Project Structure

```
backend/
├── agents/
│   ├── __init__.py
│   ├── data_fetch_agent.py     # Tavily-based Scholar scraper
│   ├── dashboard_agent.py      # Data processing & analytics
│   └── supervisor_agent.py     # Pipeline orchestration
├── main.py                     # FastAPI application
└── requirements.txt            # Python dependencies
```

## Features

- ✅ Async multi-agent architecture
- ✅ Automatic retry logic with exponential backoff
- ✅ Data validation and completeness checking
- ✅ Structured logging for observability
- ✅ CORS support for frontend integration
- ✅ Comprehensive error handling
- ✅ Pipeline execution history tracking
