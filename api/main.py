"""FastAPI application for Bank Parser Agent."""

import sys
import asyncio
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from fastapi import Request
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import aiofiles

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import BankParserAgent
from src.code_executor import CodeExecutor
from .models import (
    GenerateRequest, StatusResponse, UploadResponse, GenerateResponse,
    TestResult, LogsResponse, ValidationResponse, BenchmarkRequest,
    BenchmarkResponse, ParsersListResponse, ParserInfo, ErrorResponse,
    AgentStatus
)

# Initialize FastAPI
app = FastAPI(
    title="AI Bank Parser Agent",
    description="Automatically generate bank statement parsers using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global state
agent_instance: Optional[BankParserAgent] = None
agent_task: Optional[asyncio.Task] = None
agent_status = {
    "is_running": False,
    "current_bank": None,
    "iteration": 0,
    "max_iterations": 3,
    "status": AgentStatus.IDLE,
    "progress": 0,
    "error": None,
    "model_info": None,
    "execution_stats": {}
}


def get_agent() -> BankParserAgent:
    """Get or create agent instance."""
    global agent_instance
    if agent_instance is None:
        try:
            agent_instance = BankParserAgent()
            logger.info("Agent instance created")
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise HTTPException(status_code=500, detail=f"Agent initialization failed: {str(e)}")
    return agent_instance


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_type=type(exc).__name__,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/", response_class=FileResponse)
async def root():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current agent status."""
    global agent_status
    
    # Update status from agent if available
    if agent_instance:
        try:
            current_status = agent_instance.get_status()
            agent_status.update({
                "iteration": current_status.get("iteration", 0),
                "max_iterations": current_status.get("max_iterations", 3),
                "current_bank": current_status.get("current_bank"),
                "status": current_status.get("status", AgentStatus.IDLE),
                "progress": current_status.get("progress", 0),
                "error": current_status.get("last_error"),
                "model_info": current_status.get("model_info"),
                "execution_stats": current_status.get("execution_stats", {})
            })
        except Exception as e:
            logger.warning(f"Failed to get agent status: {e}")
    
    return StatusResponse(**agent_status)


@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(
    bank_name: str,
    pdf: UploadFile = File(..., description="Bank statement PDF file"),
    csv: UploadFile = File(..., description="Expected CSV output file")
):
    """
    Upload sample PDF and expected CSV files.
    
    Args:
        bank_name: Name of the bank
        pdf: PDF file
        csv: CSV file
    """
    try:
        # Validate file types
        if not pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF file must have .pdf extension")
        
        if not csv.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="CSV file must have .csv extension")
        
        bank_name = bank_name.lower().strip()
        if not bank_name:
            raise HTTPException(status_code=400, detail="Bank name cannot be empty")
        
        # Create data directory
        data_dir = Path("data") / bank_name
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PDF file
        pdf_path = data_dir / f"{bank_name}_sample.pdf"
        async with aiofiles.open(pdf_path, 'wb') as f:
            content = await pdf.read()
            await f.write(content)
        
        # Save CSV file
        csv_path = data_dir / f"{bank_name}_expected.csv"
        async with aiofiles.open(csv_path, 'wb') as f:
            content = await csv.read()
            await f.write(content)
        
        # Validate CSV format
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                raise HTTPException(status_code=400, detail="CSV file is empty")
        except Exception as e:
            # Clean up files
            pdf_path.unlink(missing_ok=True)
            csv_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        logger.info(f"Files uploaded for {bank_name}: PDF({pdf_path.stat().st_size} bytes), CSV({len(df)} rows)")
        
        return UploadResponse(
            status="success",
            message=f"Files uploaded successfully for {bank_name}",
            pdf_path=str(pdf_path),
            csv_path=str(csv_path),
            bank_name=bank_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def run_agent_background(bank_name: str, max_iterations: int):
    """Run agent in background task."""
    global agent_status, agent_instance
    
    try:
        agent_status["is_running"] = True
        agent_status["current_bank"] = bank_name
        agent_status["status"] = AgentStatus.STARTING
        agent_status["progress"] = 5
        agent_status["error"] = None
        
        # Get or create agent
        agent = get_agent()
        agent.reset()  # Reset state for new run
        
        # Set up paths
        data_dir = Path("data") / bank_name
        pdf_path = data_dir / f"{bank_name}_sample.pdf"
        csv_path = data_dir / f"{bank_name}_expected.csv"
        output_path = Path("custom_parsers") / f"{bank_name}_parser.py"
        
        # Validate files exist
        if not pdf_path.exists() or not csv_path.exists():
            raise FileNotFoundError("Please upload PDF and CSV files first")
        
        logger.info(f"Starting agent run for {bank_name} with {max_iterations} max iterations")
        
        # Run agent
        success, error = agent.run(
            bank_name=bank_name,
            pdf_path=str(pdf_path),
            csv_path=str(csv_path),
            output_path=str(output_path)
        )
        
        # Update final status
        if success:
            agent_status["status"] = AgentStatus.COMPLETED
            agent_status["progress"] = 100
            agent_status["error"] = None
            logger.info(f"✓ Agent completed successfully for {bank_name}")
        else:
            agent_status["status"] = AgentStatus.FAILED
            agent_status["error"] = error
            agent_status["progress"] = 0
            logger.error(f"✗ Agent failed for {bank_name}: {error}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Agent background task failed: {error_msg}")
        agent_status["status"] = AgentStatus.FAILED
        agent_status["error"] = error_msg
        agent_status["progress"] = 0
    
    finally:
        agent_status["is_running"] = False


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_parser(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate parser for specified bank.
    
    Args:
        request: Generation request with bank name and settings
    """
    global agent_status, agent_task
    
    if agent_status["is_running"]:
        raise HTTPException(
            status_code=400,
            detail="Agent is already running. Please wait for current task to complete."
        )
    
    bank_name = request.bank_name.lower().strip()
    
    # Validate input files exist
    data_dir = Path("data") / bank_name
    pdf_path = data_dir / f"{bank_name}_sample.pdf"
    csv_path = data_dir / f"{bank_name}_expected.csv"
    
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found for {bank_name}. Please upload files first."
        )
    
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found for {bank_name}. Please upload files first."
        )
    
    # Reset and start
    agent_status.update({
        "is_running": False,
        "current_bank": bank_name,
        "iteration": 0,
        "max_iterations": request.max_iterations,
        "status": AgentStatus.IDLE,
        "error": None,
        "progress": 0
    })
    
    # Start background task
    background_tasks.add_task(
        run_agent_background,
        bank_name,
        request.max_iterations
    )
    
    logger.info(f"Parser generation queued for {bank_name}")
    
    return GenerateResponse(
        status="started",
        message=f"Parser generation started for {bank_name}",
        bank_name=bank_name
    )


@app.get("/api/download/{bank_name}")
async def download_parser(bank_name: str):
    """Download generated parser code."""
    bank_name = bank_name.lower().strip()
    parser_path = Path("custom_parsers") / f"{bank_name}_parser.py"
    
    if not parser_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Parser for '{bank_name}' not found. Please generate it first."
        )
    
    return FileResponse(
        path=str(parser_path),
        media_type="text/x-python",
        filename=f"{bank_name}_parser.py",
        headers={"Content-Disposition": f"attachment; filename={bank_name}_parser.py"}
    )


@app.get("/api/test/{bank_name}", response_model=TestResult)
async def test_parser(bank_name: str):
    """Test generated parser against sample data."""
    try:
        bank_name = bank_name.lower().strip()
        parser_path = Path("custom_parsers") / f"{bank_name}_parser.py"
        pdf_path = Path("data") / bank_name / f"{bank_name}_sample.pdf"
        csv_path = Path("data") / bank_name / f"{bank_name}_expected.csv"
        
        if not parser_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Parser for '{bank_name}' not found"
            )
        
        if not pdf_path.exists() or not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Test data for '{bank_name}' not found"
            )
        
        executor = CodeExecutor()
        success, result_df, error = executor.test_parser(
            str(parser_path),
            str(pdf_path),
            str(csv_path)
        )
        
        if success and result_df is not None:
            stats = executor.get_execution_stats()
            
            return TestResult(
                status="success",
                message="Parser test passed successfully",
                rows=len(result_df),
                columns=list(result_df.columns),
                sample=result_df.head(5).to_dict(orient="records"),
                execution_time=stats.get('execution_time'),
                memory_usage=stats.get('memory_usage')
            )
        else:
            return TestResult(
                status="failed",
                message="Parser test failed",
                error=error
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test execution failed: {str(e)}")


@app.get("/api/logs", response_model=LogsResponse)
async def get_logs():
    """Get recent log entries."""
    try:
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        
        log_file = log_path / "agent.log"
        if not log_file.exists():
            return LogsResponse(logs=[], error=None)
        
        # Read last 100 lines
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            recent_logs = lines[-100:] if len(lines) > 100 else lines
        
        return LogsResponse(logs=recent_logs, error=None)
        
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        return LogsResponse(logs=[], error=str(e))


@app.get("/api/validate", response_model=ValidationResponse)
async def validate_setup():
    """Validate agent setup and dependencies."""
    try:
        agent = get_agent()
        issues = agent.validate_setup()
        
        recommendations = []
        if any("GROQ_API_KEY" in issue for issue in issues):
            recommendations.append("Get Groq API key from https://console.groq.com and add to .env file")
        
        if any("Tesseract" in issue for issue in issues):
            recommendations.append("Install Tesseract OCR and update TESSERACT_PATH in .env file")
        
        if any("Poppler" in issue for issue in issues):
            recommendations.append("Install Poppler and update POPPLER_PATH in .env file")
        
        return ValidationResponse(
            valid=len(issues) == 0,
            issues=issues,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return ValidationResponse(
            valid=False,
            issues=[f"Validation error: {str(e)}"],
            recommendations=["Check logs for detailed error information"]
        )


@app.post("/api/benchmark/{bank_name}", response_model=BenchmarkResponse)
async def benchmark_parser(bank_name: str, request: BenchmarkRequest):
    """Benchmark parser performance."""
    try:
        bank_name = bank_name.lower().strip()
        parser_path = Path("custom_parsers") / f"{bank_name}_parser.py"
        pdf_path = Path("data") / bank_name / f"{bank_name}_sample.pdf"
        
        if not parser_path.exists():
            raise HTTPException(status_code=404, detail=f"Parser for '{bank_name}' not found")
        
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"Test PDF for '{bank_name}' not found")
        
        executor = CodeExecutor()
        results = executor.benchmark_parser(
            str(parser_path),
            str(pdf_path),
            request.iterations
        )
        
        if "error" in results:
            return BenchmarkResponse(
                status="failed",
                bank_name=bank_name,
                error=results["error"]
            )
        
        return BenchmarkResponse(
            status="success",
            bank_name=bank_name,
            avg_time=results.get("avg_time"),
            min_time=results.get("min_time"),
            max_time=results.get("max_time"),
            avg_memory=results.get("avg_memory"),
            success_rate=results.get("success_rate"),
            iterations=results.get("iterations")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@app.get("/api/parsers", response_model=ParsersListResponse)
async def list_parsers():
    """List all generated parsers."""
    try:
        parsers_dir = Path("custom_parsers")
        parsers = []
        
        if parsers_dir.exists():
            for parser_file in parsers_dir.glob("*_parser.py"):
                if parser_file.name == "__init__.py":
                    continue
                
                bank_name = parser_file.stem.replace("_parser", "")
                stat = parser_file.stat()
                
                # Check if test data exists
                test_status = None
                data_dir = Path("data") / bank_name
                if (data_dir / f"{bank_name}_sample.pdf").exists():
                    test_status = "ready"
                
                parsers.append(ParserInfo(
                    bank_name=bank_name,
                    file_path=str(parser_file),
                    file_size=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    test_status=test_status
                ))
        
        return ParsersListResponse(
            parsers=parsers,
            total=len(parsers)
        )
        
    except Exception as e:
        logger.error(f"Failed to list parsers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list parsers: {str(e)}")


@app.delete("/api/parser/{bank_name}")
async def delete_parser(bank_name: str):
    """Delete a generated parser."""
    try:
        bank_name = bank_name.lower().strip()
        parser_path = Path("custom_parsers") / f"{bank_name}_parser.py"
        
        if not parser_path.exists():
            raise HTTPException(status_code=404, detail=f"Parser for '{bank_name}' not found")
        
        parser_path.unlink()
        logger.info(f"Deleted parser for {bank_name}")
        
        return {"status": "success", "message": f"Parser for '{bank_name}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete parser: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete parser: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting AI Bank Parser Agent API")
    
    # Create necessary directories
    for directory in ["data", "custom_parsers", "logs", "static"]:
        Path(directory).mkdir(exist_ok=True)
    
    # Initialize logging
    log_path = Path("logs") / "agent.log"
    logger.add(
        str(log_path),
        rotation="10 MB",
        retention="7 days",
        level="INFO"
    )
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global agent_task
    
    logger.info("Shutting down AI Bank Parser Agent API")
    
    if agent_task and not agent_task.done():
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
    
    logger.info("API shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Setup logging for development
    logger.add(
        "logs/agent.log",
        rotation="10 MB",
        level="DEBUG"
    )
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
