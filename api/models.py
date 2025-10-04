"""Pydantic models for FastAPI endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    STARTING = "starting"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    TESTING = "testing"
    CORRECTING = "correcting"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    """Request model for parser generation."""
    bank_name: str = Field(..., description="Name of the bank", min_length=1, max_length=50)
    max_iterations: Optional[int] = Field(3, description="Maximum self-correction attempts", ge=1, le=5)
    
    class Config:
        json_schema_extra = {
            "example": {
                "bank_name": "icici",
                "max_iterations": 3
            }
        }


class UploadResponse(BaseModel):
    """Response model for file upload."""
    status: str
    message: str
    pdf_path: str
    csv_path: str
    bank_name: str


class StatusResponse(BaseModel):
    """Response model for agent status."""
    is_running: bool
    current_bank: Optional[str]
    iteration: int
    max_iterations: int
    status: AgentStatus
    progress: int = Field(..., ge=0, le=100)
    error: Optional[str]
    model_info: Optional[Dict[str, Any]]
    execution_stats: Optional[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_running": True,
                "current_bank": "icici",
                "iteration": 1,
                "max_iterations": 3,
                "status": "generating",
                "progress": 45,
                "error": None,
                "model_info": {
                    "model_name": "llama-3.3-70b-versatile",
                    "temperature": 0.1,
                    "provider": "Groq"
                },
                "execution_stats": {
                    "execution_time": 2.5,
                    "memory_usage": 150.2,
                    "success": True
                }
            }
        }


class GenerateResponse(BaseModel):
    """Response model for parser generation start."""
    status: str
    message: str
    bank_name: str


class TestResult(BaseModel):
    """Model for parser test results."""
    status: str
    message: str
    # Lines 91-96 (TestResult model):
    rows: Optional[int] = None
    columns: Optional[List[str]] = None  
    sample: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Parser test passed",
                "rows": 25,
                "columns": ["Date", "Description", "Debit", "Credit", "Balance"],
                "sample": [
                    {
                        "Date": "2024-01-01",
                        "Description": "Opening Balance",
                        "Debit": "",
                        "Credit": "10000.00",
                        "Balance": "10000.00"
                    }
                ],
                "execution_time": 1.2,
                "memory_usage": 45.6
            }
        }


class LogEntry(BaseModel):
    """Model for log entries."""
    timestamp: str
    level: str
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-01 10:30:45",
                "level": "INFO",
                "message": "Parser generation started for icici"
            }
        }


class LogsResponse(BaseModel):
    """Response model for logs."""
    logs: List[str]
    error: Optional[str]


class ValidationResponse(BaseModel):
    """Response model for setup validation."""
    valid: bool
    issues: List[str]
    recommendations: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "valid": False,
                "issues": [
                    "Tesseract path not found: C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                    "GROQ_API_KEY not configured"
                ],
                "recommendations": [
                    "Install Tesseract OCR and update TESSERACT_PATH in .env",
                    "Get Groq API key from https://console.groq.com and add to .env"
                ]
            }
        }


class BenchmarkRequest(BaseModel):
    """Request model for parser benchmarking."""
    bank_name: str = Field(..., description="Name of the bank")
    iterations: Optional[int] = Field(3, description="Number of benchmark iterations", ge=1, le=10)


class BenchmarkResponse(BaseModel):
    """Response model for parser benchmarking."""
    status: str
    bank_name: str


# Lines 174-180 (BenchmarkResponse model):
    avg_time: Optional[float] = None
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    avg_memory: Optional[float] = None
    success_rate: Optional[float] = None
    iterations: Optional[int] = None
    error: Optional[str] = None


class ParserInfo(BaseModel):
    """Model for parser information."""
    bank_name: str
    file_path: str
    file_size: int
    created_at: str
    last_modified: str
    test_status: Optional[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "bank_name": "icici",
                "file_path": "custom_parsers/icici_parser.py",
                "file_size": 2048,
                "created_at": "2024-01-01T10:30:45Z",
                "last_modified": "2024-01-01T10:30:45Z",
                "test_status": "passed"
            }
        }


class ParsersListResponse(BaseModel):
    """Response model for listing parsers."""
    parsers: List[ParserInfo]
    total: int


class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str
    error_type: Optional[str]
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Parser for bank 'xyz' not found",
                "error_type": "FileNotFoundError",
                "timestamp": "2024-01-01T10:30:45Z"
            }
        }