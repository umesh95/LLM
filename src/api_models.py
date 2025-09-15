"""Pydantic models for FastAPI request/response schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class SummaryType(str, Enum):
    """Supported summarization types."""
    BASIC = "basic"
    ADVANCED = "advanced"


class ChainType(str, Enum):
    """LangChain summarization chain types."""
    STUFF = "stuff"
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"


class TextSummaryRequest(BaseModel):
    """Request model for text summarization."""
    text: str = Field(..., description="Text to summarize", min_length=10)
    summary_type: SummaryType = Field(default=SummaryType.BASIC, description="Type of summarization")
    model_name: Optional[str] = Field(default=None, description="LLM model to use")
    custom_prompt: Optional[str] = Field(default=None, description="Custom summarization prompt")
    chain_type: ChainType = Field(default=ChainType.MAP_REDUCE, description="Chain type for basic summarization")


class FileSummaryRequest(BaseModel):
    """Request model for file summarization."""
    summary_type: SummaryType = Field(default=SummaryType.BASIC, description="Type of summarization")
    model_name: Optional[str] = Field(default=None, description="LLM model to use")
    chain_type: ChainType = Field(default=ChainType.MAP_REDUCE, description="Chain type for basic summarization")


class BatchSummaryRequest(BaseModel):
    """Request model for batch summarization."""
    file_extensions: List[str] = Field(default=["txt", "md", "pdf", "docx"], description="File extensions to include")
    summary_type: SummaryType = Field(default=SummaryType.BASIC, description="Type of summarization")
    model_name: Optional[str] = Field(default=None, description="LLM model to use")
    chain_type: ChainType = Field(default=ChainType.MAP_REDUCE, description="Chain type for basic summarization")


class SummaryStats(BaseModel):
    """Statistics about the summarization."""
    original_words: int = Field(..., description="Number of words in original text")
    summary_words: int = Field(..., description="Number of words in summary")
    compression_ratio: float = Field(..., description="Compression ratio (summary/original)")
    compression_percentage: str = Field(..., description="Compression percentage")


class BasicSummaryResponse(BaseModel):
    """Response model for basic summarization."""
    summary: str = Field(..., description="Generated summary")
    stats: Optional[SummaryStats] = Field(default=None, description="Summarization statistics")
    model_used: str = Field(..., description="LLM model used")
    chain_type: str = Field(..., description="Chain type used")


class AdvancedSummaryResponse(BaseModel):
    """Response model for advanced summarization."""
    final_summary: str = Field(..., description="Final polished summary")
    initial_summary: Optional[str] = Field(default=None, description="Initial summary")
    refined_summary: Optional[str] = Field(default=None, description="Refined summary")
    key_points: List[str] = Field(default=[], description="Extracted key points")
    metadata: Dict[str, Any] = Field(default={}, description="Workflow metadata")
    model_used: str = Field(..., description="LLM model used")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_available: List[str] = Field(..., description="Available LLM models")


class FileInfo(BaseModel):
    """File information model."""
    filename: str = Field(..., description="Name of the file")
    size: int = Field(..., description="File size in bytes")
    extension: str = Field(..., description="File extension")


class BatchSummaryResponse(BaseModel):
    """Response model for batch summarization."""
    summary: str = Field(..., description="Combined summary")
    files_processed: List[FileInfo] = Field(..., description="Information about processed files")
    total_files: int = Field(..., description="Total number of files processed")
    model_used: str = Field(..., description="LLM model used")
    summary_type: str = Field(..., description="Type of summarization used")


class ConfigResponse(BaseModel):
    """Configuration response model."""
    default_model: str = Field(..., description="Default LLM model")
    max_tokens: int = Field(..., description="Maximum tokens per request")
    temperature: float = Field(..., description="LLM temperature setting")
    supported_extensions: List[str] = Field(..., description="Supported file extensions")
    chunk_size: int = Field(..., description="Text chunk size")
    chunk_overlap: int = Field(..., description="Text chunk overlap")


class AsyncTaskResponse(BaseModel):
    """Response model for async task initiation."""
    thread_id: str = Field(..., description="Unique thread ID for tracking the task")
    status: str = Field(..., description="Current task status")
    message: str = Field(..., description="Status message")
    created_at: str = Field(..., description="Task creation timestamp")
    estimated_duration: Optional[str] = Field(default=None, description="Estimated completion time")


class TaskStatusResponse(BaseModel):
    """Response model for task status check."""
    thread_id: str = Field(..., description="Thread ID")
    status: str = Field(..., description="Current task status")
    created_at: str = Field(..., description="Task creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    completed_at: Optional[str] = Field(default=None, description="Task completion timestamp")
    task_type: str = Field(..., description="Type of task")
    input: Dict[str, Any] = Field(..., description="Task input parameters")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result (if completed)")
    error: Optional[str] = Field(default=None, description="Error message (if failed)")
    workflow_state: Optional[Dict[str, Any]] = Field(default=None, description="Current LangGraph workflow state")


class AllTasksResponse(BaseModel):
    """Response model for all tasks overview."""
    total_tasks: int = Field(..., description="Total number of tasks")
    tasks: Dict[str, Dict[str, Any]] = Field(..., description="All tasks with their status")
