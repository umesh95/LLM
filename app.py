"""
FastAPI server for text summarization using LangChain and LangGraph
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import aiofiles
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import uuid
from datetime import datetime

from src.basic_summarizer import BasicSummarizer
from src.langgraph_summarizer import LangGraphSummarizer
from src.config import Config
from src.async_task_manager import task_manager
from src.api_models import (
    TextSummaryRequest, FileSummaryRequest, BatchSummaryRequest,
    BasicSummaryResponse, AdvancedSummaryResponse, ErrorResponse,
    HealthResponse, BatchSummaryResponse, ConfigResponse,
    SummaryType, FileInfo, AsyncTaskResponse, TaskStatusResponse, AllTasksResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Text Summarization API",
    description="Advanced text summarization using LangChain and LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for summarizers (initialized on startup)
basic_summarizer: Optional[BasicSummarizer] = None
advanced_summarizer: Optional[LangGraphSummarizer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize summarizers on startup."""
    global basic_summarizer, advanced_summarizer
    
    try:
        Config.validate_config()
        basic_summarizer = BasicSummarizer()
        advanced_summarizer = LangGraphSummarizer()
        print("✅ Summarizers initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize summarizers: {e}")
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service information."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_available=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        Config.validate_config()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models_available=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration."""
    return ConfigResponse(
        default_model=Config.DEFAULT_MODEL,
        max_tokens=Config.MAX_TOKENS,
        temperature=Config.TEMPERATURE,
        supported_extensions=Config.SUPPORTED_EXTENSIONS,
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )


@app.post("/summarize/text")
async def summarize_text(request: TextSummaryRequest):
    """Summarize text directly."""
    try:
        if request.summary_type == SummaryType.BASIC:
            summarizer = BasicSummarizer(request.model_name)
            
            if request.custom_prompt:
                summary = summarizer.custom_summarize(request.text, request.custom_prompt)
            else:
                summary = summarizer.summarize_text(request.text, request.chain_type.value)
            
            stats = summarizer.get_summary_stats(request.text, summary)
            
            return BasicSummaryResponse(
                summary=summary,
                stats=stats,
                model_used=request.model_name or Config.DEFAULT_MODEL,
                chain_type=request.chain_type.value
            )
        
        else:  # Advanced summarization
            summarizer = LangGraphSummarizer(request.model_name)
            result = summarizer.summarize_text(request.text)
            
            return AdvancedSummaryResponse(
                final_summary=result["final_summary"],
                initial_summary=result.get("initial_summary"),
                refined_summary=result.get("refined_summary"),
                key_points=result.get("key_points", []),
                metadata=result.get("metadata", {}),
                model_used=request.model_name or Config.DEFAULT_MODEL
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )


@app.post("/summarize/file")
async def summarize_file(
    file: UploadFile = File(...),
    summary_type: SummaryType = Form(default=SummaryType.BASIC),
    model_name: Optional[str] = Form(default=None),
    chain_type: str = Form(default="map_reduce")
):
    """Summarize an uploaded file."""
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in Config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Supported: {Config.SUPPORTED_EXTENSIONS}"
        )
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Summarize based on type
        if summary_type == SummaryType.BASIC:
            summarizer = BasicSummarizer(model_name)
            summary = summarizer.summarize_file(temp_file_path, chain_type)
            
            # Read original file for stats
            async with aiofiles.open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_text = await f.read()
            
            stats = summarizer.get_summary_stats(original_text, summary)
            
            return BasicSummaryResponse(
                summary=summary,
                stats=stats,
                model_used=model_name or Config.DEFAULT_MODEL,
                chain_type=chain_type
            )
        
        else:  # Advanced summarization
            summarizer = LangGraphSummarizer(model_name)
            result = summarizer.summarize_file(temp_file_path)
            
            return AdvancedSummaryResponse(
                final_summary=result["final_summary"],
                initial_summary=result.get("initial_summary"),
                refined_summary=result.get("refined_summary"),
                key_points=result.get("key_points", []),
                metadata=result.get("metadata", {}),
                model_used=model_name or Config.DEFAULT_MODEL
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File summarization failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/summarize/batch")
async def summarize_batch(
    files: List[UploadFile] = File(...),
    summary_type: SummaryType = Form(default=SummaryType.BASIC),
    model_name: Optional[str] = Form(default=None),
    chain_type: str = Form(default="map_reduce")
):
    """Summarize multiple uploaded files."""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file_paths = []
    file_infos = []
    
    try:
        # Save all uploaded files
        for file in files:
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in Config.SUPPORTED_EXTENSIONS:
                continue  # Skip unsupported files
            
            temp_file_path = os.path.join(temp_dir, file.filename)
            temp_file_paths.append(temp_file_path)
            
            async with aiofiles.open(temp_file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            file_infos.append(FileInfo(
                filename=file.filename,
                size=len(content),
                extension=file_extension
            ))
        
        if not temp_file_paths:
            raise HTTPException(
                status_code=400,
                detail=f"No supported files found. Supported extensions: {Config.SUPPORTED_EXTENSIONS}"
            )
        
        # Summarize based on type
        if summary_type == SummaryType.BASIC:
            summarizer = BasicSummarizer(model_name)
            summary = summarizer.summarize_multiple_files(temp_file_paths, chain_type)
        else:
            summarizer = LangGraphSummarizer(model_name)
            result = summarizer.summarize_multiple_files(temp_file_paths)
            summary = result["final_summary"]
        
        return BatchSummaryResponse(
            summary=summary,
            files_processed=file_infos,
            total_files=len(file_infos),
            model_used=model_name or Config.DEFAULT_MODEL,
            summary_type=summary_type.value
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch summarization failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/models")
async def get_available_models():
    """Get list of available models."""
    return {
        "models": [
            {
                "name": "gpt-3.5-turbo",
                "description": "Fast and cost-effective",
                "max_tokens": 4096
            },
            {
                "name": "gpt-4",
                "description": "Most capable model",
                "max_tokens": 8192
            },
            {
                "name": "gpt-4-turbo-preview",
                "description": "Latest GPT-4 with improved performance",
                "max_tokens": 128000
            }
        ]
    }


def cleanup_temp_files(temp_dir: str):
    """Background task to clean up temporary files."""
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


# =============================================================================
# ASYNC LANGGRAPH ENDPOINTS
# =============================================================================

@app.post("/summarize/async/text", response_model=AsyncTaskResponse)
async def summarize_text_async(request: TextSummaryRequest):
    """Start async text summarization and return thread ID."""
    if request.summary_type != SummaryType.ADVANCED:
        raise HTTPException(
            status_code=400,
            detail="Async endpoints only support advanced (LangGraph) summarization"
        )
    
    try:
        thread_id = await task_manager.start_text_summarization(
            text=request.text,
            model_name=request.model_name
        )
        
        return AsyncTaskResponse(
            thread_id=thread_id,
            status="pending",
            message="Text summarization task started successfully",
            created_at=datetime.now().isoformat(),
            estimated_duration="2-5 minutes depending on text length"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start async summarization: {str(e)}"
        )


@app.post("/summarize/async/file", response_model=AsyncTaskResponse)
async def summarize_file_async(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None)
):
    """Start async file summarization and return thread ID."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in Config.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported: {Config.SUPPORTED_EXTENSIONS}"
            )
        
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"temp_{uuid.uuid4()}{file_extension}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        thread_id = await task_manager.start_file_summarization(
            file_path=str(temp_file_path),
            model_name=model_name
        )
        
        return AsyncTaskResponse(
            thread_id=thread_id,
            status="pending",
            message="File summarization task started successfully",
            created_at=datetime.now().isoformat(),
            estimated_duration="3-8 minutes depending on file size"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start async file summarization: {str(e)}"
        )


@app.get("/task/status/{thread_id}", response_model=TaskStatusResponse)
async def get_task_status(thread_id: str):
    """Get the status of an async task by thread ID."""
    task = task_manager.get_task_status(thread_id)
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task with thread_id '{thread_id}' not found"
        )
    
    # Get LangGraph workflow state if task is running
    workflow_state = None
    if task["status"] == "running":
        try:
            # Try to get workflow state from LangGraph
            summarizer = LangGraphSummarizer()
            workflow_state = summarizer.get_workflow_state(thread_id)
        except:
            pass  # Workflow state not available
    
    return TaskStatusResponse(
        thread_id=task["thread_id"],
        status=task["status"],
        created_at=task["created_at"],
        updated_at=task["updated_at"],
        completed_at=task.get("completed_at"),
        task_type=task["task_type"],
        input=task["input"],
        result=task.get("result"),
        error=task.get("error"),
        workflow_state=workflow_state
    )


@app.get("/tasks/all", response_model=AllTasksResponse)
async def get_all_tasks():
    """Get all async tasks and their status."""
    all_tasks = task_manager.get_all_tasks()
    
    return AllTasksResponse(
        total_tasks=len(all_tasks),
        tasks=all_tasks
    )


@app.delete("/task/{thread_id}")
async def cancel_task(thread_id: str):
    """Cancel a running task (if possible)."""
    task = task_manager.get_task_status(thread_id)
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task with thread_id '{thread_id}' not found"
        )
    
    if task["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Task is already {task['status']}, cannot cancel"
        )
    
    # Note: Actual task cancellation would require more complex implementation
    # For now, we just return a message
    return {"message": f"Task {thread_id} cancellation requested. Note: Running tasks cannot be immediately cancelled."}


@app.post("/tasks/cleanup")
async def cleanup_old_tasks():
    """Clean up old completed tasks."""
    task_manager.cleanup_completed_tasks(max_age_hours=24)
    return {"message": "Old tasks cleaned up successfully"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
        log_level="info"
    )
