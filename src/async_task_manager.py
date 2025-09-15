"""Async task manager for background LangGraph processing."""

import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from .langgraph_summarizer import LangGraphSummarizer
from .config import Config


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AsyncTaskManager:
    """Manages async background tasks for LangGraph processing."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize the async task manager."""
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
    
    def _generate_thread_id(self) -> str:
        """Generate a unique thread ID."""
        return str(uuid.uuid4())
    
    def _update_task_status(self, thread_id: str, status: TaskStatus, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """Update task status in a thread-safe manner."""
        with self._lock:
            if thread_id in self.tasks:
                self.tasks[thread_id]["status"] = status
                self.tasks[thread_id]["updated_at"] = datetime.now().isoformat()
                
                if result:
                    self.tasks[thread_id]["result"] = result
                if error:
                    self.tasks[thread_id]["error"] = error
                
                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    self.tasks[thread_id]["completed_at"] = datetime.now().isoformat()
    
    def _run_summarization_task(self, thread_id: str, summarizer: LangGraphSummarizer, 
                               method_name: str, *args, **kwargs):
        """Run summarization task in background thread."""
        try:
            self._update_task_status(thread_id, TaskStatus.RUNNING)
            
            # Get the method and run it
            method = getattr(summarizer, method_name)
            result = method(*args, **kwargs)
            
            self._update_task_status(thread_id, TaskStatus.COMPLETED, result=result)
            
        except Exception as e:
            error_msg = f"Task failed: {str(e)}"
            self._update_task_status(thread_id, TaskStatus.FAILED, error=error_msg)
    
    async def start_text_summarization(self, text: str, model_name: Optional[str] = None) -> str:
        """Start async text summarization and return thread ID."""
        thread_id = self._generate_thread_id()
        
        # Initialize task
        with self._lock:
            self.tasks[thread_id] = {
                "thread_id": thread_id,
                "task_type": "text_summarization",
                "status": TaskStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "input": {"text_length": len(text), "model_name": model_name},
                "result": None,
                "error": None,
                "completed_at": None
            }
        
        # Create summarizer and start background task
        summarizer = LangGraphSummarizer(model_name)
        
        # Submit to thread pool
        future = self.executor.submit(
            self._run_summarization_task,
            thread_id,
            summarizer,
            "summarize_text",
            text,
            thread_id  # Pass thread_id for LangGraph state tracking
        )
        
        return thread_id
    
    async def start_file_summarization(self, file_path: str, model_name: Optional[str] = None) -> str:
        """Start async file summarization and return thread ID."""
        thread_id = self._generate_thread_id()
        
        # Initialize task
        with self._lock:
            self.tasks[thread_id] = {
                "thread_id": thread_id,
                "task_type": "file_summarization",
                "status": TaskStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "input": {"file_path": file_path, "model_name": model_name},
                "result": None,
                "error": None,
                "completed_at": None
            }
        
        # Create summarizer and start background task
        summarizer = LangGraphSummarizer(model_name)
        
        # Submit to thread pool
        future = self.executor.submit(
            self._run_summarization_task,
            thread_id,
            summarizer,
            "summarize_file",
            file_path,
            thread_id  # Pass thread_id for LangGraph state tracking
        )
        
        return thread_id
    
    async def start_batch_summarization(self, file_paths: list, model_name: Optional[str] = None) -> str:
        """Start async batch summarization and return thread ID."""
        thread_id = self._generate_thread_id()
        
        # Initialize task
        with self._lock:
            self.tasks[thread_id] = {
                "thread_id": thread_id,
                "task_type": "batch_summarization",
                "status": TaskStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "input": {"file_count": len(file_paths), "model_name": model_name},
                "result": None,
                "error": None,
                "completed_at": None
            }
        
        # Create summarizer and start background task
        summarizer = LangGraphSummarizer(model_name)
        
        # Submit to thread pool
        future = self.executor.submit(
            self._run_summarization_task,
            thread_id,
            summarizer,
            "summarize_multiple_files",
            file_paths,
            thread_id  # Pass thread_id for LangGraph state tracking
        )
        
        return thread_id
    
    def get_task_status(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by thread ID."""
        with self._lock:
            return self.tasks.get(thread_id)
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all tasks."""
        with self._lock:
            return self.tasks.copy()
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        with self._lock:
            to_remove = []
            for thread_id, task in self.tasks.items():
                if task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    try:
                        completed_at = datetime.fromisoformat(task["completed_at"])
                        if completed_at.timestamp() < cutoff_time:
                            to_remove.append(thread_id)
                    except:
                        # If we can't parse the date, remove it
                        to_remove.append(thread_id)
            
            for thread_id in to_remove:
                del self.tasks[thread_id]
    
    def shutdown(self):
        """Shutdown the task manager."""
        self.executor.shutdown(wait=True)


# Global task manager instance
task_manager = AsyncTaskManager()
