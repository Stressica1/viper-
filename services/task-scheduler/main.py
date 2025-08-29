#!/usr/bin/env python3
"""
🚀 VIPER Task Scheduler Service
Advanced job and task management for the VIPER trading system
Features:
- Distributed task scheduling
- Job queue management
- Task monitoring and reporting
- Redis-based task distribution
"""

import os
import asyncio
import json
import logging
import redis
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'task-scheduler')
PORT = int(os.getenv('TASK_SCHEDULER_PORT', '8021'))

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    SCAN_MARKET = "scan_market"
    SCORE_OPPORTUNITY = "score_opportunity"
    EXECUTE_TRADE = "execute_trade"
    MONITOR_POSITION = "monitor_position"
    CLOSE_POSITION = "close_position"
    UPDATE_BALANCE = "update_balance"
    SEND_ALERT = "send_alert"
    BACKUP_DATA = "backup_data"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

@dataclass
class Task:
    task_id: str
    task_type: TaskType
    priority: int  # 1 (highest) to 10 (lowest)
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    assigned_worker: str = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: str = None
    result: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class TaskRequest(BaseModel):
    task_type: str
    priority: int = 5
    payload: Dict[str, Any] = {}
    max_retries: int = 3

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskScheduler:
    """Advanced task scheduler with Redis backend"""
    
    def __init__(self):
        self.redis_client = None
        self.app = FastAPI(title="VIPER Task Scheduler", version="1.0.0")
        self.is_running = False
        self.workers: Dict[str, Dict] = {}
        self.task_queues: Dict[TaskType, str] = {}
        
        # Performance metrics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.active_tasks = 0
        
        # Setup routes
        self.setup_routes()
        
        logger.info("🚀 VIPER Task Scheduler initialized")

    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/tasks", response_model=TaskResponse)
        async def create_task(request: TaskRequest):
            """Create a new task"""
            try:
                task = await self.schedule_task(
                    task_type=TaskType(request.task_type),
                    priority=request.priority,
                    payload=request.payload,
                    max_retries=request.max_retries
                )
                
                return TaskResponse(
                    task_id=task.task_id,
                    status=task.status.value,
                    message="Task scheduled successfully"
                )
                
            except Exception as e:
                logger.error(f"❌ Error creating task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            """Get task status"""
            try:
                task = await self.get_task_status(task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                return {
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "status": task.status.value,
                    "priority": task.priority,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "retry_count": task.retry_count,
                    "error_message": task.error_message,
                    "result": task.result
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Error getting task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/tasks/{task_id}")
        async def cancel_task(task_id: str):
            """Cancel a task"""
            try:
                success = await self.cancel_task(task_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
                
                return {"message": "Task cancelled successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Error cancelling task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status")
        async def get_system_status():
            """Get scheduler status"""
            return {
                "service": SERVICE_NAME,
                "status": "running" if self.is_running else "stopped",
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "active_tasks": self.active_tasks,
                "active_workers": len(self.workers),
                "queues": {task_type.value: await self.get_queue_size(task_type) 
                          for task_type in TaskType}
            }
        
        @self.app.get("/workers")
        async def get_workers():
            """Get worker status"""
            return {
                "workers": self.workers,
                "total_workers": len(self.workers)
            }
        
        @self.app.post("/workers/register")
        async def register_worker(worker_data: Dict[str, Any]):
            """Register a new worker"""
            worker_id = worker_data.get('worker_id', str(uuid.uuid4()))
            self.workers[worker_id] = {
                **worker_data,
                'registered_at': datetime.now().isoformat(),
                'last_heartbeat': datetime.now().isoformat(),
                'status': 'active'
            }
            
            logger.info(f"👷 Worker {worker_id} registered")
            return {"worker_id": worker_id, "message": "Worker registered successfully"}

    async def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            logger.info("✅ Connected to Redis")
            
            # Initialize task queues
            for task_type in TaskType:
                queue_name = f"viper:tasks:{task_type.value}"
                self.task_queues[task_type] = queue_name
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    async def schedule_task(self, task_type: TaskType, priority: int = 5, 
                          payload: Dict[str, Any] = None, max_retries: int = 3) -> Task:
        """Schedule a new task"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            payload=payload or {},
            max_retries=max_retries
        )
        
        # Store task in Redis
        task_key = f"viper:task:{task_id}"
        await asyncio.get_event_loop().run_in_executor(
            None, 
            self.redis_client.setex,
            task_key,
            3600,  # 1 hour TTL
            json.dumps(asdict(task), default=str)
        )
        
        # Add to appropriate queue
        queue_name = self.task_queues[task_type]
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis_client.lpush,
            queue_name,
            task_id
        )
        
        self.total_tasks += 1
        self.active_tasks += 1
        
        logger.info(f"📝 Scheduled task {task_id} ({task_type.value}) with priority {priority}")
        return task

    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task status from Redis"""
        try:
            task_key = f"viper:task:{task_id}"
            task_data = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, task_key
            )
            
            if not task_data:
                return None
            
            task_dict = json.loads(task_data)
            
            # Convert string dates back to datetime
            for date_field in ['created_at', 'started_at', 'completed_at']:
                if task_dict.get(date_field):
                    task_dict[date_field] = datetime.fromisoformat(task_dict[date_field])
            
            # Convert enums
            task_dict['task_type'] = TaskType(task_dict['task_type'])
            task_dict['status'] = TaskStatus(task_dict['status'])
            
            return Task(**task_dict)
            
        except Exception as e:
            logger.error(f"❌ Error getting task status: {e}")
            return None

    async def update_task_status(self, task_id: str, status: TaskStatus, 
                               **updates) -> bool:
        """Update task status"""
        try:
            task = await self.get_task_status(task_id)
            if not task:
                return False
            
            # Update task
            task.status = status
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            # Update timestamps
            if status == TaskStatus.RUNNING and not task.started_at:
                task.started_at = datetime.now()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                task.completed_at = datetime.now()
                self.active_tasks -= 1
                
                if status == TaskStatus.COMPLETED:
                    self.completed_tasks += 1
                elif status == TaskStatus.FAILED:
                    self.failed_tasks += 1
            
            # Save to Redis
            task_key = f"viper:task:{task_id}"
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.setex,
                task_key,
                3600,
                json.dumps(asdict(task), default=str)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating task status: {e}")
            return False

    async def get_next_task(self, task_types: List[TaskType]) -> Optional[Task]:
        """Get next task from priority queues"""
        try:
            # Check queues in order of priority
            for task_type in task_types:
                queue_name = self.task_queues[task_type]
                task_id = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.rpop, queue_name
                )
                
                if task_id:
                    task = await self.get_task_status(task_id)
                    if task and task.status == TaskStatus.PENDING:
                        await self.update_task_status(task_id, TaskStatus.RUNNING)
                        return task
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting next task: {e}")
            return None

    async def get_queue_size(self, task_type: TaskType) -> int:
        """Get queue size"""
        try:
            queue_name = self.task_queues[task_type]
            size = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.llen, queue_name
            )
            return size or 0
        except Exception:
            return 0

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            task = await self.get_task_status(task_id)
            if not task or task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return False
            
            await self.update_task_status(task_id, TaskStatus.CANCELLED)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cancelling task: {e}")
            return False

    async def cleanup_completed_tasks(self):
        """Cleanup old completed tasks"""
        try:
            # This would implement cleanup logic for old tasks
            # For now, Redis TTL handles most cleanup
            pass
        except Exception as e:
            logger.error(f"❌ Error in cleanup: {e}")

    async def health_check(self):
        """Health check endpoint logic"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "redis_connected": self.redis_client is not None,
            "active_tasks": self.active_tasks,
            "total_workers": len(self.workers)
        }

    async def start_scheduler(self):
        """Start the task scheduler"""
        logger.info("🚀 Starting VIPER Task Scheduler")
        self.is_running = True
        
        # Background task for cleanup
        asyncio.create_task(self.periodic_cleanup())
        
        logger.info("✅ Task Scheduler started successfully")

    async def periodic_cleanup(self):
        """Periodic cleanup of old tasks and workers"""
        while self.is_running:
            try:
                await self.cleanup_completed_tasks()
                
                # Clean up inactive workers
                current_time = datetime.now()
                inactive_workers = []
                
                for worker_id, worker_data in self.workers.items():
                    last_heartbeat = datetime.fromisoformat(worker_data['last_heartbeat'])
                    if current_time - last_heartbeat > timedelta(minutes=5):
                        inactive_workers.append(worker_id)
                
                for worker_id in inactive_workers:
                    del self.workers[worker_id]
                    logger.warning(f"🔄 Removed inactive worker {worker_id}")
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in periodic cleanup: {e}")
                await asyncio.sleep(60)

    async def stop_scheduler(self):
        """Stop the task scheduler"""
        logger.info("🔄 Stopping Task Scheduler...")
        self.is_running = False

async def create_app():
    """Create and configure the FastAPI app"""
    scheduler = TaskScheduler()
    
    # Connect to Redis
    if not await scheduler.connect_redis():
        logger.error("❌ Failed to connect to Redis")
        return None
    
    # Start scheduler
    await scheduler.start_scheduler()
    
    return scheduler.app

async def main():
    """Main function"""
    try:
        app = await create_app()
        if not app:
            return
        
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=PORT,
            log_level=LOG_LEVEL.lower()
        )
        
        server = uvicorn.Server(config)
        logger.info(f"🚀 Starting Task Scheduler on port {PORT}")
        await server.serve()
        
    except Exception as e:
        logger.error(f"❌ Failed to start Task Scheduler: {e}")

if __name__ == "__main__":
    asyncio.run(main())
