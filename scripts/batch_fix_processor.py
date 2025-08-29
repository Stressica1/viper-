#!/usr/bin/env python3
"""
📦 BATCH FIX PROCESSOR - EFFICIENT MCP FIXING SYSTEM
==================================================

Batch processing system for MCP-powered error fixes.

Features:
- Parallel batch processing for efficiency
- Intelligent queue management
- Progress tracking and reporting
- Error handling and recovery
- Resource management and throttling

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import json
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import logging

@dataclass
class BatchJob:
    """Represents a batch processing job"""
    job_id: str
    batch: Any  # FixBatch object
    priority: int  # 1=Critical, 2=High, 3=Medium, 4=Low
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "queued"  # queued, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error_message: str = ""

    def __lt__(self, other):
        """Make BatchJob comparable for priority queue"""
        if not isinstance(other, BatchJob):
            return NotImplemented
        return self.priority < other.priority

    def __le__(self, other):
        """Make BatchJob comparable for priority queue"""
        if not isinstance(other, BatchJob):
            return NotImplemented
        return self.priority <= other.priority

    def __gt__(self, other):
        """Make BatchJob comparable for priority queue"""
        if not isinstance(other, BatchJob):
            return NotImplemented
        return self.priority > other.priority

    def __ge__(self, other):
        """Make BatchJob comparable for priority queue"""
        if not isinstance(other, BatchJob):
            return NotImplemented
        return self.priority >= other.priority

@dataclass
class BatchProcessorConfig:
    """Configuration for batch processor"""
    max_workers: int = 4
    queue_size: int = 1000
    batch_timeout: int = 300  # 5 minutes per batch
    retry_attempts: int = 3
    retry_delay: float = 1.0
    progress_interval: float = 5.0
    enable_parallel: bool = True

class BatchFixProcessor:
    """Batch processor for MCP fixes"""

    def __init__(self, config: BatchProcessorConfig = None, fixer_instance = None):
        self.config = config or BatchProcessorConfig()
        self.fixer = fixer_instance
        self.job_queue = queue.PriorityQueue(maxsize=self.config.queue_size)
        self.completed_jobs = []
        self.failed_jobs = []
        self.active_jobs = 0
        self.lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger('BatchProcessor')
        self.logger.setLevel(logging.INFO)

        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

        # Setup file handler
        handler = logging.FileHandler(self.logs_dir / f"batch_processor_{int(time.time())}.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def submit_job(self, batch: Any, priority: int = 3) -> str:
        """Submit a batch job for processing"""
        job_id = f"job_{int(time.time() * 1000)}_{len(self.completed_jobs) + len(self.failed_jobs)}"

        job = BatchJob(
            job_id=job_id,
            batch=batch,
            priority=priority,
            created_at=datetime.now().isoformat()
        )

        try:
            self.job_queue.put((priority, job), timeout=10)
            self.logger.info(f"Submitted job {job_id} with priority {priority}")
            return job_id
        except queue.Full:
            raise Exception("Job queue is full")

    def submit_critical_issues(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Submit critical issues for immediate processing"""
        job_ids = []

        # Group critical issues by file
        critical_by_file = defaultdict(list)
        for issue in issues:
            if issue['severity'] in ['CRITICAL', 'HIGH']:
                critical_by_file[issue['file_path']].append(issue)

        # Submit high-priority jobs for critical issues
        for file_path, file_issues in critical_by_file.items():
            # Create a mock batch object (in practice, this would be a proper FixBatch)
            mock_batch = type('MockBatch', (), {
                'batch_id': f"critical_{Path(file_path).name}_{int(time.time())}",
                'file_path': file_path,
                'issues': file_issues,
                'fixes': [],
                'status': 'pending',
                'created_at': datetime.now().isoformat()
            })()

            job_id = self.submit_job(mock_batch, priority=1)  # Highest priority
            job_ids.append(job_id)
            self.logger.info(f"Submitted critical job for {file_path}: {job_id}")

        return job_ids

    def process_queue(self) -> Dict[str, Any]:
        """Process all jobs in the queue"""
        if not self.fixer:
            raise Exception("No fixer instance provided")

        self.logger.info("Starting batch processing...")
        start_time = time.time()

        if self.config.enable_parallel:
            return self._process_parallel()
        else:
            return self._process_sequential()

    def _process_parallel(self) -> Dict[str, Any]:
        """Process jobs in parallel using thread pool"""
        results = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'processing_time': 0,
            'jobs': []
        }

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            # Submit initial batch of jobs
            for _ in range(min(self.config.max_workers * 2, self.job_queue.qsize())):
                if not self.job_queue.empty():
                    try:
                        priority, job = self.job_queue.get_nowait()
                        future = executor.submit(self._process_single_job, job)
                        futures[future] = job
                        results['total_jobs'] += 1
                        self.logger.info(f"Submitted job {job.job_id} to thread pool")
                    except queue.Empty:
                        break

            # Process completed futures
            for future in as_completed(futures):
                job = futures[future]
                try:
                    job_result = future.result(timeout=self.config.batch_timeout)
                    results['jobs'].append(job_result)

                    if job_result['status'] == 'completed':
                        results['completed_jobs'] += 1
                    else:
                        results['failed_jobs'] += 1

                except Exception as e:
                    self.logger.error(f"Job {job.job_id} failed with error: {e}")
                    results['failed_jobs'] += 1
                    results['jobs'].append({
                        'job_id': job.job_id,
                        'status': 'failed',
                        'error': str(e)
                    })

                # Submit next job if available
                if not self.job_queue.empty():
                    try:
                        priority, next_job = self.job_queue.get_nowait()
                        next_future = executor.submit(self._process_single_job, next_job)
                        futures[next_future] = next_job
                        results['total_jobs'] += 1
                    except queue.Empty:
                        pass

        results['processing_time'] = time.time() - start_time
        return results

    def _process_sequential(self) -> Dict[str, Any]:
        """Process jobs sequentially"""
        results = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'processing_time': 0,
            'jobs': []
        }

        start_time = time.time()

        while not self.job_queue.empty():
            try:
                priority, job = self.job_queue.get_nowait()
                results['total_jobs'] += 1

                job_result = self._process_single_job(job)
                results['jobs'].append(job_result)

                if job_result['status'] == 'completed':
                    results['completed_jobs'] += 1
                else:
                    results['failed_jobs'] += 1

            except queue.Empty:
                break

        results['processing_time'] = time.time() - start_time
        return results

    def _process_single_job(self, job: BatchJob) -> Dict[str, Any]:
        """Process a single job"""
        job.started_at = datetime.now().isoformat()
        job.status = "processing"

        self.logger.info(f"Processing job {job.job_id} for {job.batch.file_path}")

        try:
            # Process the batch using the fixer
            if hasattr(self.fixer, 'process_fix_batch'):
                result_batch = self.fixer.process_fix_batch(job.batch)

                job.completed_at = datetime.now().isoformat()
                job.status = result_batch.status
                job.result = {
                    'batch_id': result_batch.batch_id,
                    'file_path': result_batch.file_path,
                    'status': result_batch.status,
                    'fixes_attempted': len(result_batch.fixes),
                    'fixes_successful': sum(1 for f in result_batch.fixes if f.success)
                }

                self.logger.info(f"Job {job.job_id} completed: {job.result}")
                return asdict(job)

            else:
                raise Exception("Fixer instance does not have process_fix_batch method")

        except Exception as e:
            job.completed_at = datetime.now().isoformat()
            job.status = "failed"
            job.error_message = str(e)

            self.logger.error(f"Job {job.job_id} failed: {e}")
            return asdict(job)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            'queue_size': self.job_queue.qsize(),
            'active_jobs': self.active_jobs,
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'is_empty': self.job_queue.empty()
        }

    def wait_for_completion(self, timeout: float = None) -> bool:
        """Wait for all jobs to complete"""
        start_time = time.time()

        while not self.job_queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)

        return True

    def generate_progress_report(self, results: Dict[str, Any]) -> str:
        """Generate progress report"""
        report = f"""
🔄 BATCH PROCESSING REPORT
{'='*40}

📊 OVERVIEW
  Total Jobs: {results['total_jobs']}
  Completed: {results['completed_jobs']}
  Failed: {results['failed_jobs']}
  Success Rate: {(results['completed_jobs'] / max(results['total_jobs'], 1)) * 100:.1f}%
  Processing Time: {results['processing_time']:.2f}s

📋 JOB DETAILS
"""

        for job_result in results['jobs']:
            status_icon = "✅" if job_result.get('status') == 'completed' else "❌"
            report += f"  {status_icon} {job_result.get('job_id', 'Unknown')}\n"

            if 'result' in job_result and job_result['result']:
                result = job_result['result']
                report += f"    File: {result.get('file_path', 'Unknown')}\n"
                report += f"    Fixes: {result.get('fixes_successful', 0)}/{result.get('fixes_attempted', 0)}\n"

            if job_result.get('error'):
                report += f"    Error: {job_result['error']}\n"

        return report

class BatchFixOrchestrator:
    """Orchestrator for batch fix processing"""

    def __init__(self, fixer_config=None, processor_config=None):
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        from mcp_error_fixer import MCPErrorFixer, MCPFixerConfig

        self.fixer_config = fixer_config or MCPFixerConfig()
        self.processor_config = processor_config or BatchProcessorConfig()

        self.fixer = MCPErrorFixer(self.fixer_config)
        self.processor = BatchFixProcessor(self.processor_config, self.fixer)

    def orchestrate_fixes(self, scan_file: str = None) -> Dict[str, Any]:
        """Orchestrate the complete fix process"""
        print("🎯 BATCH FIX ORCHESTRATOR")
        print("=" * 50)

        # Load and categorize issues
        issues = self.fixer.load_scan_results(scan_file)
        if not issues:
            return {'error': 'No scan results found'}

        print(f"📊 Loaded {len(issues)} issues")

        # Categorize issues by priority
        critical_issues = [i for i in issues if i['severity'] == 'CRITICAL']
        high_issues = [i for i in issues if i['severity'] == 'HIGH']
        medium_issues = [i for i in issues if i['severity'] == 'MEDIUM']
        low_issues = [i for i in issues if i['severity'] == 'LOW']

        print(f"🚨 Critical: {len(critical_issues)}")
        print(f"🔴 High: {len(high_issues)}")
        print(f"🟡 Medium: {len(medium_issues)}")
        print(f"🟢 Low: {len(low_issues)}")

        # Submit jobs by priority
        print("\n📦 Submitting jobs...")

        # Critical issues first
        if critical_issues:
            critical_job_ids = self.processor.submit_critical_issues(critical_issues)
            print(f"✅ Submitted {len(critical_job_ids)} critical jobs")

        # High priority issues
        if high_issues:
            high_batches = self.fixer.create_fix_batches(high_issues)
            for batch in high_batches:
                self.processor.submit_job(batch, priority=2)
            print(f"✅ Submitted {len(high_batches)} high-priority batches")

        # Medium and low priority
        remaining_issues = medium_issues + low_issues
        if remaining_issues:
            remaining_batches = self.fixer.create_fix_batches(remaining_issues)
            for batch in remaining_batches:
                priority = 3 if any(i['severity'] == 'MEDIUM' for i in batch.issues) else 4
                self.processor.submit_job(batch, priority=priority)
            print(f"✅ Submitted {len(remaining_batches)} standard batches")

        # Process queue
        print("\n🔄 Processing queue...")
        results = self.processor.process_queue()

        # Generate final report
        progress_report = self.processor.generate_progress_report(results)

        print("\n" + progress_report)

        # Save comprehensive report
        final_report = {
            'orchestrator_results': results,
            'issue_breakdown': {
                'critical': len(critical_issues),
                'high': len(high_issues),
                'medium': len(medium_issues),
                'low': len(low_issues)
            },
            'processing_summary': progress_report,
            'generated_at': datetime.now().isoformat()
        }

        report_path = Path("reports") / f"batch_orchestrator_report_{int(time.time())}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        print(f"📄 Comprehensive report saved: {report_path}")

        return {
            'results': results,
            'report_path': str(report_path),
            'success': results['failed_jobs'] == 0
        }

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Batch Fix Processor - Efficient MCP Fixing System')
    parser.add_argument('--scan-file', '-s', help='Path to bug scan results JSON file')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--sequential', action='store_true', help='Process jobs sequentially instead of parallel')
    parser.add_argument('--mcp-url', help='MCP server URL')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')

    args = parser.parse_args()

    # Configure processor
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from mcp_error_fixer import MCPFixerConfig

    fixer_config = MCPFixerConfig(
        mcp_server_url=args.mcp_url or "http://localhost:3000",
        dry_run=args.dry_run
    )

    processor_config = BatchProcessorConfig(
        max_workers=args.workers if not args.sequential else 1,
        enable_parallel=not args.sequential
    )

    # Create orchestrator
    orchestrator = BatchFixOrchestrator(fixer_config, processor_config)

    # Run orchestration
    result = orchestrator.orchestrate_fixes(args.scan_file)

    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    elif result.get('success', False):
        print("🎉 All fixes completed successfully!")
        sys.exit(0)
    else:
        failed_jobs = result['results']['failed_jobs']
        print(f"⚠️  {failed_jobs} jobs failed - manual review required")
        sys.exit(1)

if __name__ == '__main__':
    main()
