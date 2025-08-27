# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "pandas"
# ]
# ///

import asyncio
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import signal
import sys
from typing import Dict, Set, Tuple, Optional

# Constants
PROMPTFOO_PROJECTS = [
    "promptfoos_private/climate-research",
    "promptfoos_private/ecommerce-analytics"
]

EVALUATION_LOGS_DIR = Path("evaluation_logs")
PROGRESS_FILE = Path("evaluation_progress.json")
IST = datetime.now().astimezone().tzinfo  # Local timezone for timestamps

class ProgressTracker:
    """Track and persist evaluation progress."""
    
    def __init__(self):
        self.progress_data = self.load_progress()
        self.setup_signal_handlers()
    
    def load_progress(self) -> Dict:
        """Load existing progress from file."""
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    data = json.load(f)
                print(f"[RESUME] Loaded progress from {PROGRESS_FILE}")
                return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"[WARNING] Could not load progress file: {e}")
        
        return {
            "completed_evaluations": {},  # {email: {attempt: [completed_projects]}}
            "current_batch_start_time": None,
            "total_emails": 0,
            "last_updated": None
        }
    
    def save_progress(self):
        """Save current progress to file."""
        self.progress_data["last_updated"] = datetime.now(IST).isoformat()
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except IOError as e:
            print(f"[ERROR] Could not save progress: {e}")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            print(f"\n[SHUTDOWN] Received signal {signum}. Saving progress...")
            self.save_progress()
            print("[SHUTDOWN] Progress saved. Exiting gracefully.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def is_project_completed(self, email: str, attempt: int, project_name: str) -> bool:
        """Check if a specific project evaluation is completed."""
        email_progress = self.progress_data["completed_evaluations"].get(email, {})
        attempt_progress = email_progress.get(str(attempt), [])
        return project_name in attempt_progress
    
    def mark_project_completed(self, email: str, attempt: int, project_name: str):
        """Mark a project as completed."""
        if email not in self.progress_data["completed_evaluations"]:
            self.progress_data["completed_evaluations"][email] = {}
        
        attempt_str = str(attempt)
        if attempt_str not in self.progress_data["completed_evaluations"][email]:
            self.progress_data["completed_evaluations"][email][attempt_str] = []
        
        if project_name not in self.progress_data["completed_evaluations"][email][attempt_str]:
            self.progress_data["completed_evaluations"][email][attempt_str].append(project_name)
        
        self.save_progress()
    
    def get_resume_info(self, df: pd.DataFrame) -> Tuple[int, Dict[str, int]]:
        """Determine where to resume from."""
        completed_emails = set()
        email_attempts = {}
        
        for email in df["Email"].unique():
            email_progress = self.progress_data["completed_evaluations"].get(email, {})
            
            if email_progress:
                # Get the latest attempt for this email
                latest_attempt = max(int(attempt) for attempt in email_progress.keys())
                completed_projects = email_progress[str(latest_attempt)]
                
                if len(completed_projects) == len(PROMPTFOO_PROJECTS):
                    # All projects completed for this email
                    completed_emails.add(email)
                    print(f"[RESUME] Email {email} fully completed (attempt {latest_attempt})")
                else:
                    # Partial completion - will resume this attempt
                    email_attempts[email] = latest_attempt
                    print(f"[RESUME] Email {email} partially completed (attempt {latest_attempt}, {len(completed_projects)}/{len(PROMPTFOO_PROJECTS)} projects)")
            else:
                # No progress for this email - will start fresh
                pass
        
        total_completed = len(completed_emails)
        return total_completed, email_attempts

def get_next_run_number(email: str) -> int:
    """Return the next attempt number for a given email."""
    base_dir = EVALUATION_LOGS_DIR / email
    base_dir.mkdir(parents=True, exist_ok=True)

    runs = [int(d.name) for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return max(runs) + 1 if runs else 1

def is_evaluation_completed(email: str, attempt: int, project_name: str) -> bool:
    """Check if evaluation was completed successfully by examining the output."""
    attempt_log_dir = EVALUATION_LOGS_DIR / email / str(attempt) / project_name
    output_file = attempt_log_dir / "output.json"
    metadata_file = attempt_log_dir / "metadata.json"
    
    if not (output_file.exists() and metadata_file.exists()):
        return False
    
    try:
        # Check if metadata indicates success
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            if metadata.get("exit_code") != 0:
                return False
        
        # Check if output file has valid content
        with open(output_file, 'r') as f:
            output_data = json.load(f)
            # Basic validation - output should have results
            return isinstance(output_data, dict) and len(output_data) > 0
    
    except (json.JSONDecodeError, IOError):
        return False

async def run_promptfoo(email: str, attempt_number: int, project_path: Path, provider_url: str, progress_tracker: ProgressTracker):
    """Run promptfoo for a specific project and store logs with metadata."""
    project_name = project_path.name
    
    # Check if already completed
    if progress_tracker.is_project_completed(email, attempt_number, project_name):
        print(f"    [SKIP] {project_name} already completed for {email} (attempt {attempt_number})")
        return True
    
    # Double-check by examining actual files
    if is_evaluation_completed(email, attempt_number, project_name):
        print(f"    [SKIP] {project_name} found completed on disk for {email} (attempt {attempt_number})")
        progress_tracker.mark_project_completed(email, attempt_number, project_name)
        return True
    
    attempt_log_dir = EVALUATION_LOGS_DIR / email / str(attempt_number) / project_name
    attempt_log_dir.mkdir(parents=True, exist_ok=True)

    # File paths for logs
    output_file = attempt_log_dir / "output.json"
    stdout_file = attempt_log_dir / "stdout.txt"
    stderr_file = attempt_log_dir / "stderr.txt"
    metadata_file = attempt_log_dir / "metadata.json"

    # Environment setup
    env = os.environ.copy()
    env["PROMPTFOO_DISABLE_OBJECT_STRINGIFY"] = "true"
    env["PROVIDER_URL"] = provider_url

    # Explicit command using correct promptfoo.yaml
    cmd = f"""
        npx -y promptfoo eval \
        --config {project_path / 'promptfoo.yaml'} \
        --output {output_file} \
        --no-cache
    """

    print(f"    [START] Running {project_name} for {email} (attempt {attempt_number}) with PROVIDER_URL={provider_url}")
    
    try:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        stdout, stderr = await process.communicate()

        # Save logs
        stdout_file.write_text(stdout.decode())
        stderr_file.write_text(stderr.decode())

        # Save metadata
        metadata = {
            "email": email,
            "url": provider_url,
            "project": project_name,
            "attempt": attempt_number,
            "exit_code": process.returncode,
            "timestamp": datetime.now(IST).isoformat()
        }
        metadata_file.write_text(json.dumps(metadata, indent=2))

        success = process.returncode == 0
        print(f"    [{'DONE' if success else 'FAILED'}] {project_name} completed with code {process.returncode}")
        
        if success:
            progress_tracker.mark_project_completed(email, attempt_number, project_name)
        
        return success
        
    except Exception as e:
        print(f"    [ERROR] Failed to run {project_name} for {email}: {e}")
        # Save error metadata
        metadata = {
            "email": email,
            "url": provider_url,
            "project": project_name,
            "attempt": attempt_number,
            "exit_code": -1,
            "error": str(e),
            "timestamp": datetime.now(IST).isoformat()
        }
        metadata_file.write_text(json.dumps(metadata, indent=2))
        return False

async def evaluate_email(email: str, provider_url: str, progress_tracker: ProgressTracker, resume_attempt: Optional[int] = None):
    """Evaluate all projects sequentially for a given email."""
    attempt_number = resume_attempt if resume_attempt else get_next_run_number(email)
    print(f"\n[EMAIL] {'Resuming' if resume_attempt else 'Starting'} evaluation for {email}, attempt {attempt_number}")

    success_count = 0
    for i, project in enumerate(PROMPTFOO_PROJECTS):
        project_name = Path(project).name
        
        # Check if this project was already completed
        if progress_tracker.is_project_completed(email, attempt_number, project_name):
            print(f"    [SKIP] {project_name} already completed")
            success_count += 1
            continue
        
        success = await run_promptfoo(email, attempt_number, Path(project), provider_url, progress_tracker)
        if success:
            success_count += 1
        
        # Wait between projects (except for the last one)
        if i < len(PROMPTFOO_PROJECTS) - 1:
            print("    Waiting 1 minute before next project...")
            await asyncio.sleep(60)
    
    completion_status = "COMPLETED" if success_count == len(PROMPTFOO_PROJECTS) else "PARTIAL"
    print(f"[EMAIL] {completion_status} evaluation for {email} ({success_count}/{len(PROMPTFOO_PROJECTS)} projects)")

async def main():
    print("[STARTUP] Loading evaluation data...")
    df = pd.read_csv("latest.csv")
    
    progress_tracker = ProgressTracker()
    progress_tracker.progress_data["total_emails"] = len(df)
    
    # Determine resume point
    completed_count, email_attempts = progress_tracker.get_resume_info(df)
    
    if completed_count > 0:
        print(f"[RESUME] Found {completed_count} completed emails, resuming remaining work...")
    
    tasks = []
    emails_to_process = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        email = row["Email"]
        provider_url = row["URL"]
        
        # Skip completed emails
        if email in progress_tracker.progress_data["completed_evaluations"]:
            latest_attempt = max(int(att) for att in progress_tracker.progress_data["completed_evaluations"][email].keys())
            completed_projects = progress_tracker.progress_data["completed_evaluations"][email][str(latest_attempt)]
            if len(completed_projects) == len(PROMPTFOO_PROJECTS):
                print(f"[SKIP] Email {email} already fully completed")
                continue
        
        emails_to_process.append((email, provider_url, email_attempts.get(email)))
    
    if not emails_to_process:
        print("[COMPLETE] All emails have been processed!")
        return
    
    print(f"[PROCESSING] Will process {len(emails_to_process)} emails")
    
    for i, (email, provider_url, resume_attempt) in enumerate(emails_to_process):
        task = asyncio.create_task(
            evaluate_email(email, provider_url, progress_tracker, resume_attempt)
        )
        tasks.append(task)
        print(f"[QUEUE] Scheduled evaluation for {email} with URL={provider_url}")
        
        # Wait before launching next task (except for the last one)
        if i < len(emails_to_process) - 1:
            await asyncio.sleep(60)  # Launch next email evaluation after 1 min

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        print("[COMPLETE] All evaluations finished!")
        
        # Clean up progress file on successful completion
        if PROGRESS_FILE.exists():
            backup_file = PROGRESS_FILE.with_suffix('.json.completed')
            PROGRESS_FILE.rename(backup_file)
            print(f"[CLEANUP] Progress file backed up to {backup_file}")
            
    except Exception as e:
        print(f"[ERROR] Error during evaluation: {e}")
        progress_tracker.save_progress()
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Script interrupted by user")
    except Exception as e:
        print(f"[FATAL] Fatal error: {e}")
        sys.exit(1)
