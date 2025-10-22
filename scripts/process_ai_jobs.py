#!/usr/bin/env python3
import os
import time
import subprocess
import fcntl
from pathlib import Path

# Queue file path must match the API enqueuer (storage/service.py)
# Default to /var/log to align with server configuration; allow env override
QUEUE_FILE = os.getenv("AI_QUEUE_FILE", "/var/log/ai_analysis_queue.txt")
LOCK_FILE = "/tmp/ai_worker.lock"
CHECK_SCRIPT = "/var/www/api.arkturian.com/scripts/check_safety_ai.py"

def get_job():
    """Atomically pop the first line from the queue file."""
    if not os.path.exists(QUEUE_FILE):
        return None
    job_line = None
    with open(QUEUE_FILE, "r+") as f:
        lines = f.readlines()
        if lines:
            job_line = lines.pop(0).strip()
            f.seek(0)
            f.writelines(lines)
            f.truncate()
    return job_line or None

def main():
    print("AI Worker started. Watching for jobs...")
    
    # Self-healing: Ensure the check script is executable
    if not os.access(CHECK_SCRIPT, os.X_OK):
        print(f"Worker WARNING: Check script {CHECK_SCRIPT} is not executable. Fixing...")
        os.chmod(CHECK_SCRIPT, 0o755)

    Path(QUEUE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(QUEUE_FILE).touch(exist_ok=True)

    while True:
        job_line = get_job()
        if job_line:
            print(f"Worker found job: {job_line}")
            status_dir = None
            try:
                if job_line.count("|") != 3:
                    raise ValueError("Invalid job format; expected 'id|type|path|filename'")
                
                object_id_str, job_type, content_path_str, verifier_filename = job_line.split("|", 3)
                object_id = int(object_id_str.strip())
                job_type = job_type.strip()
                content_path = Path(content_path_str.strip())
                verifier_filename = verifier_filename.strip()

                # Create status directory to signal "processing"
                status_dir = Path(f"/tmp/ai_analysis_status_{object_id}")
                status_dir.mkdir(exist_ok=True)

                if not content_path.exists():
                    raise FileNotFoundError(f"Content path not found: {content_path}")

                # Execute the check using the correct Python interpreter
                python_exec = "/root/.pyenv/versions/3.11.9/bin/python3.11"
                result = subprocess.run(
                    [python_exec, CHECK_SCRIPT, str(object_id), job_type, str(content_path), verifier_filename],
                    capture_output=True, text=True, timeout=1200  # 20 minutes for large CSV analysis with Gemini Pro
                )

                if result.returncode != 0:
                    raise RuntimeError(f"AI script failed with code {result.returncode}: {result.stderr}")

                # Output captured stdout for debugging (especially AI result keys)
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if 'AI RESULT' in line or 'Has prompt' in line or 'Response length' in line:
                            print(f"    {line}")

                print(f"Worker finished job for object ID: {object_id}")

            except Exception as e:
                error_message = f"Worker ERROR processing job '{job_line}': {e}"
                print(error_message)
                if status_dir:
                    try:
                        with open(status_dir / "error.log", "w") as f:
                            f.write(error_message)
                    except Exception as write_e:
                        print(f"Failed to write error log: {write_e}")
            finally:
                # On success, cleanup status dir. On failure, it's left with the error log.
                if status_dir and not (status_dir / "error.log").exists():
                    try:
                        # shutil.rmtree(status_dir) # Optional: uncomment to clean up on success
                        pass
                    except Exception as clean_e:
                        print(f"Failed to clean up status dir {status_dir}: {clean_e}")
        
        time.sleep(5)

if __name__ == "__main__":
    # Robust single-instance lock using flock
    with open(LOCK_FILE, "w") as lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another AI worker instance is already running. Exiting.")
            raise SystemExit(0)

        try:
            lock_fd.write(str(os.getpid()))
            lock_fd.flush()
            main()
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except Exception:
                pass