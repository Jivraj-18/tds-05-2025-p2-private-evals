# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi[standard]",
#   "asyncpg",
#   "uvicorn",
#   "httpx",
#   "python-jose",
#   "python-dotenv",
#   "pyjwt",
# ]
# ///
from fastapi import FastAPI, HTTPException, Request, Query, Depends, Cookie, Form, BackgroundTasks, Header
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urlencode
import os
import csv
import asyncio
import logging
import json
import zipfile
import tempfile
import atexit
import jwt
import re
from typing import Optional, Dict, Any, Tuple

# Set up logging
logging.basicConfig(
    filename="app.log",       # Log file name
    level=logging.INFO,       # Minimum log level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
)
logger = logging.getLogger(__name__)

app = FastAPI()
IST = timezone(timedelta(hours=5, minutes=30))
PROMPTFOOS_DIR = Path("promptfoos")  # Adjust if needed
EVALUATION_LOGS_DIR = Path("evaluation_logs")
evaluation_status = {}  # Make sure this is accessible

# Track evaluation status
evaluation_status = {}

# Cleanup temporary files on exit
temp_files = []

def cleanup_temp_files():
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception:
            pass

atexit.register(cleanup_temp_files)

# Enable CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    logger.warning("JWT_SECRET not set in environment. Using an insecure default for development only.")
    JWT_SECRET = "insecure-jwt-secret-for-development-only"  # Only for development!

# User submission limits
MAX_SUBMISSIONS_PER_USER = 100

# JWT token validation
def decode_jwt_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token"""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(token: Optional[str] = Cookie(None, alias="auth_token"), 
                     authorization: Optional[str] = Header(None)):
    """Get current authenticated user from JWT token in cookie or Authorization header"""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    elif not token:
        # Also check if token is in query params (for the initial redirect from Vercel)
        return None
    
    try:
        return decode_jwt_token(token)
    except HTTPException:
        return None

def require_auth(current_user = Depends(get_current_user)):
    """Dependency to require authentication"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user

def count_user_submissions(email: str) -> int:
    """Count how many submissions a user has made"""
    user_dir = Path("evaluations") / email.replace("@", "_at_").replace(".", "_")
    if not user_dir.exists():
        return 0
    
    # Count subdirectories (attempts)
    return len([d for d in user_dir.iterdir() if d.is_dir()])

def get_next_attempt_number(email: str) -> int:
    """Get the next attempt number for a user"""
    return count_user_submissions(email) + 1

async def validate_github_repo(repo_url: str) -> Tuple[bool, str]:
    """
    Validate a GitHub repository URL and check if it has an MIT license.
    Returns a tuple of (is_valid, message)
    """
    # Validate GitHub URL format
    github_pattern = r"https?://github\.com/([^/]+)/([^/]+)(/|$)"
    match = re.match(github_pattern, repo_url)
    if not match:
        return False, "Invalid GitHub repository URL format"
    
    owner = match.group(1)
    repo = match.group(2)
    
    # Check if repository exists and has license
    try:
        async with httpx.AsyncClient() as client:
            # Check repository
            repo_response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"}
            )
            

            repo_data = repo_response.json()
            if repo_response.status_code != 200:
                return False, f"GitHub repository not found or inaccessible: {repo_response.status_code}"
            if repo_data.get("license", {}).get("spdx_id", "") != "MIT":
                return False, "Repository does not have an MIT license."

            
            return True, "Repository has a valid MIT license."
    
    except Exception as e:
        return False, f"Error validating GitHub repository: {str(e)}"

async def run_evaluation_background(email: str, url: str, github_repo: str, attempt_number: int, attempt_dir: Path):
    """Run evaluation across all promptfoo setups in subdirectories."""

    evaluation_key = f"{email}_{attempt_number}"
    evaluation_status[evaluation_key] = {"status": "running", "timestamp": datetime.now(IST)}

    try:
        env = os.environ.copy()
        env["LOG_LEVEL"] = "debug"
        env["PROMPTFOO_DISABLE_OBJECT_STRINGIFY"] = "true"
        env["PROVIDER_URL"] = url  
        if "AIPIPE_TOKEN" in os.environ:
            env["OPENAI_API_KEY"] = os.environ["AIPIPE_TOKEN"]
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables")

        # Directory for this student's specific attempt
        attempt_log_dir = EVALUATION_LOGS_DIR / email / str(attempt_number)
        attempt_log_dir.mkdir(parents=True, exist_ok=True)

        # Loop through each promptfoo configuration
        promptfoo_dirs = [d for d in PROMPTFOOS_DIR.iterdir() if d.is_dir()]

        for promptfoo_dir in promptfoo_dirs:
            prompt_name = promptfoo_dir.name
            prompt_output_dir = attempt_log_dir / prompt_name
            prompt_output_dir.mkdir(parents=True, exist_ok=True)

            output_file = prompt_output_dir / "output.json"
            stdout_file = prompt_output_dir / "stdout.txt"
            stderr_file = prompt_output_dir / "stderr.txt"
            metadata_file = prompt_output_dir / "metadata.json"

            cmd = f"""
                npx -y promptfoo eval \
                --config {promptfoo_dir / 'promptfoo.yaml'} \
                --output {output_file} \
                --no-cache
            """

            logger.debug(f"Running command: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await process.communicate()

            # Save logs
            stdout_file.write_text(stdout.decode())
            stderr_file.write_text(stderr.decode())

            # Save metadata
            metadata = {
                "email": email,
                "url": url,
                "prompt": prompt_name,
                "timestamp": datetime.now(IST).isoformat(),
                "attempt": attempt_number,
                "exit_code": process.returncode
            }
            metadata_file.write_text(json.dumps(metadata, indent=2))

        evaluation_status[evaluation_key] = {
            "status": "completed",
            "timestamp": datetime.now(IST),
            "total_prompts": len(promptfoo_dirs)
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        evaluation_status[evaluation_key] = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(IST)
        }


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Get the auth portal URL from environment
    auth_portal_url = os.getenv("AUTH_PORTAL_URL")
    
    if auth_portal_url:
        # If AUTH_PORTAL_URL is set, redirect to it
        return RedirectResponse(url=auth_portal_url)
    
    # Fallback if AUTH_PORTAL_URL is not set
    return """
    <html>
        <head>
            <title>p2 submission portal</title>
        </head>
        <body>
            <h1>Welcome to Project 2 Submission Portal Backend</h1>
            <p>This is the backend server. Please visit the authentication portal to login.</p>
        </body>
    </html>
    """

def check_status(email):
    promptfoos_dir = PROMPTFOOS_DIR 
    evaluation_logs_dir = Path(EVALUATION_LOGS_DIR)
    
    # Step 1: Check if email directory exists
    email_dir = evaluation_logs_dir / email
    if not email_dir.exists():
        return {"status": "no_attempt"}
    
    # Step 2: Find the numerically highest subdirectory
    try:
        attempt_dirs = [d for d in email_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not attempt_dirs:
            return {"status": "no_attempt"}
        
        # Get the directory with highest numeric name
        latest_attempt_dir = max(attempt_dirs, key=lambda x: int(x.name))
    except Exception as e:
        logger.error(f"Error finding attempt directories for {email}: {e}")
        return {"status": "no_attempt"}
    
    # Step 3: Get promptfoo directory names
    try:
        prompt_dirs = [d.name for d in promptfoos_dir.iterdir() if d.is_dir()]
        if not prompt_dirs:
            logger.error("No prompt directories found in promptfoos")
            return {"status": "no_attempt"}
    except Exception as e:
        logger.error(f"Error reading promptfoos directory: {e}")
        return {"status": "no_attempt"}
    
    # Step 4: Check if latest attempt has all prompt directories and required files
    required_files = ["metadata.json", "output.json", "stderr.txt", "stdout.txt"]
    
    all_files_present = True
    oldest_file_time = None
    
    for prompt_name in prompt_dirs:
        eval_dir = latest_attempt_dir / prompt_name
        
        if not eval_dir.exists():
            all_files_present = False
            break
        
        for file_name in required_files:
            file_path = eval_dir / file_name
            if not file_path.exists():
                all_files_present = False
                break
            
            # Track the oldest file creation time
            try:
                file_stat = file_path.stat()
                file_time = file_stat.st_mtime
                if oldest_file_time is None or file_time < oldest_file_time:
                    oldest_file_time = file_time
            except OSError:
                pass
        
        if not all_files_present:
            break
    
    # Step 5: Determine status based on files and time
    if all_files_present:
        return {"status": "completed"}
    
    # If files are missing, check if it's been more than 1 hour since directory creation
    try:
        dir_creation_time = latest_attempt_dir.stat().st_mtime
        current_time = time.time()

        # If more than 30 minutes (1800s seconds) has passed since directory creation
        if current_time - dir_creation_time > 1800:
            return {"status": "completed"}  # Assume completed after 30 minutes even if files missing
        else:
            return {"status": "processing"}
    except OSError:
        # If we can't get directory stats, default to processing
        return {"status": "processing"}

@app.get("/submit-form", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    token: str = Query(None),
    success: str = Query(None), 
    error: str = Query(None), 
    submitted: str = Query(None), 
    attempt: str = Query(None),
    current_user = Depends(get_current_user)
):
    # Get auth portal URL from environment
    auth_portal_url = os.getenv("AUTH_PORTAL_URL", "")
    
    # If token is provided in query params, set it as a cookie
    response = None
    
    # Check if we have a token in query params but no authenticated user
    if token and not current_user:
        try:
            # Decode the token
            current_user = decode_jwt_token(token)
            # Prepare a response with cookie
            response = HTMLResponse()
            response.set_cookie(key="auth_token", value=token, httponly=True, max_age=86400)  # 1 day
        except Exception as e:
            return HTMLResponse(content=f"""
            <html>
                <head><title>Authentication Error</title></head>
                <body>
                    <h1>Authentication Error</h1>
                    <p>Invalid or expired token. Please <a href="{auth_portal_url}">login again</a>.</p>
                    <p>Error: {str(e)}</p>
                </body>
            </html>
            """)
    
    # If still no authenticated user, return error
    if not current_user:
        return HTMLResponse(content="""
        <html>
            <head><title>Authentication Required</title></head>
            <body>
                <h1>Authentication Required</h1>
                <p>Please <a href="{auth_portal_url}">login</a> to access this page.</p>
            </body>
        </html>
        """)
    
    # Proceed with the form rendering
    submission_count = count_user_submissions(current_user["email"])
    remaining = MAX_SUBMISSIONS_PER_USER - submission_count
    status_info = check_status(email=current_user["email"])
    status = status_info["status"]
    status_message = ""
    
    # Add processing message if applicable
    if status == "processing":
        status_message += '''
        <div style="background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0;">
            ⏳ Your latest submission is still being processed. Please check back later to download results.
        </div>
        '''
    
    # Add submitted message if applicable
    if submitted == "true" and attempt:
        # Get GitHub repo status from submission.csv if available
        github_status = ""
        try:
            with open("submission.csv", newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row["Email"] == current_user["email"] and row["Attempt"] == attempt:
                        github_repo = row.get("GitHub Repo", "")
                        if github_repo:
                            github_status = f'''
                            <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                ✓ GitHub repository submitted: <a href="{github_repo}" target="_blank">{github_repo}</a>
                                <br>✓ MIT license found in repository
                            </div>
                            '''
        except Exception as e:
            logger.error(f"Error reading GitHub repo status: {e}")
            
        status_message += f'''
        <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0;">
            ✅ Your submission (attempt #{attempt}) has been received and evaluation has started.
            {github_status}
        </div>
        '''

    # Determine whether to show download button
    show_download_button = (status == "completed")

    # Form HTML (only if submissions are left AND no evaluation is in progress)
    form_html = '''
        <form action="/submit" method="post" style="margin: 20px 0;" onsubmit="showSubmissionMessage()">
            <label for="url" style="display: block; margin-bottom: 5px;">Project URL:</label>
            <input type="url" id="url" name="url" required style="width: 400px; padding: 5px; margin-bottom: 10px;" placeholder="https://example.com/your-project">
            <br>
            <label for="github_repo" style="display: block; margin-bottom: 5px;">GitHub Repository URL:</label>
            <input type="url" id="github_repo" name="github_repo" required style="width: 400px; padding: 5px; margin-bottom: 10px;" placeholder="https://github.com/username/repository">
            <span id="github-check-button" class="button" style="background-color: #007bff; color: white; padding: 5px 10px; border: none; cursor: pointer; margin-left: 10px;" onclick="checkGitHubRepo()">Check GitHub Repo</span>
            <div id="github-status" style="margin-top: 5px;"></div>
            <br>
            <input type="submit" value="Submit Project" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer;">
            <div id="submission-message" style="display: none; background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin-top: 10px;">
                📤 Submitting your endpoint... Please wait while we start the evaluation process.
            </div>
        </form>
        <script>
            function showSubmissionMessage() {
                document.getElementById('submission-message').style.display = 'block';
                document.querySelector('input[type="submit"]').disabled = true;
                document.querySelector('input[type="submit"]').value = 'Submitting...';
            }

            async function checkGitHubRepo() {
                const repoUrl = document.getElementById('github_repo').value;
                if (!repoUrl) {
                    document.getElementById('github-status').innerHTML = '<div style="color: red;">Please enter a GitHub repository URL</div>';
                    return;
                }

                document.getElementById('github-status').innerHTML = '<div style="color: blue;">Checking repository...</div>';
                document.getElementById('github-check-button').disabled = true;
                
                try {
                    const response = await fetch('/validate-github-repo?repo_url=' + encodeURIComponent(repoUrl));
                    const data = await response.json();
                    
                    if (data.valid) {
                        document.getElementById('github-status').innerHTML = '<div style="color: green;">✓ ' + data.message + '</div>';
                    } else {
                        document.getElementById('github-status').innerHTML = '<div style="color: red;">✗ ' + data.message + '</div>';
                    }
                } catch (error) {
                    document.getElementById('github-status').innerHTML = '<div style="color: red;">Error checking repository: ' + error.message + '</div>';
                } finally {
                    document.getElementById('github-check-button').disabled = false;
                }
            }
        </script>
        ''' if remaining > 0 and status != "processing" else (
            '<p style="color: red; font-weight: bold;">You have reached the maximum number of submissions.</p>'
            if remaining == 0 else
            '<p style="color: orange; font-weight: bold;">An evaluation is currently in progress. Please wait until it finishes before submitting a new project.</p>'
        )

    # Final HTML page
    html_content = f"""
    <html>
        <head>
            <title>Submit Project</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .info {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Submit Project 2</h1>
            <div class="info">
                <p><strong>Welcome:</strong> {current_user["name"]} ({current_user["email"]})</p>
                <p><strong>Submissions made:</strong> {submission_count}/{MAX_SUBMISSIONS_PER_USER}</p>
                <p class="{'success' if remaining > 2 else 'warning' if remaining > 0 else 'error'}">
                    <strong>Remaining submissions:</strong> {remaining}
                </p>
            </div>

            {status_message}
            {form_html}

            <div style="margin-top: 30px;">
                {f'''
                <a href="/download/{current_user["email"]}"
                style="background-color: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 3px;">
                📦 Download Results ZIP
                </a>
                ''' if show_download_button else ''}
                
                <a href="/logout"
                style="background-color: #f44336; color: white; padding: 10px 20px; text-decoration: none; border-radius: 3px; margin-left: 10px;">
                Logout
                </a>
            </div>
        </body>
    </html>
    """
    
    # If we need to set a cookie, use the response we prepared
    if response:
        response.body = html_content.encode()
        return response
    
    # Otherwise just return the HTML
    return html_content


@app.get("/validate-github-repo")
async def validate_github_repository(repo_url: str = Query(...)):
    """Validate if a GitHub repository exists and has an MIT license"""
    is_valid, message = await validate_github_repo(repo_url)
    return {
        "valid": is_valid,
        "message": message
    }


@app.post("/submit")
async def submit_p2(
    background_tasks: BackgroundTasks, 
    url: str = Form(...), 
    github_repo: str = Form(...), 
    current_user = Depends(require_auth)
):
    if not url:
        raise HTTPException(status_code=400, detail="Submission URL is required")
    
    if not github_repo:
        raise HTTPException(status_code=400, detail="GitHub repository URL is required")
    
    # Validate GitHub repository
    is_valid, message = await validate_github_repo(github_repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    email = current_user["email"]
    
    # Check submission limit
    submission_count = count_user_submissions(email)
    if submission_count >= MAX_SUBMISSIONS_PER_USER:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_SUBMISSIONS_PER_USER} submissions allowed per user")
    
    # Create user directory structure
    safe_email = email.replace("@", "_at_").replace(".", "_")
    attempt_number = get_next_attempt_number(email)
    user_dir = Path("evaluations") / safe_email
    attempt_dir = user_dir / str(attempt_number)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    
    # Log submission immediately
    if not os.path.exists("submission.csv"):
        with open("submission.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Email", "URL", "GitHub Repo", "Timestamp", "Attempt"])
    
    with open("submission.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([email, url, github_repo, datetime.now(IST), attempt_number])

    # Start background evaluation
    background_tasks.add_task(run_evaluation_background, email, url, github_repo, attempt_number, attempt_dir)
    
    # Redirect immediately with submission confirmation
    response = RedirectResponse(url=f"/submit-form?submitted=true&attempt={attempt_number}", status_code=303)
    return response


@app.get("/download/{email}")
async def download_user_results(email: str, background_tasks: BackgroundTasks, current_user=Depends(require_auth)):
    # Ensure user is accessing their own data
    if current_user["email"] != email:
        raise HTTPException(status_code=403, detail="You can only download your own results")
    
    # Step 1: Check if email directory exists
    email_dir = EVALUATION_LOGS_DIR / email
    if not email_dir.exists():
        raise HTTPException(status_code=404, detail="No results found for this user")
    
    # Step 2: Find the numerically highest subdirectory
    try:
        attempt_dirs = [d for d in email_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not attempt_dirs:
            raise HTTPException(status_code=404, detail="No attempts found for this user")
        
        # Get the directory with highest numeric name
        latest_attempt_dir = max(attempt_dirs, key=lambda x: int(x.name))
    except Exception as e:
        logger.error(f"Error finding attempt directories for {email}: {e}")
        raise HTTPException(status_code=404, detail="No valid attempts found for this user")
    
    # Step 3: Check if the latest attempt directory has content
    if not any(latest_attempt_dir.iterdir()):
        raise HTTPException(status_code=404, detail="Latest attempt directory is empty")

    # Step 4: Create temporary zip file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            zip_path = Path(tmp_file.name)
            
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in latest_attempt_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(latest_attempt_dir)
                    zipf.write(file_path, arcname)
        
        # Check if zip file was created and has content
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            raise HTTPException(status_code=404, detail="No files found to download")
            
    except Exception as e:
        logger.error(f"Error creating zip file for {email}: {e}")
        # Clean up partial zip file if it exists
        if 'zip_path' in locals() and zip_path.exists():
            os.remove(zip_path)
        raise HTTPException(status_code=500, detail="Error creating download file")

    # Step 5: Schedule the temp file for deletion after response
    background_tasks.add_task(os.remove, zip_path)

    # Include attempt number in filename for clarity
    attempt_number = latest_attempt_dir.name
    filename = f"{email}_attempt_{attempt_number}_results.zip"

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=filename
    )


@app.get("/status/{email}/{attempt}")
async def get_evaluation_status(email: str, attempt: int, current_user = Depends(require_auth)):
    """Get the status of a specific evaluation"""
    if current_user["email"] != email:
        raise HTTPException(status_code=403, detail="You can only check your own evaluation status")
    
    evaluation_key = f"{email}_{attempt}"
    status = evaluation_status.get(evaluation_key, {"status": "unknown"})
    
    return {
        "email": email,
        "attempt": attempt,
        "evaluation_status": status
    }


# Logout endpoint
@app.get("/logout")
async def logout():
    """Logout the current user and redirect to auth portal"""
    auth_portal_url = os.getenv("AUTH_PORTAL_URL", "")
    response = RedirectResponse(url=f"{auth_portal_url}/logout")
    response.delete_cookie(key="auth_token")
    return response


# API endpoint for token validation (useful for testing)
@app.get("/api/validate-token")
async def validate_token(current_user = Depends(require_auth)):
    return {
        "valid": True,
        "user": current_user
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(IST).isoformat()}


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("uvicorn not installed. Install it with: pip install uvicorn")
        print("Or run with: uvicorn main:app --host 0.0.0.0 --port 8000")
