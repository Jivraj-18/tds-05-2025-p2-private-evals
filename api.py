# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "statsmodels",
#   "folium",
#   "shapely",
#   "scipy",
#   "geopandas",
#   "fastapi[standard]",
#   "html2text",
#   "uvicorn",
#   "asyncpg",
#   "httpx",
#   "python-jose",
#   "pyjwt",
#   "python-dotenv",
#   "requests",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "networkx",
#   "pillow",          # replaces PIL
#   "opencv-python",   # replaces cv2
#   "openai",
#   "scikit-learn",
#   "python-multipart", # useful for FastAPI file uploads
#   "pyarrow",          # for parquet I/O (instead of `parquet`)
#   "playwright",
#   "beautifulsoup4"      # for web scraping
# ]
# ///



import os
import json
import pandas as pd
import numpy as np
import requests
import subprocess
import sys
import tempfile
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import traceback
import logging
from pathlib import Path
import sqlite3
import pyarrow.parquet as pq
from PIL import Image
import cv2
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalystAutomation:
    def extract_failed_blocks(self, script_content, failed_keys):
        """Extract try/except code blocks for the failed keys from the script."""
        # This is a simple regex-based approach. For more complex scripts, consider using ast parsing.
        import re
        blocks = {}
        for key in failed_keys:
            # Look for try: ... results['key'] = ... except ... results['key_error'] = ...
            # This regex will match the try block that assigns to results['key']
            # and its except block that assigns to results['key_error']
            pattern = rf"""try:\s*\n(.*?results\s*\[\s*['\"]{key}['\"]\s*\].*?\n.*?)except.*?results\s*\[\s*['\"]{key}_error['\"]\s*\].*?\n"""
            matches = re.findall(pattern, script_content, re.DOTALL)
            if matches:
                # Try to get the full try/except block
                try_block_pattern = rf"try:[\s\S]*?results\s*\[\s*['\"]{key}_error['\"]\s*\][\s\S]*?except[\s\S]*?\n"
                try_block = re.search(try_block_pattern, script_content)
                if try_block:
                    blocks[key] = try_block.group(0)
                else:
                    blocks[key] = matches[0]
            else:
                # If not found, just note that extraction failed
                blocks[key] = f"# Could not extract code block for {key}"
        return blocks
    def __init__(self, openai_api_key=None):
        """Initialize the automated data analyst system"""
        self.openai_api_key = openai_api_key
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = None
        
        self.max_retries = 3
        # We'll use the current directory instead of a temp directory
        self.temp_dir = os.getcwd()
        self.results = {}
        
    def read_question_file(self, question_file_path):
        """Read and parse the question.txt file"""
        try:
            with open(question_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Successfully read question file: {len(content)} characters")
            return content
        except Exception as e:
            logger.error(f"Error reading question file: {e}")
            return None
    
    def process_uploaded_files(self, file_paths):
        """Process all uploaded files and load them into appropriate data structures"""
        processed_data = {}
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()
            
            try:
                if file_ext == '.csv':
                    processed_data[file_name] = pd.read_csv(file_path)
                    logger.info(f"Loaded CSV: {file_name} with {len(processed_data[file_name])} rows")
                    
                elif file_ext == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        processed_data[file_name] = json.load(f)
                    logger.info(f"Loaded JSON: {file_name}")
                    
                elif file_ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        processed_data[file_name] = f.read()
                    logger.info(f"Loaded TXT: {file_name}")
                    
                elif file_ext == '.parquet':
                    processed_data[file_name] = pd.read_parquet(file_path)
                    logger.info(f"Loaded Parquet: {file_name}")
                    
                elif file_ext == '.db' or file_ext == '.sqlite':
                    conn = sqlite3.connect(file_path)
                    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                    processed_data[file_name] = {}
                    for table_name in tables['name']:
                        processed_data[file_name][table_name] = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    conn.close()
                    logger.info(f"Loaded SQLite DB: {file_name}")
                    
                elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    processed_data[file_name] = cv2.imread(file_path)
                    logger.info(f"Loaded Image: {file_name}")
                    
                elif file_ext == '.pdf':
                    # For PDF processing, we'll generate code to extract text
                    processed_data[file_name] = file_path  # Store path for later processing
                    logger.info(f"Marked PDF for processing: {file_name}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                processed_data[file_name] = None
                
        return processed_data
            
    def generate_python_script(self, question_content, data_description):
        """Use LLM to generate Python script for analysis"""
        prompt = f"""
            You are required to generate a single, complete, and runnable Python script that fully solves the following task. 

            Question/Task:
            {question_content}

            Available Data:
            {data_description}

            STRICT REQUIREMENTS:
            1. Your output must be a single complete Python script with fully implemented, runnable code for all tasks. No notes, explanations, markdown, placeholders, or comments.
            2. Handle missing data gracefully.
            3. Generate all required visualizations as base64 PNG images under 100kB.
            4. Return results as a JSON object with EXACTLY the keys and format requested in the Question/Task.
            5. CRITICAL: Each question or required key in the JSON output MUST be implemented in its own logical block with its own try-except. 
            - Do NOT wrap multiple questions together under one try/except.
            - Each answer/plot/analysis step should be protected by a small, localized try-except block.
            6. Use mock/sample data if external APIs are not available.
            7. Do NOT include any triple backticks (```) or markdown formatting.
            8. The code must be ready to execute AS-IS without any modifications.
            9. Always return ONLY valid Python code — no natural language text at all outside of comments inside the code.
            10. CRITICAL: Match the exact response format and JSON keys requested in the Question/Task.
            11. IMPORTANT: Load data files directly from the current working directory (e.g., open('user-posts.json'), pd.read_csv('network-connections.csv')) without path prefixes.
            12. MANDATORY: At the end of your script, print the final results as JSON to standard output. Always include:
                print(json.dumps(results, indent=2))
            13. If a public API exists for the requested data, use it. Otherwise:
            * 1) Fetch the webpage HTML.
            * 2) Convert the HTML to Markdown with html2text (include the complete content).
            * 3) Send the complete Markdown to OpenAI to obtain structured JSON.
            * 4) When calling OpenAI:

            * Use client.chat.completions.create(model="gpt-4.1", ...).
            * You must use gpt-4.1 model only.
            * The messages must instruct the model to return only a single compact/minified JSON object with no spaces or line breaks, no markdown, no backticks, and no explanations; use exactly and only the required keys; use null for missing values; if extraction fails, return {{"error":"reason"}}; do not include any text before or after the JSON.
            * 5) When scraping:

            * Follow robots.txt and site TOS, use a polite User-Agent, add small randomized delays, avoid heavy/abusive requests, and never bypass CAPTCHAs, paywalls, or access controls.

            

            14. Try-Expect:
            - Each major operation that corresponds to a question or output key must have its own try-except.
            - If one visualization or analysis fails, continue with the next.
            - Add appropriate error messages to the results dictionary when operations fail.
            - Example pattern:
            results = {{}}
            # Q1
            
            try:
                results['total_posts_analyzed'] = len(posts)
            except Exception as e:
                results['total_posts_analyzed_error'] = f"{{type(e).__name__}}, {{e}}"

            # Q2
            try:
                results['high_risk_users'] = risky_users
            except Exception as e:
                results['high_risk_users_error'] = f"{{type(e).__name__}}, {{e}}"

            FINAL INSTRUCTION:
            Do not add any notes, disclaimers, or partial/incomplete indicators. 
            The output must be a single, complete, runnable Python script only, with one try/except per logical requirement.
            """


        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are a code-generation assistant that writes clean, executable Python code. You ONLY output valid Python code with no explanations, no markdown formatting, and no backticks. Your code must be ready to execute as-is. CRITICAL: You must return results in EXACTLY the format specified in the question/task, matching all required fields and structure precisely. IMPORTANT: Always end your script by printing the final results as JSON to standard output using 'print(json.dumps(results, indent=2))'. This is essential for capturing the output."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error generating script with OpenAI: {e}")
                return


    def execute_python_script(self, script_content, retry_count=0, prev_results=None, question_content=None):
        """Execute Python script, handle per-key errors, and iteratively fix only failed keys."""
        if retry_count >= self.max_retries:
            logger.error("Max retries reached. Unable to execute script successfully.")
            return prev_results if prev_results else {"error": "Max retries exceeded"}

        if not script_content:
            logger.error("Empty script content provided")
            return prev_results if prev_results else {"error": "No script content to execute"}

        try:
            script_path = "llm-code.py" if retry_count == 0 else f"llm-code-retry-{retry_count}.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            print(f"Created LLM script at: {os.path.abspath(script_path)} (Attempt {retry_count+1}/{self.max_retries})")
            env = os.environ.copy()
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, timeout=300, env=env)

            if result.returncode == 0:
                try:
                    output_json = json.loads(result.stdout)
                except json.JSONDecodeError:
                    logger.error("Script did not return valid JSON")
                    return {"output": result.stdout, "stderr": result.stderr}

                # Merge with previous results if any
                merged_results = dict(prev_results) if prev_results else {}
                merged_results.update(output_json)

                # Find failed keys (ending with _error)
                failed_keys = [k for k in merged_results if k.endswith('_error') and merged_results[k]]
                passed_keys = [k for k in merged_results if not k.endswith('_error')]

                # If no failed keys or max retries, return merged results
                if not failed_keys or retry_count >= self.max_retries - 1:
                    return merged_results

                # Otherwise, fix only failed keys
                if self.client:
                    # Extract only the failed code blocks
                    failed_blocks = self.extract_failed_blocks(script_content, failed_keys)
                    fixed_code = self.fix_script_with_llm(
                        failed_blocks=failed_blocks,
                        error_keys=failed_keys,
                        error_messages={k: merged_results[k] for k in failed_keys},
                        passed_keys=passed_keys,
                        question_content=question_content,
                        is_last_attempt=(retry_count == self.max_retries - 2)
                    )
                    if fixed_code:
                        # Run only the fixed code, merge new results
                        new_results = self.execute_python_script(
                            fixed_code, retry_count + 1, merged_results, question_content
                        )
                        return new_results
                return merged_results
            else:
                logger.error(f"Script execution failed: {result.stderr}")
                return prev_results if prev_results else {"error": result.stderr, "stdout": result.stdout}
        except subprocess.TimeoutExpired:
            logger.error("Script execution timed out")
            return prev_results if prev_results else {"error": "Script execution timed out"}
        except Exception as e:
            logger.error(f"Unexpected error executing script: {e}")
            return prev_results if prev_results else {"error": str(e)}
    
     
    def fix_script_with_llm(self, failed_blocks, error_keys, error_messages, passed_keys, question_content, is_last_attempt=False):
        """Use LLM to fix only the failed code blocks, not the whole script."""
        if not self.client:
            return None

        # Prepare the code blocks for the prompt
        code_blocks_str = "\n\n".join([f"# Block for {key}:\n{failed_blocks[key]}" for key in error_keys])

        prompt = f"""
            You are required to generate Python code that fixes only the failed parts of a previous script for the following task.

            Question/Task:
            {question_content}

            The following keys in the results had errors (with their error messages):
            {json.dumps(error_messages, indent=2)}

            The following keys PASSED and do NOT need to be regenerated:
            {json.dumps(passed_keys)}

            The code blocks below correspond to the failed keys. Only these blocks need to be fixed:
            {code_blocks_str}

            STRICT REQUIREMENTS:
            1. Your output must be a single, complete, and runnable Python code block that ONLY computes the failed keys. No notes, explanations, markdown, placeholders, or comments outside the code.
            2. Handle missing data gracefully.
            3. If a failed key requires a visualization, generate it as a base64 PNG image under 100kB.
            4. Return results as a JSON object with EXACTLY the failed keys (and their _error keys if any error occurs).
            5. CRITICAL: Each failed key in the JSON output MUST be implemented in its own logical block with its own try-except. 
            - Do NOT wrap multiple keys together under one try/except.
            - Each answer/plot/analysis step should be protected by a small, localized try-except block.
            6. Use mock/sample data if external APIs are not available.
            7. Do NOT include any triple backticks (```) or markdown formatting.
            8. The code must be ready to execute AS-IS without any modifications.
            9. Always return ONLY valid Python code — no natural language text at all outside of comments inside the code.
            10. CRITICAL: Match the exact response format and JSON keys requested in the original task, but only for the failed keys.
            11. IMPORTANT: Load data files directly from the current working directory (e.g., open('user-posts.json'), pd.read_csv('network-connections.csv')) without path prefixes.
            12. MANDATORY: At the end of your script, print the final results as JSON to standard output. Always include:
                print(json.dumps(results, indent=2))
            13. If a public API exists for the requested data, use it. Otherwise:
            * 1) Fetch the webpage HTML.
            * 2) Convert the HTML to Markdown with html2text (include the complete content).
            * 3) Send the complete Markdown to OpenAI to obtain structured JSON.
            * 4) When calling OpenAI:
            * Use client.chat.completions.create(model="gpt-4.1", ...).
            * You must use gpt-4.1 model only.
            * The messages must instruct the model to return only a single compact/minified JSON object with no spaces or line breaks, no markdown, no backticks, and no explanations; use exactly and only the required keys; use null for missing values; if extraction fails, return {"error":"reason"}; do not include any text before or after the JSON.
            * 5) When scraping:
            * Follow robots.txt and site TOS, use a polite User-Agent, add small randomized delays, avoid heavy/abusive requests, and never bypass CAPTCHAs, paywalls, or access controls.

            14. Try-Expect:
            - Each failed key must have its own try-except.
            - If one visualization or analysis fails, continue with the next.
            - Add appropriate error messages to the results dictionary when operations fail.
            - Example pattern:
            results = {{}}
            # failed_key_1
            try:
                results['failed_key_1'] = ...
            except Exception as e:
                results['failed_key_1_error'] = f"{{type(e).__name__}}, {{e}}"

            FINAL INSTRUCTION:
            Do not add any notes, disclaimers, or partial/incomplete indicators. 
            The output must be a single, complete, runnable Python code block only, with one try/except per failed key.
            {"THIS IS THE FINAL ATTEMPT. If a key still fails, just return the error message in the <key>_error field." if is_last_attempt else ""}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a code-fixing assistant that fixes only the failed parts of a Python script. You ONLY output valid Python code with no explanations, no markdown formatting, and no backticks. Your code must be ready to execute as-is. CRITICAL: Only generate code for the failed keys, and always print the results as JSON to standard output using 'print(json.dumps(results, indent=2))'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            fixed_code = response.choices[0].message.content.strip()
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code[10:]
            if fixed_code.startswith("```"):
                fixed_code = fixed_code[3:]
            if fixed_code.endswith("```"):
                fixed_code = fixed_code[:-3]
            return fixed_code.strip()
        except Exception as e:
            logger.error(f"Error fixing script with LLM: {e}")
            return None
    
    def format_results(self, raw_results, question_content):
        """Format results according to the requirements in question.txt"""
        try:
            if isinstance(raw_results, dict):
                return raw_results
            elif isinstance(raw_results, str):
                return json.loads(raw_results)
            else:
                return {"error": "Unable to format results", "raw_output": str(raw_results)}
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return {"error": f"Formatting error: {str(e)}", "raw_output": str(raw_results)}
    
    def process_request(self, question_file_path, additional_file_paths):
        """Main method to process the entire request with iterative per-key error fixing."""
        try:
            # Step 1: Read question file
            question_content = self.read_question_file(question_file_path)
            if not question_content:
                return {"error": "Unable to read question file"}

            # Step 2: Process uploaded files
            processed_data = self.process_uploaded_files(additional_file_paths)

            # Step 3: Create data description for LLM
            data_description = {}
            for filename, data in processed_data.items():
                if isinstance(data, pd.DataFrame):
                    data_description[filename] = {
                        "type": "DataFrame",
                        "shape": data.shape,
                        "columns": list(data.columns)
                    }
                elif isinstance(data, dict):
                    data_description[filename] = {
                        "type": "Dictionary",
                        "keys": list(data.keys()) if data else []
                    }
                elif isinstance(data, list):
                    data_description[filename] = {
                        "type": "List",
                        "length": len(data),
                        "sample": data[:2] if len(data) > 0 else []
                    }
                else:
                    data_description[filename] = {
                        "type": type(data).__name__,
                        "info": "Available for processing"
                    }

            # Step 4: Generate Python script using LLM
            logger.info("Generating analysis script...")
            script_content = self.generate_python_script(question_content, json.dumps(data_description, indent=2))

            # Step 5: Execute script with iterative per-key error fixing
            logger.info("Executing analysis script...")
            raw_results = self.execute_python_script(script_content, retry_count=0, prev_results=None, question_content=question_content)

            # Step 6: Format results
            formatted_results = self.format_results(raw_results, question_content)

            # write formatted_results to a file
            with open("response-api.json", "w") as f:
                f.write(json.dumps(formatted_results, indent=2))

            return formatted_results

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": f"Processing error: {str(e)}", "traceback": traceback.format_exc()}

# FastAPI API endpoint wrapper
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Any, Optional
import shutil

app = FastAPI(title="Data Analyst Automation API", 
              description="API for automated data analysis using LLM",
              version="1.0.0")

# Initialize the automation system
analyst = DataAnalystAutomation(openai_api_key=os.getenv('OPENAI_API_KEY'))

@app.post("/api", response_class=JSONResponse)
async def analyze_data(request: Request):
    try:
        # Process the multipart form data
        form_data = await request.form()
        
        # Check if questions.txt is present
        if "questions.txt" not in form_data:
            raise HTTPException(status_code=400, detail="questions.txt file is required")
            
        # Save uploaded files to the current directory for easier access
        upload_dir = os.getcwd()
        file_paths = []
        
        # Save question file
        question_file = form_data["questions.txt"]
        question_path = os.path.join(upload_dir, 'questions.txt')
        
        # Handle different types of form data
        if hasattr(question_file, "filename"):
            # Save the question file content
            with open(question_path, "wb") as buffer:
                content = await question_file.read()
                buffer.write(content)
        else:
            # Handle string data
            with open(question_path, "w", encoding="utf-8") as buffer:
                buffer.write(str(question_file))
        
        # Save other files
        for field_name, file_obj in form_data.items():
            if field_name != "questions.txt" and hasattr(file_obj, "filename"):
                filepath = os.path.join(upload_dir, file_obj.filename or field_name)
                with open(filepath, "wb") as buffer:
                    content = await file_obj.read()
                    buffer.write(content)
                file_paths.append(filepath)
            elif field_name != "questions.txt":
                # Handle other non-file fields if any
                filepath = os.path.join(upload_dir, field_name)
                with open(filepath, "w", encoding="utf-8") as buffer:
                    buffer.write(str(file_obj))
                file_paths.append(filepath)
        
        logger.info(f"Received files: {', '.join([os.path.basename(p) for p in file_paths])}")
        
        # Process the request
        results = analyst.process_request(question_path, file_paths)
        
        # No need to clean up files as they're in the current directory
        # This ensures files remain accessible for debugging
        
        return results
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return {"error": str(e), "_metadata": {"processed_at": datetime.now().isoformat()}}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == '__main__':
    # For standalone testing
    if len(sys.argv) > 1:
        question_file = sys.argv[1]
        additional_files = sys.argv[2:] if len(sys.argv) > 2 else []
        
        analyst = DataAnalystAutomation()
        results = analyst.process_request(question_file, additional_files)
        print(json.dumps(results, indent=2))
    else:
        # Run FastAPI with uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
