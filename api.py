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
    
    def scrape_web_data(self, url, scraping_type="general"):
        """Scrape data from web sources using Playwright with LLM assistance"""
        try:
            # Import Playwright components
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Set a realistic user agent
                page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                
                # Navigate to the URL
                logger.info(f"Navigating to {url}")
                page.goto(url, wait_until="networkidle", timeout=60000)
                
                # Get page content and DOM structure
                html_content = page.content()
                dom_structure = self.extract_dom_structure(page)
                
                # Use LLM to determine scraping strategy based on DOM structure and scraping type
                data = self.extract_with_llm(page, dom_structure, url, scraping_type)
                
                browser.close()
                return data
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {"error": f"Failed to scrape {url}: {str(e)}"}
    
    def extract_dom_structure(self, page):
        """Extract key information about DOM structure for LLM analysis"""
        try:
            return page.evaluate("""() => {
                function getBasicStructure(element, depth = 0, maxDepth = 3) {
                    if (depth > maxDepth || !element) return null;
                    
                    const tagName = element.tagName ? element.tagName.toLowerCase() : 'text';
                    const id = element.id ? `#${element.id}` : '';
                    const classes = element.classList ? Array.from(element.classList).map(c => `.${c}`).join('') : '';
                    
                    let result = {
                        selector: `${tagName}${id}${classes}`,
                        type: tagName
                    };
                    
                    // Add text content for relevant elements
                    if (['h1', 'h2', 'h3', 'h4', 'h5', 'table', 'ul', 'ol', 'div', 'p', 'span', 'a'].includes(tagName)) {
                        const text = element.textContent ? element.textContent.trim() : '';
                        if (text) {
                            result.textPreview = text.substring(0, 100) + (text.length > 100 ? '...' : '');
                        }
                    }
                    
                    // Count elements by type for summary
                    if (depth === 0) {
                        const counts = {};
                        document.querySelectorAll('*').forEach(el => {
                            const tag = el.tagName.toLowerCase();
                            counts[tag] = (counts[tag] || 0) + 1;
                        });
                        result.elementCounts = counts;
                    }
                    
                    // Process important children
                    if (depth < maxDepth) {
                        const children = [];
                        for (const child of element.children || []) {
                            const childStructure = getBasicStructure(child, depth + 1, maxDepth);
                            if (childStructure) children.push(childStructure);
                        }
                        if (children.length > 0) {
                            result.children = children;
                        }
                    }
                    
                    return result;
                }
                
                return {
                    title: document.title,
                    url: window.location.href,
                    structure: getBasicStructure(document.body, 0, 2)
                };
            }""")
        except Exception as e:
            logger.error(f"Error extracting DOM structure: {e}")
            return {"error": "Failed to extract DOM structure"}

    def extract_with_llm(self, page, dom_structure, url, scraping_type):
        """Use LLM to determine how to scrape the page based on DOM structure"""
        if not self.client:
            logger.warning("No OpenAI client available for LLM-assisted scraping")
            return {"error": "LLM-assisted scraping requires OpenAI API key"}
        
        try:
            # Create prompt for LLM with DOM structure
            prompt = f"""
            I need to extract structured data from this webpage: {url}
            
            Type of data needed: {scraping_type}
            
            Here's information about the webpage's DOM structure:
            Title: {dom_structure.get('title', 'Unknown')}
            URL: {dom_structure.get('url', url)}
            
            DOM Structure Summary:
            {json.dumps(dom_structure, indent=2)}
            
            Based on this information, generate a JavaScript function that:
            1. Extracts the most relevant structured data from this page based on the scraping type
            2. Returns a clean JSON-serializable object or array of objects with the extracted data
            3. Handles cases where elements might not exist
            4. Cleans and normalizes the extracted text
            
            The function should be generic and adaptable to the page's structure.
            Return ONLY valid JavaScript code that can be executed with page.evaluate().
            Do not include any explanations, markdown, or code blocks - just the raw JavaScript function.
            """
            
            # Get scraping strategy from LLM
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            # Extract code from response
            js_code = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            js_code = re.sub(r'^```javascript|^```js|^```|```$', '', js_code, flags=re.MULTILINE).strip()
            
            logger.info(f"Executing generated extraction code for {url}")

            # Execute the generated code
            result = page.evaluate(js_code)
            
            # If result is empty, try a second approach
            if not result or (isinstance(result, list) and len(result) == 0) or (isinstance(result, dict) and len(result) == 0):
                logger.warning(f"First extraction attempt returned no data for {url}, trying simpler approach")
                simple_result = self.extract_basic_data(page)
                if simple_result and (len(simple_result) > 0):
                    return simple_result
                return {"warning": "No data could be extracted", "url": url}
                
            return result
                        
        except Exception as e:
            logger.error(f"Error in LLM-assisted scraping: {e}")
            return {"error": str(e), "url": url}
    
    def extract_basic_data(self, page):
        """Extract basic data from a webpage as fallback when LLM extraction fails"""
        try:
            # Get page metadata
            metadata = page.evaluate("""() => {
                return {
                    title: document.title,
                    url: window.location.href,
                    description: document.querySelector('meta[name="description"]')?.content || "",
                    h1: Array.from(document.querySelectorAll('h1')).map(h => h.textContent.trim()),
                    h2: Array.from(document.querySelectorAll('h2')).map(h => h.textContent.trim()).slice(0, 10),
                    links: Array.from(document.querySelectorAll('a[href]'))
                        .slice(0, 20)
                        .map(a => ({ 
                            text: a.textContent.trim(), 
                            href: a.href 
                        }))
                        .filter(link => link.text && link.text.length > 1),
                    tables: Array.from(document.querySelectorAll('table')).slice(0, 3).map(table => {
                        const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
                        const rows = Array.from(table.querySelectorAll('tr')).slice(headers.length > 0 ? 1 : 0).map(row => {
                            return Array.from(row.querySelectorAll('td')).map(td => td.textContent.trim());
                        });
                        return { headers, rows: rows.slice(0, 10) };
                    })
                };
            }""");
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting basic page data: {e}")
            return {"error": "Failed to extract basic page data"}
    
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
                return self.get_fallback_script(question_content, data_description)
        else:
            logger.warning("No OpenAI API key provided, using fallback script")
            return self.get_fallback_script(question_content, data_description)
            
    def get_fallback_script(self, question_content, data_description):
        """Generate a fallback Python script when OpenAI API fails"""
        # Print the path where this script will be saved
        print(f"Generating fallback script (will be saved as llm-code.py in the current directory)")
        
        # Create a basic script that at least attempts to load the data
        fallback_script = """
import json
import pandas as pd
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

# Print script location for debugging
print(f"Running fallback script: {__file__}")

# Main results dictionary that will be returned as JSON
results = {
    "status": "partial_analysis",
    "message": "Analysis completed with fallback script",
    "data_summary": {}
}

# Try to extract required response format from question content
try:
    print("Analyzing question content for required response format...")
    # Look for question.txt or questions.txt
    question_file_path = 'question.txt' if os.path.exists('question.txt') else 'questions.txt'
    if os.path.exists(question_file_path):
        question_lines = open(question_file_path, 'r').read().lower().split('\n')
        format_lines = [line for line in question_lines if "format" in line or "return" in line or "json" in line]
        if format_lines:
            print(f"Found format requirements: {format_lines}")
            for key in ["result", "analysis", "visualization", "data"]:
                if any(key in line for line in format_lines):
                    if key not in results:
                        results[key] = f"Placeholder for {key}"
    else:
        print("Neither question.txt nor questions.txt was found.")
except Exception as e:
    print(f"Error analyzing question format: {e}")
    results["format_analysis_error"] = str(e)

# Load available data files
data = {}

# Print current directory and list files for debugging
try:
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    files_in_dir = os.listdir(current_dir)
    print(f"Files in current directory: {files_in_dir}")
except Exception as e:
    print(f"Error getting directory info: {e}")

# Individual try-except blocks for each major operation
try:
    print("Checking for CSV files and other data sources...")
    # Check for CSV files
    for filename in os.listdir('.'):
        try:
            if filename.endswith('.csv'):
                data[filename] = pd.read_csv(filename)
                print(f"Loaded {filename}")
            elif filename.endswith('.json'):
                with open(filename, 'r') as f:
                    data[filename] = json.load(f)
                print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            results[f"error_loading_{filename}"] = str(e)
except Exception as e:
    results["error_loading_files"] = str(e)

# Create data summaries - separate try-except block
try:
    print("Creating data summaries...")
    for name, dataset in data.items():
        try:
            if isinstance(dataset, pd.DataFrame):
                results["data_summary"][name] = {
                    "rows": len(dataset),
                    "columns": list(dataset.columns),
                    "sample": dataset.head(3).to_dict(orient='records') if not dataset.empty else []
                }
            elif isinstance(dataset, dict):
                results["data_summary"][name] = {
                    "keys": list(dataset.keys()),
                    "sample": "Dictionary with " + str(len(dataset)) + " keys"
                }
            elif isinstance(dataset, list):
                results["data_summary"][name] = {
                    "length": len(dataset),
                    "sample": dataset[:3] if dataset else []
                }
        except Exception as e:
            results[f"error_summarizing_{name}"] = str(e)
except Exception as e:
    results["error_creating_summaries"] = str(e)

# Try to create basic visualizations for numerical data - separate try-except block
try:
    print("Creating basic visualizations...")
    for name, dataset in data.items():
        if isinstance(dataset, pd.DataFrame) and len(dataset) > 0:
            try:
                # Find numeric columns
                numeric_cols = dataset.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) > 0:
                    plt.figure(figsize=(10, 6))
                    for col in numeric_cols[:3]:  # First 3 numeric columns
                        plt.hist(dataset[col].dropna(), alpha=0.5, label=col)
                    
                    plt.title(f"Distribution of numeric columns in {name}")
                    plt.legend()
                    
                    # Save as base64
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    results[f"visualization_{name}"] = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close()
            except Exception as e:
                results[f"error_visualizing_{name}"] = str(e)
except Exception as e:
    results["error_creating_visualizations"] = str(e)

# Print the final results as JSON
try:
    print("Generating final JSON output...")
    print(json.dumps(results, indent=2))
except Exception as e:
    print(f"Error generating final JSON: {e}")
    print(json.dumps({"error": str(e), "partial_results": str(results)}, indent=2))
"""
        return fallback_script

    def execute_python_script(self, script_content, retry_count=0):
        """Execute Python script with error handling and retry logic"""
        if retry_count >= self.max_retries:
            logger.error("Max retries reached. Unable to execute script successfully.")
            return {"error": "Max retries exceeded"}
            
        if not script_content:
            logger.error("Empty script content provided")
            return {"error": "No script content to execute"}
        
        # Clean the script content - remove any backticks or explanation text
        # script_content = self.clean_script_content(script_content)
        
        try:
            # Create a permanent script in the current directory for easier debugging
            script_path = "llm-code.py" if retry_count == 0 else f"llm-code-retry-{retry_count}.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Print script path to terminal
            print(f"Created LLM script at: {os.path.abspath(script_path)} (Attempt {retry_count+1}/{self.max_retries})")
            
            # Copy any necessary data files to the current directory if needed
            # This is already taken care of since we're running in the current directory
            
            # Execute script in the current directory
            env = os.environ.copy()
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, timeout=300, env=env)

            if result.returncode == 0:
                try:
                    # Try to parse as JSON
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # If not JSON, return as string
                    return {"output": result.stdout, "stderr": result.stderr}
            else:
                logger.error(f"Script execution failed: {result.stderr}")
                
                # Try to fix the script using LLM
                if self.client and retry_count < self.max_retries - 1:
                    # Check if this is the last attempt
                    is_last_attempt = (retry_count == self.max_retries - 2)
                    fixed_script = self.fix_script_with_llm(script_content, result.stderr, is_last_attempt)
                    if fixed_script:
                        return self.execute_python_script(fixed_script, retry_count + 1)
                
                return {"error": result.stderr, "stdout": result.stdout}
                
        except subprocess.TimeoutExpired:
            logger.error("Script execution timed out")
            return {"error": "Script execution timed out"}
        except Exception as e:
            logger.error(f"Unexpected error executing script: {e}")
            return {"error": str(e)}
    
    def clean_script_content(self, script_content):
        """Clean script content to remove markdown formatting and explanations"""
        if not script_content:
            return script_content
            
        # Remove markdown code blocks
        if "```python" in script_content:
            lines = script_content.split("\n")
            clean_lines = []
            in_code_block = False
            for line in lines:
                if line.strip() == "```python" or line.strip() == "```py":
                    in_code_block = True
                    continue
                elif line.strip() == "```" and in_code_block:
                    in_code_block = False
                    continue
                elif in_code_block or not "```" in line:
                    clean_lines.append(line)
            script_content = "\n".join(clean_lines)
        
        # If the whole content is wrapped in backticks, remove them
        script_content = re.sub(r'^```python\n|^```\n|```$', '', script_content, flags=re.MULTILINE).strip()
        
        # Check if we have an explanation before the actual code (common pattern)
        if script_content.startswith("The Python script") or script_content.startswith("Here is"):
            try:
                # Find the first import statement which likely indicates the start of actual code
                import_match = re.search(r'^import\s+', script_content, re.MULTILINE)
                if import_match:
                    start_idx = import_match.start()
                    script_content = script_content[start_idx:]
            except:
                pass
                
        return script_content
    
    def fix_script_with_llm(self, original_script, error_message, is_last_attempt=False):
        """Use LLM to fix script errors
        
        Args:
            original_script: The script that failed
            error_message: The error message from the failure
            is_last_attempt: Flag to indicate if this is the final retry attempt
        """
        if not self.client:
            return None
            
        prompt = f"""
        The following Python script failed with this error:
        
        Error: {error_message}
        
        Original Script:
        {original_script}
        
        Fix the script to resolve the error. Return ONLY the corrected Python code without any explanations or markdown formatting like backticks.
        
        {"THIS IS THE FINAL ATTEMPT. " if is_last_attempt else ""}Use python alone. {"Since this is the final attempt, you MUST" if is_last_attempt else "In the final attempt"} keep try blocks for small segments of code so that whole script does not fail. If some questions/features fail it's okay, but make sure the script runs to completion and returns a valid JSON result with EXACTLY the format specified in the original task.
        
        {"IMPORTANT: Wrap each distinct logical operation in its own try-except block to isolate failures. Ensure the JSON response format matches exactly what was requested in the original task." if is_last_attempt else ""}
        
        CRITICAL: Look for data files in the current working directory. Access files directly with their names without path prefixes:
        - Use open('user-posts.json') NOT open('/path/to/user-posts.json')
        - Use pd.read_csv('network-connections.csv') NOT pd.read_csv('/path/to/network-connections.csv')
        
        MANDATORY: Make sure the script prints the final results as JSON to standard output. Add this code at the end if it's not already there:
        
        # Print results to standard output
        print(json.dumps(results, indent=2))
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a code-fixing assistant that fixes Python code errors. You ONLY output valid Python code with no explanations, no markdown formatting, and no backticks. Your code must be ready to execute as-is. CRITICAL: Always maintain the exact response format required in the original task. Files should be accessed from the current working directory without path prefixes. IMPORTANT: Always ensure the script prints the final results as JSON to standard output using 'print(json.dumps(results, indent=2))'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Remove any backticks that might be present
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
        """Main method to process the entire request"""
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
            
            # Step 5: Execute script with retry logic
            logger.info("Executing analysis script...")
            raw_results = self.execute_python_script(script_content)
            
            # Step 6: Format results
            formatted_results = self.format_results(raw_results, question_content)
            
            # write formatted_results to a file 
            with open("response-api.json","w")  as f :
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
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")