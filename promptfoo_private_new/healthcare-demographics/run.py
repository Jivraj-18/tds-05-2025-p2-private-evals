# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
# ]
# ///

import subprocess
import sys

url = sys.argv[1]

curl_command = [
    "curl",
    url,
    "-F", "questions.txt=@question.txt",
    "-F", "disease-data.json=@disease-data.json",
    "-F", "health-indicators.csv=@health-indicators.csv",
    "-F", "hospitals.json=@hospitals.json"
]

try:
    result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: {e.stderr}")
