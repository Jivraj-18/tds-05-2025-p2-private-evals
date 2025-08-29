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
    "-F", "financial-centers.json=@financial-centers.json",
    "-F", "gdp-data.csv=@gdp-data.csv",
    "-F", "stock-market-data.json=@stock-market-data.json"
]

try:
    result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: {e.stderr}")
