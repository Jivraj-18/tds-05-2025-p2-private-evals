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
    "--max-time", "300",
    url,
    "-F", "questions.txt=@question.txt",
    "-F", "co2-emissions.json=@co2-emissions.json",
    "-F", "temperature-anomalies.csv=@temperature-anomalies.csv",
    "-F", "weather-stations.json=@weather-stations.json"
]

try:
    result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: {e.stderr}")
