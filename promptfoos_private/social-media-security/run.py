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
    "--max-time", "180",
    url,
    "-F", "questions.txt=@question.txt",
    "-F", "network-connections.csv=@network-connections.csv",
    "-F", "security-incidents.csv=@security-incidents.csv",
    "-F", "user-posts.json=@user-posts.json"
]

try:
    result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: {e.stderr}")
