import requests as r
import os
import json
import time
import tempfile



FOLDER_PATH = "closed_markets_2026_03_06-26"
OUT_FILE = "closed_markets"

GAMMA_API_URL = "https://gamma-api.polymarket.com"
API = "markets"
LIMIT = 500
START_DATE = "2026-03-06T00:00:00Z"
END_DATE = "2026-03-26T23:59:59Z"
CONFIG_FILE = "config.json"
SLEEP_TIME = 0.05

CONFIG_FILE_PATH = os.path.join(FOLDER_PATH, CONFIG_FILE)

# Construct the URL with the specified parameters
base_url = f"{GAMMA_API_URL}/{API}/keyset?"\
          f"limit={LIMIT}"\
          f"&order=start_date"\
          f"&ascending=true"\
          f"&start_date_min={START_DATE}"\
          f"&start_date_max={END_DATE}"\
          f"&end_date_min={START_DATE}"\
          f"&end_date_max={END_DATE}"\
          f"&closed=true"


print(f"Base URL: {base_url}")

def atomic_write(path, text, encoding="utf-8"):
    """
    Atomically write text to a file.

    The target file is replaced only after the full text has been written
    and flushed to disk. If interrupted, the file is either unchanged or
    fully replaced; it should not be half-written.
    """
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=folder,
        prefix=".tmp_",
        suffix=".tmp",
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    except Exception:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise

# Check if the folder exists, if not create it
if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)

# Check if a config.json file exists, if not create it
if not os.path.exists(CONFIG_FILE_PATH):
    atomic_write(CONFIG_FILE_PATH, json.dumps({"base_url": base_url, "next_cursor": None, "counter": 0}))
# Read the config file
with open(CONFIG_FILE_PATH, "r") as f:
    config = json.load(f)

base_url = config.get("base_url")
next_cursor = config.get("next_cursor")
counter = config.get("counter")

if next_cursor:
    url = base_url + f"&after_cursor={next_cursor}"
else:
    url = base_url

# Make the GET request to the API and get the response
response = r.get(url)
data = None
if response.status_code == 200:
    data = response.json()
else:
    print(f"Failed to retrieve data: {response.status_code}")
    exit()


while True:
    objects = data.get(API, [])

    # Write current page
    data_string = "\n".join(
        json.dumps(obj, ensure_ascii=False)
        for obj in objects
    )

    if data_string:
        data_string += "\n"

    page_path = os.path.join(
        FOLDER_PATH,
        f"{OUT_FILE}_{counter:05d}.jsonl"
    )

    atomic_write(page_path, data_string, encoding="utf-8")

    print(f"Wrote page {counter:05d}: {len(objects)} {API}")

    # Get cursor for next page
    next_cursor = data.get("next_cursor")

    # Move counter forward after successfully writing this page
    counter += 1

    # Save config after successful page write
    atomic_write(
        CONFIG_FILE_PATH,
        json.dumps(
            {
                "base_url": base_url,
                "next_cursor": next_cursor,
                "counter": counter,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not next_cursor:
        print("Finished downloading all pages.")
        break

    time.sleep(SLEEP_TIME)

    url = base_url + f"&after_cursor={next_cursor}"
    response = r.get(url, timeout=30)

    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        print(response.text[:500])
        exit()