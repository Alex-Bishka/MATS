import os
import csv
import time
from datetime import datetime
from neuronpedia import steering_completion

from prompts import MISCONCEPTIONS

# --- CONFIGURATION ---
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"runs/misconceptions"
LOG_DIR = f"logs/misconceptions"

# Create directories if they don't exist
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_FILENAME = f"{RUN_DIR}/run_{RUN_TIMESTAMP}.csv"
LOG_FILENAME = f"{LOG_DIR}/run_{RUN_TIMESTAMP}.txt"

# The "Doubt-Truth" Combo
FEATURE_SET = [
    (80216, 35, "9-gemmascope-res-131k"),   # L9: Uncertainty (The Destabilizer)
    (107788, 15, "20-gemmascope-res-131k")  # L20: Truth (The Guide)

    # (10441, 35, "9-gemmascope-res-131k"),
    # (59115, 15, "20-gemmascope-res-131k")
]

def log_print(message, file):
    print(message)
    file.write(message + "\n")

print(f"--- STARTING GENERALIZATION TEST (n={len(MISCONCEPTIONS)}) ---")
print(f"Saving to: {CSV_FILENAME}")

with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as csv_file, \
     open(LOG_FILENAME, mode='w', encoding='utf-8') as log_file:
    
    writer = csv.DictWriter(csv_file, fieldnames=["prompt", "default_response", "steered_response", "status"])
    writer.writeheader()

    for i, prompt in enumerate(MISCONCEPTIONS, 1):
        log_print(f"[{i}/{len(MISCONCEPTIONS)}] Testing: '{prompt}'", log_file)
        
        try:
            steered, default = steering_completion(prompt, FEATURE_SET)
            
            # Clean strings for CSV
            clean_default = default.replace("\n", " ").replace("\r", "")
            clean_steered = steered.replace("\n", " ").replace("\r", "")
            
            writer.writerow({
                "prompt": prompt,
                "default_response": clean_default,
                "steered_response": clean_steered,
                "status": "Success"
            })
            
            log_print(f"   -> Default: {clean_default}", log_file)
            log_print(f"   -> Steered: {clean_steered}", log_file)
            
        except Exception as e:
            log_print(f"   -> ERROR: {e}", log_file)
            writer.writerow({"prompt": prompt, "status": f"Error: {e}"})

        log_print("-" * 60, log_file)
        time.sleep(1) # Rate limit kindness

print("\nDone! Results saved.")