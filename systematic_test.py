import os
import csv
import time
from datetime import datetime
from neuronpedia import steering_completion
from prompts import confabulation_prompts

# --- CONFIGURATION ---
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"runs/stress_test"
LOG_DIR = f"logs/stress_test"

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_FILENAME = f"{RUN_DIR}/stress_run_{RUN_TIMESTAMP}.csv"

# Define Features
# FEAT_UNCERTAINTY = {"index": 80216, "layer": "9-gemmascope-res-131k", "name": "L9-Uncertainty"}
# FEAT_TRUTH = {"index": 107788, "layer": "20-gemmascope-res-131k", "name": "L20-Truth"}

# FEAT_UNCERTAINTY = {"index": 10441, "layer": "9-gemmascope-res-131k", "name": "L9-Uncertainty"}
# FEAT_TRUTH = {"index": 59115, "layer": "20-gemmascope-res-131k", "name": "L20-Truth"}

FEAT_UNCERTAINTY = {"index": 122482, "layer": "9-gemmascope-res-131k", "name": "L9-Uncertainty"}
# FEAT_TRUTH = {"index": 80603, "layer": "20-gemmascope-res-131k", "name": "L20-Truth"}

# Define Test Cases (The Grid Search)
TEST_CASES = [
#     {"name": "Truth Only (Low)",   "features": [(FEAT_TRUTH, 15)]},
#     {"name": "Truth Only (Med)",   "features": [(FEAT_TRUTH, 35)]},
#     {"name": "Truth Only (High)",  "features": [(FEAT_TRUTH, 75)]},
#     {"name": "Truth Only (Max)",   "features": [(FEAT_TRUTH, 120)]},
#     {"name": "Uncertainty (Low)",  "features": [(FEAT_UNCERTAINTY, 20)]},
    {"name": "Uncertainty (Med)",  "features": [(FEAT_UNCERTAINTY, 35)]},
    # {"name": "Uncertainty (High)", "features": [(FEAT_UNCERTAINTY, 60)]},
    # {"name": "Combo (Baseline)",   "features": [(FEAT_UNCERTAINTY, 35), (FEAT_TRUTH, 15)]},
    # {"name": "Combo (High-High)", "features": [(FEAT_UNCERTAINTY, 60), (FEAT_TRUTH, 75)]},
    # {"name": "Combo (Low-Low)",   "features": [(FEAT_UNCERTAINTY, 20), (FEAT_TRUTH, 10)]},
]

# Prompts to stress test
PROMPTS = confabulation_prompts["best_examples"]

print(f"--- STARTING SYSTEMATIC STRESS TEST ---")
print(f"Testing {len(PROMPTS)} prompts across {len(TEST_CASES)} configurations.")

with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ["prompt", "config_name", "settings", "steered_response"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for prompt in PROMPTS:
        print(f"\n>>> PROMPT: {prompt}")
        
        for case in TEST_CASES:
            # Construct settings string for record keeping
            settings_str = ", ".join([f"{f[0]['name']}={f[1]}" for f in case['features']])
            print(f"   Testing: {case['name']} [{settings_str}]")
            
            try:
                # We only need the steered response for stress testing usually
                # Adjusting logic to unpack your function return correctly
                features_for_call = [(f["index"], s, f["layer"]) for (f, s) in case["features"]]
                steered, _ = steering_completion(prompt, features_for_call)
                clean_result = steered.replace("\n", " ").replace("\r", "")
                
                writer.writerow({
                    "prompt": prompt,
                    "config_name": case['name'],
                    "settings": settings_str,
                    "steered_response": clean_result
                })
                print(f"      Result: {clean_result[:80]}...")
                
            except Exception as e:
                print(f"      ERROR: {e}")
                writer.writerow({
                    "prompt": prompt,
                    "config_name": case['name'],
                    "settings": settings_str,
                    "steered_response": f"ERROR: {e}"
                })
            
            time.sleep(1)

print(f"\nDone! Systematic results saved to {CSV_FILENAME}")