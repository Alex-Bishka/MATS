from flask import Flask, render_template, request, jsonify
import pandas as pd
from pathlib import Path

app = Flask(__name__)

RUNS_BASE = Path(__file__).parent.parent / "runs"
RUN_TYPES = {
    "misconceptions": {
        "dir": RUNS_BASE / "misconceptions",
        "title": "Misconceptions Runs",
        "empty_msg": "No runs found in runs/misconceptions/",
        "template": "run.html",
    },
    "stress_test": {
        "dir": RUNS_BASE / "stress_test",
        "title": "Stress Test Runs",
        "empty_msg": "No runs found in runs/stress_test/",
        "template": "run_stress.html",
    },
}


def get_available_runs(run_type: str):
    """Get list of available CSV run files for a given run type."""
    runs_dir = RUN_TYPES[run_type]["dir"]
    csv_files = sorted(runs_dir.glob("*.csv"), reverse=True)
    return [f.stem for f in csv_files]


def load_run_data(run_type: str, run_name: str) -> pd.DataFrame:
    """Load data from a specific run CSV file."""
    runs_dir = RUN_TYPES[run_type]["dir"]
    csv_path = runs_dir / f"{run_name}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


@app.route("/")
def index():
    """Main page showing misconceptions runs (default)."""
    return index_by_type("misconceptions")


@app.route("/<run_type>")
def index_by_type(run_type: str):
    """Main page showing list of available runs for a given type."""
    if run_type not in RUN_TYPES:
        return render_template("error.html", message=f"Unknown run type '{run_type}'"), 404

    runs = get_available_runs(run_type)
    return render_template(
        "index.html",
        runs=runs,
        run_type=run_type,
        run_types=RUN_TYPES,
        title=RUN_TYPES[run_type]["title"],
        empty_msg=RUN_TYPES[run_type]["empty_msg"],
    )


@app.route("/<run_type>/run/<run_name>")
def view_run(run_type: str, run_name: str):
    """View details of a specific run."""
    if run_type not in RUN_TYPES:
        return render_template("error.html", message=f"Unknown run type '{run_type}'"), 404

    df = load_run_data(run_type, run_name)
    if df.empty:
        return render_template("error.html", message=f"Run '{run_name}' not found"), 404

    # Ensure 'corrected' column exists
    if "corrected" not in df.columns:
        df["corrected"] = False

    records = df.to_dict("records")

    # Extract unique filter options for stress tests
    filter_options = {}
    if run_type == "stress_test":
        filter_options = {
            "config_names": sorted(df["config_name"].unique().tolist()),
            "settings": sorted(df["settings"].unique().tolist()),
        }

    template = RUN_TYPES[run_type]["template"]
    return render_template(
        template,
        run_name=run_name,
        run_type=run_type,
        records=records,
        title=RUN_TYPES[run_type]["title"],
        filter_options=filter_options,
    )


@app.route("/<run_type>/run/<run_name>/update_corrected", methods=["POST"])
def update_corrected(run_type: str, run_name: str):
    """Update the corrected status for a specific record."""
    if run_type not in RUN_TYPES:
        return jsonify({"error": f"Unknown run type '{run_type}'"}), 404

    data = request.get_json()
    record_index = data.get("index")
    corrected = data.get("corrected", False)

    runs_dir = RUN_TYPES[run_type]["dir"]
    csv_path = runs_dir / f"{run_name}.csv"

    if not csv_path.exists():
        return jsonify({"error": f"Run '{run_name}' not found"}), 404

    df = pd.read_csv(csv_path)

    # Ensure 'corrected' column exists
    if "corrected" not in df.columns:
        df["corrected"] = False

    # Update the specific record
    if 0 <= record_index < len(df):
        df.at[record_index, "corrected"] = corrected
        df.to_csv(csv_path, index=False)
        return jsonify({"success": True, "corrected": corrected})
    else:
        return jsonify({"error": "Invalid record index"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
