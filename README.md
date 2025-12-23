# MATS - Confabulation Steering Experiments

Research project exploring steering interventions to reduce LLM confabulation and sycophancy using Sparse Autoencoders (SAEs).

## Project Structure

- `prompts.py` - Confabulation-inducing test prompts across categories (future events, fictional entities, fake concepts, etc.)
- `feature_validation.py` - SAE feature validation and testing
- `neuronpedia.py` - Integration with Neuronpedia for feature analysis
- `frontend/` - Flask web app for viewing experiment results
- `runs/` - Experiment output data (CSV files)

## Running the Frontend

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Run the Flask app:
   ```bash
   python frontend/app.py
   ```

3. Open http://localhost:5000 in your browser to view experiment results.

## Dependencies

Install with [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```
