"""Execute the companion notebook and exercise its actual ipywidget callbacks."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for name, directory in [("JUPYTER_CONFIG_DIR", "config"), ("JUPYTER_DATA_DIR", "data"),
                        ("JUPYTER_RUNTIME_DIR", "runtime"), ("IPYTHONDIR", "ipython")]:
    path = ROOT / ".venv/jupyter" / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(name, str(path))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "Scripts/project/artifacts/.matplotlib"))

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


SMOKE = """
import numpy as np
from PIL import Image as PILImage
for view in ['latest', 'holdout']:
    for variant in ['all_inputs', 'without_news']:
        console.view.value = view
        console.model.value = variant
        console.rows.value = 5
        console.threshold.value = 0.03
        console.run_button.click()
        assert console.last_error is None, console.last_error
        assert not console.run_button.disabled
        assert len(console.last_predictions) == 5
        assert np.isfinite(console.last_predictions.predicted_return_5d).all()
        if view == 'holdout':
            saved = pd.read_csv(OUTPUTS_DIR / 'arabica_all_inputs_news_holdout_predictions.csv')
            np.testing.assert_allclose(console.last_predictions.predicted_return_5d,
                saved[f'{variant}_predicted_return_5d'].tail(5), atol=1e-12)
        assert console.last_dashboard_path.exists()
        pixels = np.asarray(PILImage.open(console.last_dashboard_path).convert('RGB'))
        assert pixels.shape[0] > 1000 and pixels.std() > 20
console.view.value = 'latest'
console.model.value = 'all_inputs'
console.rows.value = 30
console.threshold.value = 0.005
console.run_button.click()
assert console.last_error is None
console.tabs.selected_index = 0
print('PASS: four widget mode/model combinations, holdout replay, threshold/rows, image pixels, final-state restoration.')
"""


def main():
    path = ROOT / "ResearchModelTrainingNews.ipynb"
    notebook = nbformat.read(path, as_version=4)
    original_count = len(notebook.cells)
    notebook.cells.append(nbformat.v4.new_code_cell(SMOKE))
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}}, store_widget_state=True)
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    notebook.cells = notebook.cells[:original_count]
    nbformat.validate(notebook)
    errors = [output for cell in notebook.cells for output in cell.get("outputs", []) if output.output_type == "error"]
    assert not errors
    assert notebook.metadata.get("widgets"), "Executed widget state was not saved."
    nbformat.write(notebook, path)
    report = {"notebook": path.name, "executed_code_cells": sum(c.cell_type == "code" for c in notebook.cells),
              "error_outputs": len(errors), "widget_combinations_tested": 4,
              "holdout_predictions_reproduced": True, "nonblank_dashboard_images": True,
              "final_widget_state": "latest/all_inputs/30 rows/0.5% threshold"}
    (ROOT / "Scripts/project/artifacts/outputs/news_notebook_execution_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
