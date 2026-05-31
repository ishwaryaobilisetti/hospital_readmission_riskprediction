import json
import os
import uuid


def _cell(cell_type, source):
    cell = {
        "cell_type": cell_type,
        "metadata": {
            "id": uuid.uuid4().hex[:8],
            "language": cell_type,
        },
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _write_notebook(path, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1)


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    notebooks_dir = os.path.join(base_dir, "notebooks")
    os.makedirs(notebooks_dir, exist_ok=True)

    eda_cells = [
        _cell("markdown", [
            "# Exploratory Data Analysis\n",
            "\n",
            "Quick notebook for ingestion, target binarization, and leakage filtering.\n",
        ]),
        _cell("code", [
            "import os\n",
            "import pandas as pd\n",
            "data_path = os.path.join('data', 'diabetic_data.csv')\n",
            "df = pd.read_csv(data_path)\n",
            "df.shape\n",
        ]),
        _cell("code", [
            "df['target'] = (df['readmitted'] == '<30').astype(int)\n",
            "df['target'].value_counts(normalize=True)\n",
        ]),
        _cell("code", [
            "leakage_ids = [11, 13, 14, 19, 20, 21]\n",
            "df_clean = df[~df['discharge_disposition_id'].isin(leakage_ids)].copy()\n",
            "df_clean.shape\n",
        ]),
    ]

    modelling_cells = [
        _cell("markdown", [
            "# Modeling, Calibration, and Responsible AI Audit\n",
            "\n",
            "This notebook runs the end-to-end training script and displays the generated outputs.\n",
        ]),
        _cell("code", [
            "from src.train import main\n",
            "main()\n",
        ]),
        _cell("markdown", [
            "## Outputs\n",
            "\n",
            "The training script writes `pipeline.pkl`, calibration and SHAP plots, and `outputs/bias_audit.csv`.\n",
        ]),
    ]

    _write_notebook(os.path.join(notebooks_dir, "eda.ipynb"), eda_cells)
    _write_notebook(os.path.join(notebooks_dir, "modelling.ipynb"), modelling_cells)
    print(f"Notebook files created in {notebooks_dir}")


if __name__ == "__main__":
    main()