"""Run the demo server: load reference_cases.csv, init matcher, and start uvicorn."""
from __future__ import annotations

import pandas as pd
import uvicorn
from api.predict import initialize_demo, app


def main():
    df = pd.read_csv("data/reference_cases.csv")
    numeric = ["age", "erythema"]
    categorical = ["sex", "biologic"]
    initialize_demo(df, numeric, categorical)
    print("Demo matcher initialized with", len(df), "cases")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == '__main__':
    main()
