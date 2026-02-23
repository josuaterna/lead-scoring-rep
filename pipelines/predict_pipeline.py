import pandas as pd


def predict(model, csv_path: str, output_path: str):

    df = pd.read_csv(csv_path, sep=";")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    proba = model.predict_proba(df)[:, 1]

    df["lead_score"] = proba

    df.to_csv(output_path, index=False)
