from pipelines.train_pipeline import train

if __name__ == "__main__":
    train(
        csv_path="data/raw/historico_llamadas.csv",
        target_col="contacto_positivo"
    )