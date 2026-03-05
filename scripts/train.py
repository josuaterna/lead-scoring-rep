#from pipelines.train_pipeline import train
from pipelines.train_pipeline import train_big_data
def train(csv_path=None, exp_name=None, model_name=None, target_col=None):
    train_big_data(csv_path, exp_name, model_name, target_col)
# if __name__ == "__main__":
#     #train(csv_path=None, exp_name=None, model_name=None, target_col=None)
#     train(csv_path=None, exp_name=None, model_name=None, target_col=None)