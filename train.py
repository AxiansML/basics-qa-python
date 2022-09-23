import os
from argparse import ArgumentParser

import mlflow
from pytorch_lightning import Trainer

from model import DummyDataModule, LinearRegressor


def main():
    parser = ArgumentParser(description="")

    parser.add_argument("--resume_checkpoint", type=str)
    parser.add_argument("--resume_run_id", type=str)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--accelerator", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--test_ratio", type=float)
    parser.add_argument("--random_seed")
    parser.add_argument("--dataset", type=str)

    mlflow.pytorch.autolog()

    args = parser.parse_args()
    dict_args = vars(args)

    for arg in dict_args:
        if dict_args[arg] == "None":
            dict_args[arg] = None

    # If resuming from a previously training run, download the checkpoint file and
    # intantiate the model from it, but may have different hyperparameters
    if dict_args["resume_checkpoint"]:
        mlflow_client = mlflow.tracking.MlflowClient()
        mlflow_client.download_artifacts(
            dict_args["resume_run_id"],
            dict_args["resume_checkpoint"],
            dst_path=os.getcwd(),
        )
        ckpt_path = os.path.join(os.getcwd(), dict_args["resume_checkpoint"])
        model = LinearRegressor.load_from_checkpoint(ckpt_path)
        model.set_hyperparams(dict_args)
    else:
        model = LinearRegressor(**dict_args)

    dm = DummyDataModule(**dict_args)
    dm.setup(stage="fit")

    model = LinearRegressor(**dict_args)

    train_params = model.get_trainer_params()
    trainer = Trainer(**train_params)
    trainer.fit(model, dm)
    if dict_args["test_ratio"] > 0:
        trainer.test(datamodule=dm)
