import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, X, y):
        """
        :param X: input features
        :param y: target output values
        """
        self.X = X
        self.y = y

    def __len__(self):
        """
        :return: returns the number of datapoints in the dataframe
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns the input features and the targets of the specified index
        :param item: Index of sample
        :return: Returns the dictionary of input features and targets
        """

        data_dict = {"X": torch.Tensor([self.X[idx]]), "y": torch.Tensor([self.y[idx]])}

        return data_dict


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(DummyDataModule, self).__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.args = kwargs

    def setup(self, stage=None):
        """
        Loads the data, parse it and split it into train, test, validation
        :param stage: Stage - training or testing
        """

        df = pd.read_json(self.args["dataset"])

        if self.args["num_samples"] > 0:
            df = df.sample(
                self.args["num_samples"], random_state=self.args["random_seed"]
            )

        ratio1 = self.args["val_ratio"] + self.args["test_ratio"]
        ratio2 = 0
        if ratio1 > 0:
            df_train, df_val = train_test_split(
                df, test_size=ratio1, random_state=self.args["random_seed"]
            )
            self.df_train = df_train
            self.df_val = df_val
            ratio2 = self.args["test_ratio"] / ratio1
        else:
            self.df_train = df

        if ratio2 > 0:
            df_val, df_test = train_test_split(
                df_val, test_size=ratio2, random_state=self.args["random_seed"]
            )
            self.df_val = df_val
            self.df_test = df_test

    def create_data_loader(self, df, batch_size):
        """
        Generic data loader function
        :param df: Input dataframe
        :param batch_size: Batch size for training
        :return: Returns the constructed dataloader
        """
        ds = DummyDataset(
            df.X.to_numpy(),
            df.y.to_numpy(),
        )

        return DataLoader(ds, batch_size=batch_size, num_workers=os.cpu_count())

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            self.df_train, self.args["batch_size"]
        )

        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.args["batch_size"]
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.args["batch_size"]
        )
        return self.test_data_loader


class LinearRegressor(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super(LinearRegressor, self).__init__()

        self.save_hyperparameters()
        self.args = kwargs

        self.linear_reg = torch.nn.Linear(1, 1)
        self.loss = torch.nn.MSELoss()

    def set_hyperparams(self, dict_args):
        for k, v in dict_args.items():
            self.args[k] = v

    def get_trainer_params(self):
        """Define callbacks and put some of the train hyperparameters into a dictionary.

        Returns:
            dict: Dictionary with some of the train hyperparameters and callbacks.
        """
        callbacks = [
            EarlyStopping(monitor="val_score", mode="max", verbose=True),
            ModelCheckpoint(
                dirpath=os.path.join(os.getcwd(), "checkpoints"),
                save_top_k=1,
                verbose=True,
                monitor="val_score",
                mode="max",
            ),
            LearningRateMonitor(),
        ]
        train_params = dict(
            num_sanity_val_steps=self.args["num_sanity_val_steps"],
            accumulate_grad_batches=self.args["gradient_accumulation_steps"],
            max_epochs=self.args["max_epochs"],
            gpus=self.args["gpus"],
            strategy="ddp",
            precision=32,
            gradient_clip_val=self.args["max_grad_norm"],
            val_check_interval=self.args["val_check_interval"],
            callbacks=callbacks,
        )

        return train_params

    def forward(self, X):
        """
        :param X: Input data
        :return: prediction
        """
        prediction = self.linear_reg(X)

        return prediction

    def training_step(self, batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch
        :param train_batch Batch data
        :return: output - Training loss
        """
        inputs = batch["X"]
        targets = batch["y"]

        predictions = self(inputs)
        train_loss = self.loss(predictions, targets)

        with torch.no_grad():
            train_score = r2_score(targets, predictions)

        self.log("train_loss", train_loss)
        self.log("train_score", train_score)
        self.log("weight", self.get_parameter("linear_reg.weight"))
        self.log("bias", self.get_parameter("linear_reg.bias"))

        return {"loss": train_loss, "score": torch.Tensor([train_score])}

    def validation_step(self, batch, batch_idx):
        """
        Performs validation of data in batches
        :param val_batch: Batch data
        :return: output - valid step loss
        """
        with torch.no_grad():
            inputs = batch["X"]
            targets = batch["y"]

            predictions = self(inputs)
            val_loss = self.loss(predictions, targets)
            val_score = r2_score(targets, predictions)

        return {"val_loss": val_loss, "val_score": torch.Tensor([val_score])}

    def test_step(self, batch, batch_idx):
        """
        Performs test and computes the accuracy of the model
        :param test_batch: Batch data
        :return: output - Testing accuracy
        """
        with torch.no_grad():
            inputs = batch["X"]
            targets = batch["y"]

            predictions = self(inputs)
            test_score = r2_score(targets, predictions)

        return {"test_score": torch.Tensor([test_score])}

    def training_epoch_end(self, outputs):
        """Log the average training loss, at the end of a training epoch.

        Args:
            outputs (dict): Dictionary containing all the individual step losses.
        """
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_train_score = torch.stack([x["score"] for x in outputs]).mean()

        self.log("epoch_train_loss", avg_train_loss, sync_dist=True)
        self.log("epoch_train_score", avg_train_score, sync_dist=True)

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_score = torch.stack([x["val_score"] for x in outputs]).mean()
        self.log("epoch_val_loss", avg_val_loss, sync_dist=True)
        self.log("epoch_val_score", avg_val_score, sync_dist=True)

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        :param outputs: outputs after every epoch end
        :return: output - average test loss
        """
        avg_test_score = torch.stack([x["test_score"] for x in outputs]).mean()
        self.log("test_score", avg_test_score, sync_dist=True)

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler
        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.args["lr"])

        return [self.optimizer]
