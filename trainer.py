import os
import numpy as np

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from torch.autograd import Variable

from tqdm import tqdm, trange

from metrics import compute_metrics
from model import GRUModel

import logging

logger = logging.getLogger(__name__)


def to_tensor(x):
    return x.detach().cpu().numpy()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GRUTrainer(object):
    def __init__(self,
                 config,
                 train_dataset=None,
                 test_dataset=None,
                 ):

        hidden_dim = 64
        layer_dim = 1
        output_dim = 1

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.model = GRUModel(input_size=44,
                              hidden_size=hidden_dim,
                              num_layers=layer_dim,
                              seq_length=self.config.window_size,
                              num_classes=output_dim)
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.learning_rate))

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.train_batch_size)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.config.train_batch_size}")

        train_iterator = trange(int(self.config.num_train_epochs), desc="Epoch")
        set_seed(self.config.seed)
        tr_loss = 0.0
        global_step = 0

        self.model.zero_grad()

        for _ in train_iterator:
            epoch_iterator = tqdm(train_loader, desc="Iteration")
            for batch_idx, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                x = batch[0]
                labels = batch[1]

                outputs = self.model(x)
                loss = self.criterion(outputs, labels)

                tr_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                global_step += 1

                if global_step % self.config.save_steps == 0:
                    if self.config.save_model:
                        self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self):
        eval_dataloader = DataLoader(self.test_dataset, batch_size=self.config.eval_batch_size)

        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(self.test_dataset)}")
        logger.info(f"  Batch size = {self.config.eval_batch_size}")

        eval_loss = 0.0
        nb_eval_steps = 0
        answers = []
        true_labels = []

        for idx, batch in enumerate(tqdm(eval_dataloader)):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                x = batch[0]
                labels = batch[1]

                outputs = self.model(x)
                tmp_eval_loss = self.criterion(outputs, labels)
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            answers.extend(to_tensor(outputs))
            true_labels.extend(to_tensor(labels))

        eval_loss = eval_loss / nb_eval_steps

        results = compute_metrics(true=np.array(true_labels), pred=np.array(answers))

        logger.info("***** Eval results *****")
        logger.info(f"  Eval MSE loss = {eval_loss}")
        for key, value in results.items():
            logger.info(f"  {key} = {value}")

        if self.config.save_predictions:
            if not os.path.exists(self.config.prediction_dir):
                os.makedirs(self.config.prediction_dir)

            with open(os.path.join(self.config.prediction_dir, f"predictions.txt"), "w") as writer:
                logger.info("***** Predictions *****")
                for key, value in results.items():
                    writer.write(f"{key}: {value}\n")
                for true_label, pred_label in zip(true_labels, answers):
                    writer.write(f"{true_label[0]},{pred_label[0]}\n")

        return results

    def save_model(self):
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)

        torch.save(self.config, os.path.join(self.config.model_dir, "config.pt"))
        torch.save(self.model.state_dict(), os.path.join(self.config.model_dir, "model.pt"))
        logger.info(f"Model saved in {self.config.model_dir}")

    def load_model(self):
        if not os.path.exists(self.config.model_dir):
            raise Exception("Model not found ! ")

        try:
            self.config = torch.load(os.path.join(self.config.model_dir, "config.pt"))
            logger.info(f"***** Config loaded from {self.config.model_dir} *****")

            self.model.load_state_dict(torch.load(os.path.join(self.config.model_dir, "model.pt")))
            self.model.to(self.device)
            logger.info(f"***** Model loaded from {self.config.model_dir} *****")
        except:
            raise Exception("Some Model files might be missing !")
