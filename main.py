import yaml
from easydict import EasyDict

from data_loader import Loader
from trainer import GRUTrainer
import logging
import argparse

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    with open("config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    loader = Loader(config)
    train_dataset = loader.get_dataset()
    test_dataset = loader.get_dataset(evaluate=True)

    trainer = GRUTrainer(config,
                         train_dataset=train_dataset,
                         test_dataset=test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    args = parser.parse_args()
    main(args)
