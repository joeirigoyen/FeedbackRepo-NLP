# Note: to use flair library you need to have a version of python < 3.10
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from flair.data import Corpus
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings


DOWNSAMPLE_PERC = 0.3
LEARNING_RATE = 0.01
BATCH_SIZE = 16
MAX_EPOCHS = 30


class NERTrainer:
    
    def __init__(self, data_path: Path, col_format: dict[int, str], percent_of_dataset_to_train: float = 0.2) -> None:
        # Set data paths
        self._data_path = data_path
        self._train_path = data_path.joinpath("train.txt")
        self._test_path = data_path.joinpath("test.txt")
        self._dev_path = data_path.joinpath("dev.txt")
        # Set corpus
        self._corpus = ColumnCorpus(
            self._data_path, 
            col_format, 
            train_file=self._train_path,
            test_file=self._test_path,
            dev_file=self._dev_path
            )
        self._corpus.downsample(DOWNSAMPLE_PERC)
        # Set embeddings
        self._embeddings = StackedEmbeddings(embeddings=[WordEmbeddings('glove')])
        self._tag_dictionary = None
        # Set tagger
        self._tagger = None
        self._refresh_settings()

    def _refresh_data(self):
        self._train_path = self._data_path.joinpath("train.txt")
        self._test_path = self._data_path.joinpath("test.txt")
        self._dev_path = self._data_path.joinpath("dev.txt")
    
    def _refresh_settings(self):
        self._tag_dictionary = self._corpus.make_label_dictionary(label_type='ner')
        self._tagger = SequenceTagger(
            hidden_size=256, 
            embeddings=self._embeddings, 
            tag_dictionary=self._tag_dictionary, 
            tag_type='ner', 
            use_crf=True
            )

    @property
    def corpus(self) -> Corpus:
        return self._corpus
    
    @corpus.setter
    def corpus(self, data_path: Path, col_format: dict[str, int]) -> None:
        self._data_path = data_path
        self._refresh_data()
        new_corpus = ColumnCorpus(
            data_path, 
            col_format, 
            train_file=self._train_path,
            test_file=self._test_path,
            dev_file=self._dev_path
            )
        new_corpus.downsample(0.3)
        self._corpus = new_corpus
        self._refresh_settings()

    def train_corpus(self, log_path: Path) -> None:
        trainer = ModelTrainer(self._tagger, self._corpus)
        trainer.train(
            log_path,
            learning_rate=LEARNING_RATE,
            mini_batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS
            )

    def get_history(log_path: Path) -> pd.DataFrame:
        history = pd.read_csv(log_path, delimitersep='\t')
        return history

    @staticmethod
    def plot_history(history: pd.DataFrame, plot_path: Path) -> None:
        figure = plt.figure()
        figure.suptitle('Loss')
        plt.plot(history["EPOCH"], history["TRAIN_LOSS"], label='Train Loss')
        plt.plot(history["EPOCH"], history["DEV_LOSS"], color='orange', label='Dev Loss')
        plt.legend(loc='upper right')
        plt.savefig(plot_path)
        print(f"Plot saved in path: {plot_path}")


def test_ner_trainer():
    data_path = Path("./data/ner_data").absolute()
    log_path = Path("./scripts/testing/flairNERLoss.tsv").absolute()
    print(data_path)
    print(log_path)
    col_format = {0: 'text', 1: 'ner'}
    ner_trainer = NERTrainer(data_path, col_format)
    ner_trainer.train_corpus(log_path)
    trainer_history = ner_trainer.get_history(log_path)
    ner_trainer.plot_history(trainer_history)
