# Note: to use flair library you need to have a version of python < 3.10
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from flair.data import Corpus
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings

# LOVE that you are using type hints here! :) 

class NERTrainer:
    
    def __init__(self, data_path: Path, col_format: dict[int, str], percent_of_dataset_to_train: float = 0.2) -> None:
        # Set data paths
        # Use a single underscore for private vars.
        # https://stackoverflow.com/questions/159720/what-is-the-naming-convention-in-python-for-variable-and-function
        # https://towardsdatascience.com/whats-the-meaning-of-single-and-double-underscores-in-python-3d27d57d6bd1
        self.__data_path = data_path
        self.__train_path = data_path.joinpath("train.txt")
        self.__test_path = data_path.joinpath("test.txt")
        self.__dev_path = data_path.joinpath("dev.txt")
        # Set corpus
        self.__corpus = ColumnCorpus(
            self.__data_path, 
            col_format, 
            train_file=self.__train_path,
            test_file=self.__test_path,
            dev_file=self.__dev_path
            )
        self.__corpus.downsample(0.3)
        # Set embeddings
        self.__embeddings = StackedEmbeddings(embeddings=[WordEmbeddings('glove')])
        self.__tag_dictionary = None
        # Set tagger
        self.__tagger = None
        self.__refresh_settings()

    def __refresh_data(self):
        self.__train_path = self.__data_path.joinpath("train.txt")
        self.__test_path = self.__data_path.joinpath("test.txt")
        self.__dev_path = self.__data_path.joinpath("dev.txt")
    
    def __refresh_settings(self):
        self.__tag_dictionary = self.__corpus.make_label_dictionary(label_type='ner')
        self.__tagger = SequenceTagger(
            hidden_size=256, 
            embeddings=self.__embeddings, 
            tag_dictionary=self.__tag_dictionary, 
            tag_type='ner', 
            use_crf=True
            )

    @property
    def corpus(self) -> Corpus:
        return self.__corpus
    
    @corpus.setter
    def corpus(self, data_path: Path, col_format: dict[str, int]) -> None:
        self.__data_path = data_path
        self.__refresh_data()
        new_corpus = ColumnCorpus(
            data_path, 
            col_format, 
            train_file=self.__train_path,
            test_file=self.__test_path,
            dev_file=self.__dev_path
            )
        new_corpus.downsample(0.3)
        self.__corpus = new_corpus
        self.__refresh_settings()

    def train_corpus(self, log_path: Path) -> None:
        trainer = ModelTrainer(self.__tagger, self.__corpus)
        trainer.train(
            log_path,
            learning_rate=0.01, # nit: some reviewers will have you put all constants at the top as named constants. not a big deal for this HW but worth considering. 
            mini_batch_size=16,
            max_epochs=30
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


# you don’t need to change it for this homework, but in general, best practices are to have a structure like this: https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure
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


test_ner_trainer()
