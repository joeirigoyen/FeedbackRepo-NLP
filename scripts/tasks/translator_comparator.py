import os
import deepl
import numpy as np
import logging as log
from abc import abstractmethod
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate
from nltk.translate.bleu_score import sentence_bleu


load_dotenv()


class Translator:
    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path
        self._lines_to_translate = self.get_lines(data_path.joinpath("ES-EN.txt"))
        self._reference_lines = self.get_lines(data_path.joinpath("EN-ES.txt"))

    @staticmethod
    def get_lines(filepath: Path, limit: int = -1) -> list[str]:
        log.debug(f"Getting first {limit} text lines from {filepath}")
        with open(filepath, 'r') as f:
            file_lines = []
            for index, line in enumerate(f):
                file_lines.append(line)
                if index + 1 == limit:
                    log.debug(f"Finished retrieving text lines. Got {len(file_lines)} lines\n")
                    break
            else:
                log.debug(f"Finished retrieving text lines. Got {len(file_lines)} lines\n")
        return file_lines

    def get_avg_bleu(self, lines) -> float | None:
        if len(lines) > 0:
            log.debug("Getting average bleu score")
            bleu_scores = []
            for ref, res in zip(self._reference_lines, lines):
                bleu_scores.append(sentence_bleu([ref], res))
            result = np.average(np.asarray(bleu_scores))
            log.debug(f"Bleu score: {result}")
            return result
        else:
            log.warn("Cannot get average. Not enough result lines")
            return None
    
    @staticmethod
    def _write_translations(filepath: Path, lines: list[str]) -> None:
        log.debug(f"Writing translation to {filepath}")
        with open(filepath, 'w') as f:
            f.writelines(lines)
    
    @abstractmethod
    def translate(self, write_to_file: bool = False) -> list[str]:
        pass


class GoogleCloudTranslator(Translator):
    def __init__(self, data_path: Path, target_language: str = "es") -> None:
        super().__init__(data_path)
        # Set Google Cloud client
        self._google_api_path = Path("./cloudapikey.json")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(self._google_api_path)
        self._client = translate.Client()
        # Set target language
        self._target_language = target_language
        # Set writing path
        self._google_cloud_filepath = Path("./scripts/testing/googleCloudTranslation.txt")

    def translate(self, write_to_file: bool = False) -> list[str]:
        log.debug("Translating lines with Google Cloud...")
        lines = [self._client.translate(line, target_language=self._target_language)["translatedText"] + "\n" for line in self._lines_to_translate]
        if write_to_file:
            self._write_translations(self._google_cloud_filepath, lines)
        return lines


class DeepLTranslator(Translator):
    def __init__(self, data_path: Path, target_language: str = "es") -> None:
        super().__init__(data_path)
        # Set DeepL client
        self._deepl_api_key = os.getenv('DEEPL_API_KEY')
        self._client = deepl.Translator(self._deepl_api_key)
        # Set target language
        self._target_language = target_language
        # Set writing path
        self._deepl_filepath = Path("./scripts/testing/deepLTranslation.txt")

    def translate(self, write_to_file: bool = False) -> list[str]:
        log.debug("Translating lines with Google Cloud...")
        lines = [self._client.translate_text(line, target_lang=self._target_language).text for line in self._lines_to_translate] 
        if write_to_file:
            self._write_translations(self._deepl_filepath, lines)
        return lines


def test_comparison() -> None:
    # Set paths
    data_path = Path("./data")
    # Create translator instances
    google_cloud_translator = GoogleCloudTranslator(data_path)
    deepl_translator = DeepLTranslator(data_path)
    # Get translations and bleu scores
    google_cloud_translations = google_cloud_translator.translate(write_to_file=True)
    google_cloud_bleu = google_cloud_translator.get_avg_bleu(google_cloud_translations)
    print(f"Google Cloud BLEU score: {google_cloud_bleu}")
    deepl_translations = deepl_translator.translate(write_to_file=True)
    deepl_bleu = deepl_translator.get_avg_bleu(deepl_translations)
    print(f"DeepL SE BLEU score: {deepl_bleu}")
