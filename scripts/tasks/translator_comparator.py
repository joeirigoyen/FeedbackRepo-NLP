import os
import deepl
import numpy as np
import logging as log
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate
from nltk.translate.bleu_score import sentence_bleu


load_dotenv()


def get_lines(filepath: str | Path, limit: int) -> list[str]:
    log.debug(f"Getting first {limit} text lines from {filepath}")
    with open(filepath, 'r') as f:
        file_lines = []
        for index, line in enumerate(f):
            file_lines.append(line)
            if index + 1 == limit:
                break
    log.debug(f"Finished retrieving text lines. Got {len(file_lines)} lines\n")
    return file_lines


def get_avg_bleu(ref_lines: list[str], res_lines: list[str]) -> float:
    bleu_scores = []
    log.debug("Getting average bleu score")
    for ref_line, res_line in zip(ref_lines, res_lines):
        bleu_scores.append(sentence_bleu([ref_line], res_line))
    avg = np.average(np.asarray(bleu_scores))
    log.debug(f"Got score of {avg}")
    return avg


def get_cloud_lines(client, lines: list[str], target_language: str = "es") -> list[str]:
    translations = []
    for line in lines:
        translations.append(client.translate(line, target_language=target_language)["translatedText"] + "\n")
    return translations


def get_deepl_lines(client: deepl.translator.Translator, lines: list[str], target_language: str = "es") -> list[str]:
    translations = []
    for line in lines:
        translations.append(client.translate_text(line, target_lang=target_language).text)
    return translations


def test_comparison(limit: int = 100) -> None:
    # Set paths
    data_path = Path("./data")
    source_path = data_path.joinpath("ES-EN.txt")
    reference_path = data_path.joinpath("EN-ES.txt")
    # Set clients from both APIs
    google_api_path = Path("./cloudapikey.json")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(google_api_path)
    cloud_client = translate.Client()
    deepl_api_key = os.getenv('DEEPL_API_KEY')
    deepl_client = deepl.Translator(deepl_api_key)
    # Get lines from text
    lines_to_translate = get_lines(source_path, limit)
    actual_translated_lines = get_lines(reference_path, limit)
    # Translate lines
    cloud_translated_lines = get_cloud_lines(cloud_client, lines_to_translate)
    deepl_translated_lines = get_deepl_lines(deepl_client, lines_to_translate)
    # Write translations to files
    google_cloud_file = Path("./scripts/testing/googleCloudTranslation.txt")
    deepl_file = Path("./scripts/testing/deepLTranslation.txt")
    log.debug(f"Writing Google Cloud translation to {google_cloud_file}")
    log.debug(f"Writing DeepL translation to {deepl_file}\n")
    with open(google_cloud_file, 'w') as f:
        f.writelines(cloud_translated_lines)
    with open(deepl_file, 'w') as f:
        f.writelines(deepl_translated_lines)
    # Compare translations
    cloud_bleu_score = get_avg_bleu(actual_translated_lines, cloud_translated_lines)
    deepl_bleu_score = get_avg_bleu(actual_translated_lines, deepl_translated_lines)
    print(f"Google Cloud BLEU score: {cloud_bleu_score}")
    print(f"DeepL SE BLEU score: {deepl_bleu_score}")
