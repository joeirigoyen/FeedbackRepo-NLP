from pathlib import Path
from utils.tools import get_text_lines
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_line_scores(line: str) -> dict:
    result = SentimentIntensityAnalyzer()
    return result.polarity_scores(line)


def assume_polarity(scores: dict) -> str:
    return "POSITIVE" if scores['pos'] > scores['neg'] else "NEGATIVE" if scores['neg'] > scores['pos'] else "NEUTRAL"


def show_text_score(filepath: str | Path, limit: int = 0):
    text_lines = get_text_lines(filepath)
    for index, line in enumerate(text_lines):
        if 0 < limit <= index:
            break
        scores = get_line_scores(line)
        print(
            f"\"{line[:10]}\"... | Negative: {scores['neg']} | Positive: {scores['pos']} | Polarity: {assume_polarity(scores)}")


def test_analysis(num_lines: str = "0"):
    datapath = Path.cwd().joinpath('data', 'reviews.txt')
    show_text_score(datapath, limit=int(num_lines))
