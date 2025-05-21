"""Скрипт, реферующий текст при помощи алгоритма  TextRank"""
import re
import spacy
import numpy as np
import networkx as nx
from datasets import load_dataset
from spacy.language import Language, Doc


def preprocess_text(text: str, nlp: Language) -> Doc:
    # Так как данные расчитаны использование в качестве выборки 
    # для регулировки параметров модели, сами статьи лежат в кавычках
    # от которых необходимо избавиться. Для этого было решено использовать 
    # регулярные выражения, из-за чего пришлось также избавиться от всех 
    # символов новой строки.
    clean_text = re.sub('"', "", (re.search('"(.*)"', re.sub("\n", " ", text))[0]))
    clean_text = nlp(clean_text)
    return clean_text

def get_similarities(text: Doc) -> np.ndarray:
    sentences = list(text.sents)
    sim_mtx = np.zeros((len(sentences), len(sentences)))

    for idx1, sent1 in enumerate(sentences):
        for idx2, sent2 in enumerate(sentences):
            if idx1 != idx2:
                sim_mtx[idx1][idx2] = sent1.similarity(sent2)
    return sim_mtx

def TextRank(text: Doc, sim_mtx: np.ndarray, n_sents=None) -> list:
    if n_sents is None:
        n_sents = 5

    graph = nx.from_numpy_array(sim_mtx)
    scores = nx.pagerank(graph)
    ranked_sents = sorted([(scores[idx], sent.text) for idx, sent in enumerate(text.sents)], reverse=True)
    ans = [sent[1] for sent in ranked_sents[:n_sents]]
    return ans


if __name__=="__main__":
    nlp = spacy.load("ru_core_news_md")
    dataset = load_dataset("RussianNLP/Mixed-Summarization-Dataset", split="test[:40]")
    with open("./results/TextRank_results.txt", "w") as f:
        for articleno, article in enumerate(dataset["text"]):
            clean_text = preprocess_text(article, nlp)
            sim_mtx = get_similarities(clean_text)
            summary = TextRank(clean_text, sim_mtx, n_sents=3)
            print(f'Исходная статья: статья #{articleno+1}')
            print(f'\n{" ".join(summary)}\n\n')
            f.write(f'Исходная статья: статья #{articleno+1}')
            f.write(f'\n{" ".join(summary)}\n\n')

