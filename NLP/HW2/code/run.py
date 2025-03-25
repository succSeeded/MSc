import spacy
from collections import defaultdict, Counter
from typing import Iterable, Callable
from spacy.symbols import amod, ADJ, NOUN
from numpy import log2, power, sqrt


collocation_types = [
        ("NOUN", "NOUN"),
        ("ADJ", "NOUN"),
        ("VERB", "NOUN"),
        ]


# Note that Dice accepts N only to have consistent signature with the other
# metrics. Although this defeats the whole purpose of this metric, I could not 
# think of a better way to do this.
collocation_metrics = {
        "Dice": lambda f_ab, f_a, f_b, N: 2.0 * f_ab / (f_a + f_b),
        "MI": lambda f_ab, f_a, f_b, N: log2(f_ab * N / f_a / f_b),
        "MI3": lambda f_ab, f_a, f_b, N: log2(power(f_ab, 3) * N / f_a /f_b),
        "T-score": lambda f_ab, f_a, f_b, N: (f_ab - f_a * f_b / N) / sqrt(f_ab),
        }


def is_noun_noun_collocation(token):
    return (token.pos == NOUN) and (token.dep == amod) and (token.head.pos == NOUN) and (token.text != "-") and (token.head.text != "-")


def is_adj_noun_collocation(token):
    return (token.pos == ADJ) and (token.head.pos == NOUN) and (token.text != "-") and (token.head.text != "-")



if __name__=="__main__":
    
    wordforms = []
    lemmi = []
    wordform_collocations = defaultdict()
    lemmi_collocations = defaultdict()

    nlp = spacy.load("ru_core_news_sm")
    with open("./dataset.txt", "r", encoding="utf8") as file:
        lines_read = 0
        for line in file:
            
            line = line.strip()
            tokenized_line = nlp(line)
            
            for token in tokenized_line:
                
                if not token.is_stop and not token.is_punct: 
                    
                    wordforms += [token.text]
                    
                    lemmi += [token.lemma_]

                    if is_noun_noun_collocation(token):
                        collocation_wordform = f"{token.text} {token.head.text}"
                        collocation_lemma = f"{token.lemma_} {token.head.lemma_}"
                        if collocation_wordform in wordform_collocations:
                            wordform_collocations[collocation_wordform]["n_uses"] += 1
                        else:
                            wordform_collocations[collocation_wordform] = {"n_uses": 1, "type": "N N"}
                        if collocation_lemma in lemmi_collocations:
                            lemmi_collocations[collocation_lemma]["n_uses"] += 1
                        else:
                            lemmi_collocations[collocation_lemma] = {"n_uses": 1, "type": "N N"}
                    
                    if is_adj_noun_collocation(token):
                        collocation_wordform = f"{token.text} {token.head.text}"
                        collocation_lemma = f"{token.lemma_} {token.head.lemma_}"
                        if collocation_wordform in wordform_collocations:
                            wordform_collocations[collocation_wordform]["n_uses"] += 1
                        else:
                            wordform_collocations[collocation_wordform] = {"n_uses": 1, "type": "A N"}
                        if collocation_lemma in lemmi_collocations:
                            lemmi_collocations[collocation_lemma]["n_uses"] += 1
                        else:
                            lemmi_collocations[collocation_lemma] = {"n_uses": 1, "type": "A N"}
    
    lemmi_counts = Counter(lemmi)
    wordform_counts = Counter(wordforms)

    def evaluate_metrics(collocation_candidates: dict, frequencies: dict, metrics: dict[str, Callable]=None) -> dict:
        """
        Evaluate provided metrics (or mutual information metric if they are not provided) for a given set of co-occuring word pairs and wordform/lemma frequencies.
        """
        N = sum(frequencies.values())
        for candidate in collocation_candidates:
            first_member, second_member = candidate.split(" ", maxsplit=2)
            f_ab = collocation_candidates[candidate]["n_uses"]
            f_a = frequencies[first_member]
            f_b = frequencies[second_member]

            # metrics are not evaluated for defective collocations
            if f_a == 0 or f_b == 0:
                print(f"Warning: failed to process {candidate}")
                continue

            if metrics is None:
                collocation_candidates[candidate]["MI"] = log2(f_ab * N / f_a / f_b)
            else:
                for metric in metrics:
                    collocation_candidates[candidate][metric] = metrics[metric](f_ab, f_a, f_b, N)
        return collocation_candidates

    lemmi_collocations = evaluate_metrics(lemmi_collocations, lemmi_counts, collocation_metrics)
    wordform_collocations = evaluate_metrics(wordform_collocations, wordform_counts, collocation_metrics)

    def get_key(x:tuple, metric:str):
        if metric in x[1]:
            return x[1][metric]
        else:
            return -1e-6

    with open("./result.txt", "w", encoding="utf8") as file:

        for metric_name in collocation_metrics:
            out = f"топ-20 словосочетаний(по леммам, метрика {metric_name}):\n\n"
            file.write(out)
            for collocation in sorted(lemmi_collocations.items(), key = lambda x: get_key(x, metric_name), reverse=True)[:20]:
                out = f"{collocation[0]}:{collocation[1][metric_name]:0.6f}\n"
                file.write(out)

            out = "=====================================================\n\n"
            file.write(out)

            out = f"топ-20 словосочетаний(по словоформам, метрика {metric_name}):\n\n"
            file.write(out)
            for collocation in sorted(wordform_collocations.items(), key = lambda x: get_key(x, metric_name), reverse=True)[:20]:
                out = f"{collocation[0]}:{collocation[1][metric_name]:0.6f}\n"
                file.write(out)

            out = "=====================================================\n\n"
            file.write(out)

