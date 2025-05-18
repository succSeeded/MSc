import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.cluster import KMeans, OPTICS
from gensim.models import KeyedVectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    similarity = a.dot(b) / np.linalg.norm(a) / np.linalg.norm(b)
    return 1.0 - similarity


if __name__ == "__main__":
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "model_path", action="store", help="Path to the text to be parsed.", nargs='?', const=None
    )
    args = vars(parser.parse_args())
    
    if args.get("textpath") is None:
        model_path = "~/Downloads/ruwikiruscorpora_upos_skipgram_300_2_2019/model.bin"
    else:
        model_path = args["textpath"]

    print("Loading the pretrained embeddings...")
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Embeddings are loaded!")

    keys_to_cluster = [
        # subset 1
        ['железный_ADJ', 'серебряный_ADJ', 'стальной_ADJ', 'твердый_ADJ',
         'добрый_ADJ', 'твердый_ADJ', 'выносливый_ADJ', 'терпеливый_ADJ',
         'алмазный_ADJ', 'алюминиевый_ADJ', 'верный_ADJ', 'пластиковый_ADJ',
         'жестокий_ADJ', 'отважный_ADJ', 'высокомерный_ADJ', 'надменный_ADJ',
         'деревянный_ADJ', 'золотой_ADJ', 'кожаный_ADJ', 'медный_ADJ', 
         'бронзовый_ADJ', 'внимательный_ADJ', 'раздражительный_ADJ',
         'хитрый_ADJ', 'мудрый_ADJ'],
        # subset 2
        ['киви_NOUN', 'смородина_NOUN', 'лиса_NOUN', 'лисичка_NOUN', 
         'ара_NOUN', 'клубника_NOUN', 'земляника_NOUN', 'малина_NOUN', 
         'черника_NOUN', 'ежевика_NOUN', 'огурец_NOUN', 'облепиха_NOUN', 
         'перец_NOUN', 'яблоко_NOUN', 'черешня_NOUN', 'крыжовник_NOUN', 
         'мандарин_NOUN', 'мандаринка_NOUN', 'ворон_NOUN', 'сорока_NOUN',
         'беркут_NOUN', 'орел_NOUN', 'сокол_NOUN', 'страус_NOUN', 'эму_NOUN',
         'голубь_NOUN', 'трясогузка_NOUN', 'казуар_NOUN'],
        # subset 3
        ['руль_NOUN', 'штурвал_NOUN', 'ручка_NOUN', 'дверь_NOUN', 
         'кабина_NOUN', 'кокпит_NOUN', 'шасси_NOUN', 'трансмиссия_NOUN',
         'фара_NOUN', 'элерон_NOUN', 'тормоз_NOUN', 'крыло_NOUN', 
         'хвост_NOUN', 'тяга_NOUN', 'зеркало_NOUN', 'колесо_NOUN',
         'шина_NOUN', 'покрышка_NOUN', 'кузов_NOUN', 'оперение_NOUN',
         'крыло_NOUN', 'планер_NOUN', 'машина_NOUN', 'капот_NOUN', 
         'бак_NOUN', 'закрылок_NOUN', 'тангаж_NOUN']
    ]

    for n_subset in range(3):

        embeddings = np.array([w2v_model.get_vector(key) for key in keys_to_cluster[n_subset][:]])

        # В данной задаче было решено использовать алгоритм K-Means, но
        # так как этот метод расчитан на использование только евклидового 
        # расстояния, для класстеризации с использованием косинусного растояния
        # был использован алгоритм кластеризации OPTICS (при чем стоит заметить
        # что были использованы как косинусное так и евклидово растояния).
        optic_cosine = OPTICS(metric=cosine_similarity).fit(embeddings)
        optic_euclid = OPTICS(metric="euclidean").fit(embeddings)
        kmeans = KMeans(n_clusters=2, random_state=42).fit(embeddings)

        # Результаты работы алгоритмов кластеризации собираются в таблицу
        # и сохраняются в виде .csv файла
        label_dict = {key: [kmeans.labels_[idx], optic_cosine.labels_[idx], optic_euclid.labels_[idx]] for idx, key in enumerate(keys_to_cluster[n_subset])}
        df = pd.DataFrame.from_dict(label_dict, orient="index", columns=["K-Means", "OPTICS (косинусное расcтояние)", "OPTICS (евклидово раcстояние)"])
        df.to_csv(f"./subset{n_subset}_labels.csv")
        print(df)
