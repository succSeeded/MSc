"""
Use python run.py -h in order to look up information about arguments.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pymorphy2 import MorphAnalyzer
from pandas import DataFrame
from numpy import zeros


def get_sents(infile: str):
    """
    Get word forms from a corpus at `infile`.
    """
    with open(infile, "r", encoding="utf8") as fin:
        answer, curr_sent = [], []
        for line in fin:
            line = line.strip()
            if line == "":
                if len(curr_sent) > 0:
                    answer.append(curr_sent)
                curr_sent = []
                continue
            splitted = line.split("\t")
            if len(splitted) == 10:
                word = splitted[1]
                lemma = splitted[2]
                pos = splitted[3]
                tags = splitted[4:]
            elif len(splitted) == 9:
                word = splitted[1]
                pos = splitted[2]
                tags = splitted[3:]
                lemma = None
            else:
                raise ValueError(
                    f"Each line should have 9 or 10 columns. Got {len(splitted)}"
                )
            if tags[1] != "_":
                tags = dict(elem.split("=") for elem in tags[1].split("|"))
            else:
                tags = dict()
            curr_sent.append([word, pos, tags, lemma])
    if len(curr_sent) > 0:
        answer.append(curr_sent)
    return answer


def get_cats_to_measure(pos):
    """
    In the corpus used in this task there are several POS that are intended to be used, which are speciefied below. Note that several POS are merged into one (e.g. participles and adjectives) in the corpus we used.
    """
    if pos == "NOUN":
        return ["Animacy", "Gender", "Number", "Case"]
    elif pos == "ADJ":
        return ["Gender", "Number", "Case", "Variant", "Degree"]
    elif pos == "PRON":
        return ["Gender", "Number", "Case"]
    elif pos == "DET":
        return ["Gender", "Number", "Case"]
    elif pos == "VERB":
        return ["Aspect", "Gender", "Number", "VerbForm", "Mood", "Tense"]
    elif pos == "ADV":
        return ["Degree"]
    elif pos == "NUM":
        return ["Gender", "Case", "NumForm"]
    else:
        return []


# Due to the issues stemming from the way pymorphy2 tags work, we have to introduce
# the following dictionaries so that tag comparison works correctly.
POS_ALIASES = {
    "ADJ": ["ADJF", "ADJS", "COMP", "PRTF", "PRTS"],
    "ADV": ["ADVB"],
    "NOUN": ["NOUN"],
    "VERB": ["INFN", "GRND"],
    "DET": ["NPRO"],
    "PRON": ["NPRO"],
    "NUM": ["NUMR"],
    "PROPN": ["NOUN"],
}


VALUE_ALIASES = {
    "short": ["ADJS", "PRTS"],
    "cmp": ["COMP"],
    "sup": ["ADJF"],
    "pos": ["ADVB"],
    "anim": ["anim"],
    "inan": ["inan"],
    "perf": ["perf"],
    "imp": ["impf"],
    "nom": ["nomn"],
    "gen": ["gent"],
    "dat": ["datv"],
    "acc": ["accs"],
    "loc": ["loct"],
    "ind": ["indc"],
    "ins": ["ablt"],
    "fem": ["femn"],
    "masc": ["masc"],
    "neut": ["neut"],
    "sing": ["sing"],
    "plur": ["plur"],
    "Brev": ["Short", "Brev"],
    "Short": ["Short", "Brev"],
    "past": ["past"],
    "notpast": ["pres", "futr"],
    "inf": ["INFN"],
    "1": ["1per"],
    "2": ["2per"],
    "3": ["3per"],
}


def count_matching_tags(pos: str, first: dict, second: MorphAnalyzer.TagClass) -> int:
    """
    Count the number of matching tags
    """
    ans = 0
    cats_to_measure = get_cats_to_measure(pos)
    for cat, value in first.items():
        if cat in cats_to_measure:
            ans += set(VALUE_ALIASES.get(value.lower(), [])) in second
    return ans


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "textpath", action="store", help="Path to the text to be parsed."
    )
    args = vars(parser.parse_args())

    SENTS = get_sents(args["textpath"])
    MorphoAnalyzer = MorphAnalyzer()

    df_total = DataFrame(
        data=zeros((1, 10)),
        columns=[
            "sentences",
            "words",
            "tags",
            "NOUN",
            "ADJ",
            "PRON",
            "DET",
            "VERB",
            "ADV",
            "NUM",
        ],
    )
    df_corr = df_total.copy()

    corr_words_prev = 0
    total_words_prev = 0

    df_total.at[0, "sentences"] = len(SENTS)

    for SENT in SENTS:

        corr_words_prev = df_corr.at[0, "words"]
        total_words_prev = df_total.at[0, "words"]

        for WORD in SENT:

            # We do not consider punctiation, particles, conjugates etc.
            if get_cats_to_measure(WORD[1]):

                pymorphy_tags = MorphoAnalyzer.parse(WORD[0])[0].tag

                if not (WORD[1] == "NUM" and WORD[2].get("Form", False)) and (
                    str(pymorphy_tags.POS) == WORD[1]
                    or str(pymorphy_tags.POS) in POS_ALIASES.get(WORD[1])
                ):

                    eq_tags = count_matching_tags(WORD[1], WORD[2], pymorphy_tags)
                    all_tags = len(WORD[2])

                    df_corr.at[0, "tags"] += eq_tags
                    df_total.at[0, "tags"] += all_tags

                    if all_tags == eq_tags:
                        df_corr.at[0, "words"] += 1
                        df_corr.at[0, WORD[1]] += 1

                    df_total.at[0, WORD[1]] += 1
                    df_total.at[0, "words"] += 1

        if (
            df_corr.at[0, "words"] - corr_words_prev
            == df_total.at[0, "words"] - total_words_prev
        ):
            df_corr.at[0, "sentences"] += 1

    print(df_corr)
    print(df_total)
    print(df_corr.div(df_total))
