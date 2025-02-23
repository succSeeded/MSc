from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pymorphy2 import MorphAnalyzer


def get_sents(infile):
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


POS_ALIASES = {
    "ADJ": ["ADJF", 'ADJS', "COMP", "PRTF", "PRTS"],
    "ADV": ["ADVB"],
    "VERB": ["VERB", "INFN", "GRND"],
    "PRON": ["NPRO"],
    "NUM": ["NUMR"],
}

VALUE_ALIASES = {
    "Inf": ["INFN"],
    "Brev": ["Short", "Brev"],
    "Short": ["Short", "Brev"],
    "Notpast": ["Pres", "Fut"],
    "NumForm": ["Form"],
    "loct": ["loc"],
    "gent": ["gen"],
    "accs": ["acc"],
    "femn": ["fem"],
    "indc": ["ind"],
    "3per": ["3"],
}


def are_equal_tags(pos, first, second):
    cats_to_measure = get_cats_to_measure(pos)
    for cat, value in first.items():
        if cat in cats_to_measure:
            second_value = second.get(cat)
            if not (
                second_value == value or second_value in VALUE_ALIASES.get(value, [])
            ):
                return False
    return True

def get_tags_from_string(tags_string):
    tags = []
    POS = tags_string[:4]
    for pos_key, pos_aliases in POS_ALIASES.items():
        if POS in pos_aliases:
            tags.append(pos_key)
            break
    


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "textpath", action="store", help="Path to the text to be parsed."
    )
    args = vars(parser.parse_args())

    SENTS = get_sents(args["textpath"])
    MorphoAnalyzer = MorphAnalyzer()

    for SENT in SENTS:
        for WORD in SENT:
            base_form = WORD[0]
            likeliest_morphology = MorphoAnalyzer.parse(base_form)[0]
            
            WORD_PARSED = [likeliest_morphology["word"], likeliest_morphology["normal_form"]]
