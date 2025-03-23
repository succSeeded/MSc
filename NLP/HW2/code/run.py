import spacy

if __name__=="__main__":
    nlp = spacy.load("ru_core_news_sm")
    with open("dataset.txt", "r", encoding="utf8") as file:
        candidates_dict = {}
        for i in range(100):
            fline = file.readline().strip()
            tokenized_line = nlp(fline)
            for i in range(len(tokenized_line)-1):
                if (tokenized_line[i].pos_, tokenized_line[i+1].pos_) in [("NOUN", "NOUN"), ("PROPN", "NOUN"), ("NOUN", "PROPN"), ("NOUN", "ADJ"), ("ADJ", "NOUN"), ("PROPN", "ADJ"), ("ADJ", "PROPN")] and (tokenized_line[i].text != "-") and (tokenized_line[i+1].text != "-"):
                    collocation_candidate = f"{tokenized_line[i].text} {tokenized_line[i+1].text}"
                    if collocation_candidate in candidates_dict:
                        candidates_dict[collocation_candidate] += 1
                    else:
                        candidates_dict[collocation_candidate] = 1
        candidates_dict = sorted(candidates_dict.items(), key=lambda x: x[1])
        print(candidates_dict[0], candidates_dict[1], candidates_dict[2])
        print(candidates_dict[-3], candidates_dict[-2], candidates_dict[-1])
