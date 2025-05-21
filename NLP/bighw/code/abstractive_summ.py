"""Скрипт, использующий несколько различных моделей с отрегулированными параметрами для реферирования текста."""
import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from tqdm.auto import tqdm, trange
from os import getenv


MODELS = {
    "FRED-T5-Summarizer": {
        "address": "RussianNLP/FRED-T5-Summarizer", 
        "assistant": "RussianNLP/FRED-T5-Summarizer",
    },
    "rut5-base-finetuned": {
        "address": getenv("HOME")+"/models/rut5-base/checkpoint-14817",
        "assistant": "cointegrated/rut5-base",
    },
}


class RuTextSummarizer:

    def __init__(self, model_name):
        self.model_ = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer_ = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(self, batch):
        inputs = self.tokenizer_(
            batch["text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        labels = self.tokenizer_(
            batch["summary"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = labels["input_ids"]
        return inputs


    def tune(self, dataset: Dataset):
        tokenized_dataset = dataset.map(self.preprocess_function, batched=True)

        training_args = TrainingArguments(
             # output_dir="/home/succ_seed/models/ruflan-t5-base",
             output_dir="/home/succ_seed/models/rut5-base",
             per_device_train_batch_size=2,
             num_train_epochs=3,
        )
        trainer = Trainer(
             model=self.model_,
             args=training_args,
             train_dataset=tokenized_dataset,
             eval_dataset=tokenized_dataset,
             data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        )
        trainer.train()
        return None


    def summarize(self, text, assistant_model=None) -> str:
        inputs = self.tokenizer_(
             text,
             return_tensors="pt",
             max_length=512,
             truncation=True,
             padding="max_length",
        )
        if assistant_model is None:
            summary_ids = self.model_.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=4,
                length_penalty=2.0,
                repetition_penalty = 10.0,
            )
        else:
            summary_ids = self.model_.generate(
                inputs["input_ids"],
                assistant_model=assistant_model,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                prompt_lookup_num_tokens=3,
                repetition_penalty=1.5,
                do_sample=True, 
                temperature=0.4
            )
        return self.tokenizer_.decode(summary_ids[0], skip_special_tokens=True)


if __name__=="__main__":
    ds = load_dataset("RussianNLP/Mixed-Summarization-Dataset", split="test[:40]")
    prefix = "Summarize: "

    for model_name, model_info in MODELS.items():
        assistant_model = AutoModelForSeq2SeqLM.from_pretrained(model_info["assistant"])
        summarizer = RuTextSummarizer(model_info["address"])
        # Сначала смотрим на вывод с русскоязычными префиксами (кратко изложи содержание текста и т.п.)
        print(f"Результаты {model_name} (рус перфиксы, promt lookup):")
        with open(f"./results/{model_name}_lookup.txt", "w") as f:
            for articleno, article in tqdm(enumerate(ds["text"])):
                res = summarizer.summarize(article, assistant_model=assistant_model)
                # print(f'Исходная статья: статья #{articleno+1}')
                # print(f'\n{res}\n\n')
                f.write(f'Исходная статья: статья #{articleno+1}')
                f.write(f'\n{res}\n\n')
        print(f"Результаты {model_name} (рус перфиксы, beam search):")
        with open(f"./results/{model_name}_beam.txt", "w") as f:
            for articleno, article in tqdm(enumerate(ds["text"])):
                res = summarizer.summarize(article)
                # print(f'Исходная статья: статья #{articleno+1}')
                # print(f'\n{res}\n\n')
                f.write(f'Исходная статья: статья #{articleno+1}')
                f.write(f'\n{res}\n\n')

        # Посмотрим так же на результаты при использовании англоязычного префикса, так как некоторые модели могут вести себя лучше с ними.
        print(f"Результаты {model_name} (англ префиксы, prompt lookup):")
        with open(f"./results/{model_name}_lookup_en_prefix.txt", "w") as f:
            for articleno, article in tqdm(enumerate(ds["text"])):
                article = re.sub('"', "", (re.search('"(.*)"', re.sub("\n", " ", article))[0]))
                res = summarizer.summarize(article, assistant_model=assistant_model)
                # print(f'Исходная статья: статья #{articleno+1}')
                # print(f'\n{res}\n\n')
                f.write(f'Исходная статья: статья #{articleno+1}')
                f.write(f'\n{res}\n\n')

        print(f"Результаты {model_name} (англ префиксы, beam search):")
        with open(f"./results/{model_name}_beam_en_prefix.txt", "w") as f:
            for articleno, article in tqdm(enumerate(ds["text"])):
                article = re.sub('"', "", (re.search('"(.*)"', re.sub("\n", " ", article))[0]))
                res = summarizer.summarize(article, assistant_model=assistant_model)
                # print(f'Исходная статья: статья #{articleno+1}')
                # print(f'\n{res}\n\n')
                f.write(f'Исходная статья: статья #{articleno+1}')
                f.write(f'\n{res}\n\n')
