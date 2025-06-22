# Load dataset from huggingface and preprocess

from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader


def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name,split="train")
    return dataset

def example_tokenization():
    example_text = "My name is John Locke."
    tokenized_example_text = tokenizer(example_text,return_tensors='pt').input_ids
    print(f"Original Text: {example_text}\nTokenized Text: {tokenized_example_text}\nDecoded Text: {tokenizer.decode(tokenized_example_text[0])}")



def get_preprocessed_data(is_hf=False):
    model_name = "bert-base-uncased"
    num_labels = 2
    model = BertForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_sequence_length = 512
    dataset_name = "stanfordnlp/imdb"

    dataset = get_dataset(dataset_name)

    def tokenization(example):
        return tokenizer(example["text"])

    tokenized_dataset = dataset.map(tokenization)

    # max context length for BERT is 512, so remove instances above 512 tokens
    filtered_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_sequence_length)

    # finally shuffle dataset and put into batches

    if not is_hf:
        filtered_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return filtered_dataset


# data = get_preprocessed_data()