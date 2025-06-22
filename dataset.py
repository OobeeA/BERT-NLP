# Load dataset from huggingface and preprocess

from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset("stanfordnlp/imdb",split="train")
model_name = "bert_base_uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
