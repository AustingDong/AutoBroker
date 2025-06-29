# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="tiiuae/Falcon3-7B-Instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
