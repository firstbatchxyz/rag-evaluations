import os
import datasets as huggingface
from src.generate import GPT, Replicate, Cohere, Ollama


def load_dataset(sl=100):
    return huggingface.load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{sl}%]")


def load_models():
    gpt = GPT(api_key=os.environ["OPENAI_API_KEY"])
    cohere = Cohere(api_key=os.environ["COHERE_API_KEY"])
    replicate = Replicate(api_key=os.environ["REPLICATE_API_KEY"])
    ollama = Ollama()
    models = {
        "gpt-4-0125-preview": gpt,
        "gpt-3.5-turbo-0125": gpt,
        "command-r": cohere,
        "meta/llama-2-70b-chat": replicate,
        "meta/llama-2-13b-chat": replicate,
        "mistralai/mixtral-8x7b-instruct-v0.1": replicate,
        "llama2:7b-chat": ollama,
    }
    return models
