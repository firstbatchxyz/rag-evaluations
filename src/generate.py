import os
import cohere
import replicate
from openai import OpenAI
from typing import Optional, Dict


class Cohere:
    """
    This class is used to generate responses using the Cohere API.
    """

    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)

    def generate(self, model: str, prompt: str, context: Dict) -> Optional[str]:
        if context is None:
            context = {"title": "", "text": ""}
        try:
            result = self.client.chat(
                model=model,
                message=prompt,
                documents=[
                    {"title": context["title"], "snippet": context["text"]},
                ])
            return result.text
        except Exception as e:
            raise Exception(f"Error generating response with Cohere: {e}") from e


class GPT:
    """
    This class is used to generate responses using the OpenAI GPT-3.5 and GPT-4.5.
    """

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = "Step 1: Analyze context for answering questions.\n"

    "Step 2: Decide context is relevant with question or not relevant with question.\n "
    "Step 3: If any topic about question mentioned in context, use that information for question.\n "
    "Step 4: If context has not mention on question, ignore that context I give you and use your self knowledge.\n "
    "Step 5: Answer the question.\n "

    def generate(self, model: str, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.01
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error generating response with GPT: {e}") from e


class Replicate:
    """
    This class is used to generate responses using the replicate API.
    """

    def __init__(self, api_key: str):
        os.environ["REPLICATE_API_TOKEN"] = api_key
        self.system_prompt = "Step 1: Analyze context for answering questions.\n"
        "Step 2: Decide context is relevant with question or not relevant with question.\n "
        "Step 3: If any topic about question mentioned in context, use that information for question.\n "
        "Step 4: If context has not mention on question, ignore that context I give you and use your self knowledge.\n "
        "Step 5: Answer the question.\n "

    def generate(self, model: str, prompt: str) -> Optional[str]:
        try:
            output = replicate.stream(
                model,
                input={
                    "debug": False,
                    "top_k": 50,
                    "top_p": 0.9,
                    "prompt": prompt,
                    "system_prompt": self.system_prompt,
                    "temperature": 0.01,
                    "max_new_tokens": 512,
                    "min_new_tokens": -1,
                    "prompt_template": "<s>[INST] {prompt} [/INST] ",
                    "repetition_penalty": 1.15
                },
            )
            generated_text = ''""
            for item in output:
                generated_text += item.data
            return generated_text
        except Exception as e:
            raise Exception(f"Error generating response with Replicate: {e}") from e
