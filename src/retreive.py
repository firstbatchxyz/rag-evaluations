import time
from collections import Counter

from dria import DriaLocal
from dria.exceptions import DriaRequestError

from src.embedding import BGELarge
from src.generate import Cohere


def get_relevant_paragraph(question, text, title):
    """
    Get the most relevant paraphrases from the context

    Args:
    question (str): The question
    text (str): The context
    title (str): The title of the context
    """
    context_keywords = Counter()

    for idx, i in enumerate(text):
        commons = len(list(set((title + question).split()).intersection(set(i.split()))))
        context_keywords[idx] = commons
    return "\n".join([text[i[0]] for i in Counter.most_common(context_keywords, 2)])


class RAG:
    def __init__(self, model, num_passages=1):
        super().__init__()
        self.bge_encoder = BGELarge()
        self.top_n = num_passages
        self.retrieve = DriaLocal().query
        self.model = model

    def forward(self, prompt, question, simple_answer, model_id):
        """
        Retrieve most relevant context from Dria and then generate the response using the model.

        Args:
        prompt (str): The prompt to use for the model
        question (str): The question
        simple_answer (str): The answer from the without RAG model
        model_id (str): The model to use

        """
        question_embed = self.bge_encoder.encode(question)
        try:
            context = self.retrieve(question_embed, top_n=self.top_n)
        except DriaRequestError:
            return [], simple_answer

        context = self.filter(context)
        if not context and simple_answer:
            # If no context is found, return the answer from the without RAG model
            return context, simple_answer
        title = context[0]['metadata']['title']
        context = get_relevant_paragraph(question, context[0]['metadata']['text'].split("\n"), title)

        return self.call_model_with_retry(context, prompt, question, model_id, title)

    @staticmethod
    def filter(context, threshold=0.7):
        return [c for c in context if c["score"] > threshold]

    def call_model_with_retry(self, context, prompt, question, model_id, title, max_retries=3):
        for i in range(max_retries):
            try:
                if self.model.__class__ == Cohere:
                    return context, self.model(model_id, prompt.format("", question), {"title": title, "text": context})
                return context, self.model(model_id, prompt.format(context, question))
            except Exception as e:
                if i < max_retries - 1:  # i is zero indexed
                    time.sleep(2 ** i)  # exponential backoff
                else:
                    return "I don't know", "Error"


def generate(model, prompt, question, model_id, simple_answer):
    rag = RAG(model)
    return rag.forward(prompt, question, simple_answer, model_id)
