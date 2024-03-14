from openai import OpenAI


class Judge:
    """
    This class is used to evaluate if the response generated by the Language Learning Model (LLM) aligns with the correct answer.
    """

    def __init__(self, api_key: str):
        """
        Initialize the Judge class with the OpenAI API key.

        :param api_key: The API key to use for the OpenAI API.
        """
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = (
            "You are a judge of if two answer aligns or not. "
            "Evaluate LLM's response then evaluate correct answer. "
            "Compare them each other. Then decide if they align or not align. "
            "If LLM's response aligns with Correct answer, return 'Correct'. "
            "If they not align return 'Incorrect'. "
            "Please do not use any other words except Correct or Incorrect"
        )

    def evaluate(self, question: str, correct: str, response: str) -> int:
        """
        Evaluate if the LLM's response aligns with the correct answer.

        :param question: The question that was asked.
        :param correct: The correct answer to the question.
        :param response: The response generated by the LLM.
        :return: The evaluation result ('Correct' or 'Incorrect') and the prompt.
        """
        prompt = f"**Question**: {question}\n\n" \
                 f"**Correct answer**: {correct}\n\n" \
                 f"**LLM's response**: {response}"
        response = self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return 1 if "Correct" in response.choices[0].message.content.strip() else 0