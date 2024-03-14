from src.generate import Cohere
from src.retreive import generate


def evaluate(row, evaluator, model_id, judge):
    prompt = "\n'''{0}'''\n\n" \
             "**Question**: {1}\n\n"
    if evaluator.__class__ == Cohere:
        result_simple = evaluator.generate(model_id, prompt.format("", row["question"]), context=None)
    else:
        result_simple = evaluator.generate(model_id, prompt.format("", row["question"]))
    context, result_rag = generate(evaluator, prompt.format("{0}", "{1}"), row["question"], model_id,
                                   result_simple)

    acc_rag = judge.evaluate(row["question"], row["answer"], result_simple)
    acc_simple = judge.evaluate(row["question"], row["answer"], result_rag)

    return {
        "question": row["question"],
        "truth": row["answer"],
        "prediction_simple": acc_simple,
        "prediction_rag": acc_rag,
        "prediction_response_simple": result_simple,
        "prediction_response_rag": result_rag,
        "context": context,
    }
