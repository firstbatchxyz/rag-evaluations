import os
import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from src.evaluator import evaluate
from src.judge import Judge
from src.utils import load_dataset, load_models


def main(max_worker, output_dir, dataset_slice):
    ds = load_dataset(dataset_slice)
    models = load_models()
    gpt_judge = Judge(api_key=os.environ["OPENAI_API_KEY"])

    os.makedirs(output_dir, exist_ok=True)

    for model, evaluator in models.items():
        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            results = list(
                tqdm(executor.map(lambda row: evaluate(row, evaluator, model, gpt_judge), ds), total=len(ds)))

        df = pd.DataFrame(results)
        model_safe_name = model.replace('/', '_')
        df.to_csv(os.path.join(output_dir, f"{model_safe_name}_results"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models on the HotpotQA dataset.')
    parser.add_argument('--max_worker', type=int, default=32,
                        help='an integer for the max_worker')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='a string for the output directory')
    parser.add_argument('--dataset_slice', type=str, default='100',
                        help='a string for the dataset slice percentage')

    args = parser.parse_args()
    main(args.max_worker, args.output_dir, args.dataset_slice)
