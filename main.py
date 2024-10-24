import argparse
from eval_sentiment import infer


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--data", type=str, required=True,
                    default="data/data_test_ur.json")


if __name__ == "__main__":
    args = parser.parse_args()
    infer(model_path=args.model, data_path=args.data)
