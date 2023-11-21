"""This module contains utility functions and classes/enums for the LLM project."""

import enum
import subprocess
from typing import Callable, List, Tuple
import pandas as pd
import os


class ApiUrl(enum.Enum):
    GPT2 = "https://api-inference.huggingface.co/models/gpt2"
    MISTRAL_7B = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
    MISTRAL_7B_INSTRUCT = (
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    )
    ZEPHYR_7B = "https://api-inference.huggingface.co/models/mistralai/Zephyr-7B-beta"


class DatasetName(enum.Enum):
    """Enum class for dataset paths."""

    MULTIBOOKED_CA = "multibooked_ca"
    MULTIBOOKED_EU = "multibooked_eu"
    NOREC = "norec"
    OPENER_EN = "opener_en"
    OPENER_ES = "opener_es"
    MPQA = "mpqa"
    DS_UNIS = "darmstadt_unis"


class InstructionType(enum.Enum):
    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "chain-of-thought"


class ICLUsage(enum.Enum):
    ZERO_SHOT = "zero-shot"
    ONE_SHOT = "one-shot"
    FEW_SHOT = "few-shot"


class ExperimentType(enum.Enum):
    EXPERIMENT1 = "experiment1"
    EXPERIMENT2 = "experiment2"


def get_dataset_path(dataset: DatasetName):
    """Returns the path to the dataset."""
    return os.path.join(
        "semeval22_structured_sentiment-master", "data", dataset.value, "test.json"
    )


def get_instruction_path(instruction_type: InstructionType):
    """Returns the path to the instruction file."""
    return os.path.join("LLMs", "instructions", f"{instruction_type.value}.txt")


def calculate_f1_score(dataset_name: DatasetName, predictionsPath: str) -> float:
    """Calculates the F1 score for the given dataset and predictions."""
    dataset_path = get_dataset_path(dataset_name)
    f1_score_bytes = subprocess.check_output(
        f"python semeval22_structured_sentiment-master/evaluation/evaluate_single_dataset.py {dataset_path} {predictionsPath}"
    )
    f1_score_str = f1_score_bytes.decode("utf-8").split(":")[-1].strip()
    return float(f1_score_str)


def _prepare_experiment_csv(results: pd.DataFrame, file_path: str):
    """Helper method prepararing a CSV file containing results from an experiment."""
    results = results.drop_duplicates()
    results = results.pivot(
        index=["api_url", "instruction_path", "icl_usage"],
        columns="dataset",
        values="f1_score",
    )
    results["average"] = results.mean(axis=1)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    results.to_csv(file_path, index=True)


def run_experiment(
    combinations: List[Tuple[DatasetName, ApiUrl, InstructionType, ICLUsage]],
    experiment_type: ExperimentType,
    gather_predictions_function: Callable[[DatasetName, ApiUrl, InstructionType], None],
):
    """Runs a single experiment based on the given combinations of parameters and function for gathering predictions."""
    columns = ["dataset", "api_url", "instruction_path", "icl_usage", "f1_score"]
    results = pd.DataFrame(columns=columns)
    for dataset_path, api_url, instruction_path, icl_usage in combinations:
        if api_url is None or instruction_path is None or icl_usage is None:
            api_url = None
            instruction_path = None
            icl_usage = None
        # If results have already been gathered, skip
        if (
            results[
                (results["dataset"] == dataset_path.name)
                & (results["api_url"] == (api_url.name if api_url else "None"))
                & (
                    results["instruction_path"]
                    == (instruction_path.name if instruction_path else "None")
                )
                & (results["icl_usage"] == (icl_usage.name if icl_usage else "None"))
            ].shape[0]
            > 0
        ):
            continue
        gather_predictions_function(dataset_path, api_url, instruction_path, icl_usage)
        if experiment_type == ExperimentType.EXPERIMENT1:
            path_params = [
                "LLMs",
                "predictions",
                experiment_type.value,
                dataset_path.name,
                api_url.name,
                instruction_path.name,
                icl_usage.value,
                "predictions.json",
            ]
        else:
            extra_params = (
                ["None"]
                if api_url is None
                else [
                    api_url.name,
                    instruction_path.name,
                    icl_usage.value,
                ]
            )
            path_params = [
                "LLMs",
                "predictions",
                experiment_type.value,
                dataset_path.name,
                *extra_params,
                "predictions.json",
            ]
        predictions_path = os.path.join(*path_params)
        if not (os.path.exists(predictions_path)):
            continue
        api_name = "None" if api_url is None else api_url.name
        instruction_name = "None" if instruction_path is None else instruction_path.name
        icl_usage = "None" if icl_usage is None else icl_usage.name
        print("*" * 150)
        print(
            f"Running {experiment_type.value} with the following parameters:\n{dataset_path.name=}\n{api_name=}\n{instruction_name=}\n{icl_usage=}"
        )
        f1_score = calculate_f1_score(dataset_path, predictions_path)
        print(f"Result: {f1_score}")
        print("*" * 150)
        if f1_score is not None:
            result = pd.DataFrame(
                [
                    [
                        dataset_path.name,
                        api_name,
                        instruction_name,
                        icl_usage,
                        f1_score,
                    ]
                ],
                columns=columns,
            )
            results = pd.concat([results, result])
    file_path = os.path.join(
        "LLMs",
        "experiments",
        f"{experiment_type.value}.csv",
    )
    _prepare_experiment_csv(results, file_path)


def main():
    """Main function testing the calculate_f1_score function."""
    predictions_path = os.path.join(
        "LLMs",
        "predictions",
        DatasetName.OPENER_EN.name,
        ApiUrl.MISTRAL_7B_INSTRUCT.name,
        InstructionType.STANDARD.name,
        ICLUsage.ONE_SHOT.value,
        "predictions.json",
    )
    print(f"{calculate_f1_score(DatasetName.OPENER_EN, predictions_path)=}")


if __name__ == "__main__":
    main()
