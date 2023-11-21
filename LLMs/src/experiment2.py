"""Run the second experiment exploiting the metric and characteristics of datasets.
In order to utilize the sentiment polarity assessement from LLMs, the
LLMS/src/generate_llm_responses.py should be run first to generate predictions.
Modify the combinations variable in the main function to run the experiment with different parameters,
ensuring that generate_llm_responses.py has been run with the same parameters first to ensure that
corresponding prediction are generated."""

import json
import os

from utils import (
    ApiUrl,
    DatasetName,
    ICLUsage,
    InstructionType,
    ExperimentType,
    get_dataset_path,
    run_experiment,
)
from itertools import product


def exploit_metric(
    dataset_name: DatasetName,
    api_url: ApiUrl = None,
    instruction_path: InstructionType = None,
    icl_usage: ICLUsage = None,
):
    """Try to exploit the performance metric and characteristics of the datasets.
    If an API URL and instruction path are provided, the LLM is used to evaluate polarity.
    """
    dataset_path = get_dataset_path(dataset_name)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    predictions = []
    llm_predictions = None
    if api_url is not None and instruction_path is not None and icl_usage is not None:
        predictions_path = os.path.join(
            "LLMs",
            "predictions",
            ExperimentType.EXPERIMENT1.value,
            dataset_name.name,
            api_url.name,
            instruction_path.name,
            icl_usage.value,
            "predictions.json",
        )
        if os.path.exists(predictions_path):
            with open(predictions_path, "r") as f:
                llm_predictions: list = json.load(f)
        else:
            print(f"No LLM predictions found for the prediction path: {predictions_path}")
            return
    for data_dict in dataset:
        text = data_dict.get("text")
        polarity = "Negative" if dataset_name == DatasetName.DS_UNIS else "Positive"
        if llm_predictions is not None:
            llm_prediction = next(
                (
                    prediction
                    for prediction in llm_predictions
                    if prediction["sent_id"] == data_dict.get("sent_id")
                ),
                None,
            )
            if llm_prediction is not None:
                text = llm_prediction["text"]
                polarities = [
                    opinion["Polarity"] for opinion in llm_prediction["opinions"]
                ]
                standard_polarity = (
                    "Negative" if dataset_name == DatasetName.DS_UNIS else "Positive"
                )
                if len(polarities) == 0:
                    polarity = None
                else:
                    # The standard polarity is chosen if there is a tie
                    polarity_key = lambda x: polarities.count(x) + (
                        0.5 if x == standard_polarity else 0
                    )
                    polarity = max(set(polarities), key=polarity_key)
        predictions.append(
            {
                "text": text,
                "sent_id": data_dict.get("sent_id"),
                "opinions": [
                    {
                        "Source": [[text], [f"0:{len(text)-1}"]]
                        if dataset_name == DatasetName.MPQA
                        else [[], []],
                        "Target": [[text], [f"0:{len(text)-1}"]],
                        "Polar_expression": [[text], [f"0:{len(text)-1}"]],
                        "Polarity": polarity,
                    },
                ],
            }
            if polarity is not None
            else {
                "text": text,
                "sent_id": data_dict.get("sent_id"),
                "opinions": [],
            }
        )
    extra_params = (
        ["None"]
        if api_url is None
        else [
            api_url.name,
            instruction_path.name,
            icl_usage.value,
        ]
    )
    file_path = os.path.join(
        "LLMs",
        "predictions",
        ExperimentType.EXPERIMENT2.value,
        dataset_name.name,
        *extra_params,
        "predictions.json",
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(predictions, f, indent=4)


def main():
    """Main function running experiment 2."""
    combinations = product(
        [DatasetName.NOREC, DatasetName.OPENER_EN, DatasetName.MPQA],
        [None, ApiUrl.MISTRAL_7B_INSTRUCT],
        [InstructionType.STANDARD, InstructionType.CHAIN_OF_THOUGHT],
        [ICLUsage.ZERO_SHOT, ICLUsage.ONE_SHOT, ICLUsage.FEW_SHOT],
    )
    run_experiment(combinations, ExperimentType.EXPERIMENT2, exploit_metric)


if __name__ == "__main__":
    main()
