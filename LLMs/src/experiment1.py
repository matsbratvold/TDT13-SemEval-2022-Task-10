"""Run the first experiment evaluating the LLMs' perforance. 
It assumes that the LLMs/src/generate_llm_responses.py has been run first to generate predictions.
Modify the combinations variable in the main function to run the experiment with different parameters.
Note that the LLMs/src/generate_llm_responses.py should be run with the same parameters first to ensure that
corresponding prediction are generated."""

import json
import os
from utils import (
    ApiUrl,
    DatasetName,
    ICLUsage,
    InstructionType,
    run_experiment,
    ExperimentType,
)
from itertools import product

def _capitalize_first_letter(string: str) -> str:
    """Helper method capitalize the first letter of a string."""
    return string[0].upper() + string[1:]

def clean_prediction(prediction: dict) -> dict:
    """Makes sure the given prediction has the correct format."""
    indeces_to_remove = []
    if not prediction.get("opinions"):
        prediction["opinions"] = []
        return prediction
    for index, opinion in enumerate(prediction.get("opinions", [])):
        if (
                    opinion.get("Polar_expression") is None
                    or opinion.get("Polarity") is None
                ):
            indeces_to_remove.append(index)
            continue
        for key in ["Source", "Target", "Polar_expression"]:
            if (
                        not isinstance(opinion.get(key, None), list)
                        or len(opinion[key]) != 2
                    ):
                opinion[key] = [[], []]
        opinion["Polarity"] = (
                    _capitalize_first_letter(opinion["Polarity"])
                    if opinion.get("Polarity")
                    else None
                )
    for index in sorted(indeces_to_remove, reverse=True):
        del prediction["opinions"][index]
    return prediction


def gather_predictions(
    dataset_path: DatasetName,
    api_url: ApiUrl,
    instruction_path: InstructionType,
    icl_usage: ICLUsage,
):
    """Gather predictions from individual files into a single file."""
    base_directory = os.path.join(
        "LLMs",
        "predictions",
        ExperimentType.EXPERIMENT1.value,
        dataset_path.name,
        api_url.name,
        instruction_path.name,
        icl_usage.value,
    )
    predictions_path = os.path.join(base_directory, "file_preds")
    output_file_path = os.path.join(base_directory, "predictions.json")

    predictions = []
    if not os.path.exists(predictions_path):
        return
    for file_name in os.listdir(predictions_path):
        file_path = os.path.join(predictions_path, file_name)
        with open(file_path, "r") as f:
            prediction: dict = json.load(f)
            prediction = clean_prediction(prediction)
            predictions.append(prediction)
    with open(output_file_path, "w") as f:
        json.dump(predictions, f, indent=4)




def main():
    combinations = product(
        [DatasetName.NOREC, DatasetName.OPENER_EN, DatasetName.MPQA],
        [ApiUrl.MISTRAL_7B_INSTRUCT],
        [InstructionType.STANDARD, InstructionType.CHAIN_OF_THOUGHT],
        [ICLUsage.ZERO_SHOT, ICLUsage.ONE_SHOT, ICLUsage.FEW_SHOT],
    )
    run_experiment(
        combinations,
        ExperimentType.EXPERIMENT1,
        gather_predictions,
    )


if __name__ == "__main__":
    main()
