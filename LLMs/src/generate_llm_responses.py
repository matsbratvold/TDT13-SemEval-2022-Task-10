"""Generate LLM predictions that can be utilized in experiment1 and experiment2.
Modify the combinations variable in the main function to change what LLM configurations are used.
Running the same parameters twice will not overwrite existing predictions, so you will need to manually
deleted predictions in the file-preds folder if you want to regenerate them."""

from itertools import product
import requests
import regex as re
import os
import json
from utils import (
    ApiUrl,
    DatasetName,
    ExperimentType,
    ICLUsage,
    InstructionType,
    get_dataset_path,
    get_instruction_path,
)
import traceback
import time

API_TOKEN = os.getenv("HF_TOKEN")


class ExitType:
    """Enum for exit types of the generate_predictions function."""

    ERROR = 0
    RATE_LIMIT_REACHED = 1
    SUCCESS = 2


def generate_response(
    text: str,
    instruction: str,
    api_url: str,
    max_new_tokens: int = 250,
) -> dict:
    """Generate a response from a text and instruction using the Inference API
    from Huggingface."""
    inputs = f"{instruction} Analyse the following text: {text} ---Answer:"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
    }
    parameters = {
        "return_full_text": False,
        "max_new_tokens": max_new_tokens,
    }
    payload = {
        "inputs": inputs,
        "parameters": parameters,
    }
    output = requests.post(api_url, headers=headers, json=payload).json()
    return output


def _add_character_offsets_to_opinion(opinion: dict, input_text: str, key: str) -> list:
    """Add character offsets to the opinions."""
    if opinion.get(key):
        value = opinion[key]
        start_index = input_text.find(value)
        if start_index == -1:
            opinion[key] = [[], []]
        else:
            opinion[key] = [
                [value],
                [f"{start_index}:{start_index+len(value)}"],
            ]
    return opinion


def postprocess_response(raw_response: list, input_text: str) -> dict:
    """Postprocess the raw response from the Inference API by adding character offsets"""
    response_dict: dict = raw_response[0]
    opinion_tuples_string = response_dict.get("generated_text")
    if opinion_tuples_string:
        # Only use the first opinions dict if there are multiple
        left_curly_bracket_count = 0
        right_curly_bracket_count = 0
        split_index = 0
        for index, character in enumerate(opinion_tuples_string):
            if character == "{":
                left_curly_bracket_count += 1
            elif character == "}":
                right_curly_bracket_count += 1
            if (
                left_curly_bracket_count == right_curly_bracket_count
                and left_curly_bracket_count > 0
            ):
                split_index = index
                break
        opinion_tuples_string = opinion_tuples_string[: split_index + 1]
        try:
            parsed_response: dict = json.loads(opinion_tuples_string)
        except json.JSONDecodeError:
            return {"opinions": []}
        # Add character offsets
        for opinion in parsed_response.get("opinions", []):
            for key in ["Source", "Target", "Polar_expression"]:
                opinion = _add_character_offsets_to_opinion(opinion, input_text, key)
        return parsed_response
    return {"error": "No opinions found"}


def generate_predictions(
    dataset_name: DatasetName,
    api_url: ApiUrl,
    instruction_type: InstructionType,
    icl_usage: ICLUsage,
) -> ExitType:
    """Run inference on a dataset."""
    data_set_path = get_dataset_path(dataset_name)
    with open(data_set_path, "r") as f:
        dataset = json.load(f)
    with open(get_instruction_path(instruction_type), "r") as f:
        instruction = f.read()
        instruction = re.sub(r"\s+", " ", instruction)
    if icl_usage != ICLUsage.ZERO_SHOT:
        example_path = os.path.join(
            "LLMS",
            "instructions",
            dataset_name.name,
            instruction_type.name,
            f"{icl_usage.value}.txt",
        )
        with open(example_path, "r") as f:
            example = f.read()
            example = re.sub(r"\s+", " ", example)
            instruction = f"{instruction} {example}"
    base_directory = os.path.join(
        "LLMs",
        "predictions",
        ExperimentType.EXPERIMENT1.value,
        dataset_name.name,
        api_url.name,
        instruction_type.name,
        icl_usage.value,
        "file_preds",
    )
    os.makedirs(base_directory, exist_ok=True)
    error_count = 0
    for index, data_dict in enumerate(dataset):
        text = data_dict.get("text")
        sent_id = data_dict.get("sent_id")
        sent_id_suffix = sent_id.split("/")[-1]
        # Only generate response if the file does not already exist
        pred_file_path = os.path.join(base_directory, f"{sent_id_suffix}.json")
        if os.path.exists(pred_file_path):
            continue
        raw_response = generate_response(
            text=text, instruction=instruction, api_url=api_url.value
        )
        print(f"{raw_response=}")
        try:
            output = postprocess_response(raw_response, text)
        except Exception as e:
            traceback.print_exc()
            output = raw_response
        if output.get("error"):
            error_count += 1
            error = output.get("error")
            if error.find("Rate limit reached") != -1:
                return ExitType.RATE_LIMIT_REACHED
            print(f"Error for {sent_id}: {error}")
            if error_count > 9:
                return ExitType.ERROR
        else:
            output["text"] = text
            output["sent_id"] = sent_id
            with open(pred_file_path, "w") as f:
                json.dump(output, f, indent=4)
        print(f"Finished {index+1}/{len(dataset)}", end="\r")
    return ExitType.SUCCESS


def main():
    """Main function."""
    combinations = product(
        [DatasetName.NOREC, DatasetName.OPENER_EN, DatasetName.MPQA],
        [ApiUrl.MISTRAL_7B_INSTRUCT],
        [InstructionType.STANDARD, InstructionType.CHAIN_OF_THOUGHT],
        [ICLUsage.ZERO_SHOT, ICLUsage.ONE_SHOT, ICLUsage.FEW_SHOT],
    )
    for dataset_name, api_url, instruction_type, icl_usage in combinations:
        dataset = json.load(open(get_dataset_path(dataset_name), "r"))
        example_path = os.path.join(
            "LLMs",
            "instructions",
            dataset_name.name,
            instruction_type.name,
            f"{icl_usage.value}.txt",
        )
        if icl_usage != ICLUsage.ZERO_SHOT and (not os.path.exists(example_path)):
            print(
                f"No example file found for the dataset {dataset_name.name} with instruction type {instruction_type.name} and ICL usage {icl_usage.name}, skipping..."
            )
            continue
        file_path = os.path.join(
            "LLMs",
            "predictions",
            ExperimentType.EXPERIMENT1.value,
            dataset_name.name,
            api_url.name,
            instruction_type.name,
            icl_usage.value,
            "file_preds",
        )
        os.makedirs(file_path, exist_ok=True)
        print(
            f"Generating samples using parameters:\n{dataset_name.name=}\n{api_url.name=}\n{instruction_type.name=}\n{icl_usage.name=}"
        )
        if len(os.listdir(file_path)) == len(dataset):
            print(f"Already generated all samples for these parameters, skipping...")
            continue
        while len(os.listdir(file_path)) < len(dataset):
            exit_type = generate_predictions(
                dataset_name, api_url, instruction_type, icl_usage
            )
            if exit_type == ExitType.ERROR:
                # The model might be temporarily unavailable, so we wait 1 minute before trying again
                current_time = time.time()
                while time.time() - current_time < 60:
                    print(
                        f"Error encountered. Restarting in {59 - (time.time() - current_time):.0f} seconds",
                        end="\r",
                    )
                    time.sleep(1)
            elif exit_type == ExitType.RATE_LIMIT_REACHED:
                # If rate limit is reached, you have to wait until the next hour
                while time.localtime().tm_min != 0:
                    print(
                        f"Rate limit used. Restarting in {59 - time.localtime().tm_min:02}:{59 - time.localtime().tm_sec:02}",
                        end="\r",
                    )
                    time.sleep(1)


if __name__ == "__main__":
    main()
