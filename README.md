# TDT13-SemEval-2022-Task-10
Repository hosting the code used for the project in TDT13 Advanced Text Analysis at NTNU during the fall semester of 2024. 

The goal is to assess the capability of LLMs on structured sentiment analysis by comparing the results with baseline models and other entries at task 10 at the SemEval 2022 competition (https://competitions.codalab.org/competitions/33556)

## Run experiments
To run the experiments on the existing predictions, you should run the [experiment1.py](LLMs/src/experiment1.py) and [experiment2.py](LLMs/src/experiment2.py) files from the repos root folder.
Modify the combinations variable within the files' main function to change what datasets and prompt techniques are used.
To make sure the experiments have predictions to evaluate, new predictions should be generated first, using the methodology in the next sections. 
As long as the combinations variables match, and the correct instructions are provided, the experiments should run as expected.

## Generate new predictions

In order to run the experiments on new datasets or prompt configurations, you need to generate new LLM predictions using the [generate_llm_responses.py](LLMs/src/generate_llm_responses.py) file.
Modify the combinations variable in the main function to change what predictions are made, using the provided enums.
Note that in order to generate samples using ICL from other datasets than OpeNER_en, NoReC, and MPQA which are utilized in the project, you need to create corresponding instructions in the LLMs/instructions folder.
Use the following path format: LLMs/instructions/{{DatasetName}}/{{PromptType}}/{{ICLUsage}}.txt

Since the file uses the HuggingFace Inference API to generate responses, you first need to create an account at HuggingFace (https://huggingface.co/), and then generate an API token. See (https://huggingface.co/docs/api-inference/quicktour) for how to generate such a token.
After the token is generated it should be stored as a user environmental variabel, namely ***HF_TOKEN***

Note that only Mistral 7B-Instruct is the only model that is tested thorougly, so it might be necessary to make changes to instructions or add more error handling if using other models through the Inference API.

## Exploration of predictions vs ground truth annotations
The repo also contains a [jupyter notebook](LLMs/src/data_exploration.ipynb) that compares characteristics of the LLM predictions and ground truth annotations in the test datasets. 
The notebook produces figures which are available in the [LLMs/figures](LLMs/figures/) folder. 
Due to an error, the legends are only visible inside the jupyter notebook and not inside the image files.
Nevertheless, the following order is used for all figures: Ground Truth, Std-zero-shot, Std-one-shot, Std-few-shot, CoT-zero-shot, and Cot-one-shot. 

## Data and code for baselines
The [semeval22_structured_sentiment-master](semeval22_structured_sentiment-master/) folder contains code from the original repository associated with the competition (https://github.com/jerbarnes/semeval22_structured_sentiment). 
The datasets are available through the [data folder](semeval22_structured_sentiment-master/data/), which includes training and test datasets for each of the seven datasets that were utilized during SemEval 2022 Task 10.
It also contains code for evaluating results based on the SF1 metric, as provided in the [evaluate.py](semeval22_structured_sentiment-master/evaluation/evaluate.py) file.
Note that the writer was not able to run the baselines due to trouble with versions of python libraries, so modifications might be necessary to property run them.
