import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
from functools import partial
from datasets import Dataset

from helm.common.general import ensure_directory_exists
from helm.proxy.clients.huggingface_model_client import HuggingFaceModelClient
from helm.benchmark.run_spec import RunSpec

from llmsanitize.utils.logger import get_child_logger
from llmsanitize.closed_data_methods.ts_guessing_question_based import filter_data

logger = get_child_logger("ts_guessing_question_multichoice")

def build_prompt(example, eval_data_name):
    text = example["text"]
    choices = example["choices"]
    answer_index = example["answer_index"]
    answer = choices[answer_index]
    wrong_choices_indices = [i for i in range(len(choices)) if i != answer_index]
    wrong_choice_index = np.random.choice(wrong_choices_indices)
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    wrong_letter = alphabet[wrong_choice_index]
    prompt = f"Please fill in the [] in option {wrong_letter} based on your benchmark knowledge."
    prompt += "\n\nThe crucial rule is that you should provide a different answer in other options below."
    prompt += f"\n\nQuestion: {text}"
    prompt += "\nOptions:"
    
    for i, choice in enumerate(choices):
        letter = alphabet[i]
        prompt += f"\n{letter}: [{'MASK' if i == wrong_choice_index else choice}]"
    
    prompt += "\n\nReply with answer only."
    
    return prompt, answer, wrong_letter

def process_response(response, wrong_letter):
    symbol = wrong_letter + ":"
    if symbol in response:
        response = response.split(symbol)[1]
        sents = sent_tokenize(response)
        if len(sents) > 0:
            response = sents[0]
    return response

def inference(data_point, eval_data_name, model_client):
    prompt, answer, wrong_letter = build_prompt(data_point, eval_data_name)
    response = model_client.make_request(prompt)
    response = process_response(response, wrong_letter)
    
    data_point["answer"] = answer
    data_point["response"] = response
    return data_point

def main_ts_guessing_question_multichoice(
    eval_data: list,
    eval_data_name: str,
    model_name: str,
    n_eval_data_points: int = 100,
    num_proc: int = 16
):
    data_points = filter_data(eval_data, eval_data_name)
    logger.info(f"{len(data_points)} data points left after filtering")
    
    if n_eval_data_points > 0:
        np.random.shuffle(data_points)
        data_points = data_points[:n_eval_data_points]
        logger.info(f"{len(data_points)} data points left after subsampling")
    
    data_points = Dataset.from_list(data_points)
    model_client = HuggingFaceModelClient(RunSpec(model=model_name))
    
    process_fn = partial(inference, eval_data_name=eval_data_name, model_client=model_client)
    ts_guessing_results = data_points.map(process_fn, num_proc=num_proc)
    
    answers = [x["answer"].lower() for x in ts_guessing_results]
    responses = [x["response"].lower() for x in ts_guessing_results]
    em = sum(1 for i in range(len(responses)) if responses[i] == answers[i]) / len(responses)
    scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    rl = np.mean([scorer.score(responses[i], answers[i])['rougeLsum'].fmeasure for i in range(len(responses))])
    
    logger.info(f"Question-based guessing")
    logger.info(f"Exact Match (EM): {em:.2f}, ROUGE-L F1: {rl:.2f}")
