import numpy as np
import os
import sys
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever_v2 import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample

import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import re
from copy import deepcopy as dp
from termcolor import colored
import openai

import warnings
warnings.filterwarnings("ignore")

def dict_to_text(data):
    """
    Converts a dictionary containing a question, translated paths, and reasoning paths
    into a formatted text representation.

    Args:
        data (dict): The dictionary containing:
            - "question": The main question.
            - "translated_paths": A list of paths.
            - "reasoning_paths": A list of reasoning steps corresponding to each path.

    Returns:
        str: A formatted string representation of the data.
    """
    text = f"Question: {data['question']}\nThe paths are:\n"

    for i, (path, reasoning) in enumerate(zip(data["translated_paths"], data["reasoning_paths"]), 1):
        text += f"{i}. {path}, the corresponding reasoning path is: {reasoning}\n\n"

    return text.strip()  # Remove the trailing newline



def extract_latest_answer(response_text):
    # Split into different turns based on [INST] markers
    turns = re.split(r"\[INST\]", response_text)
    
    if len(turns) < 2:
        return response_text  # Return raw response if it doesn't follow the expected format
    
    # Get the last assistant's response (after the last user question)
    last_turn = turns[-1]  # Last user question + assistant response
    last_answer = re.sub(r"</?s>", "", last_turn)  # Remove <s> and </s>
    
    # Extract only the assistant's part (text after [/INST])
    last_answer = last_answer.split("[/INST]")[-1].strip()
    
    return last_answer


def convert_messages_to_response_text(messages):
    """
    Converts a list of structured chat messages into a formatted response text 
    with <s> and [INST] markers, mimicking Mistral-7B's output.

    Args:
        messages (list): List of {"role": "user"/"assistant", "content": "..."} messages.

    Returns:
        str: The formatted chat response string.
    """
    response_text = ""
    
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            response_text += f"<s> [INST] {msg['content']} [/INST] "
        elif msg["role"] == "assistant":
            response_text += f"{msg['content']} </s>"

    return response_text.strip()

def chat(args):
    # Set your OpenAI API key
    key_id = args.which_key
    if key_id == 0:
        print ('type your key id here')
        openai.api_key = "xxx"

    train_split_id = args.train_part
    dataset = args.dataset
    
    path_tr = f"data_files/{dataset}/processed/train_text_dict_list.pickle"
    path_val = f"data_files/{dataset}/processed/val_text_dict_list.pickle"
    path_val = f"data_files/{dataset}/processed/test_text_dict_list.pickle"
    with open(path_tr, "rb") as f:
        train_data = pickle.load(f)
    with open(path_val, "rb") as f:
        val_data = pickle.load(f)
    # with open(path_test, "rb") as f:
    #     val_data = pickle.load(f)  #! remember to change back
    print (f"val data len: {len(val_data)}")
    
    # Define the system and prompt as per the original function
    first_prompt = """
    For the QA task, follow the following template to answer the question and list the rational paths:
    
    <Solution> The rational paths are:
    1. <relation path1>
    2. <relation_path2>
    ....

     <Solution> is the special token here.
    Next, let's start with a example.
    Question: what character did john noble play in lord of the rings
    The reasoning paths are:
    1. John Noble -> film.actor.film -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['film.actor.film', 'film.performance.character']
    2. John Noble -> film.actor.film -> m.0528y98 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['film.actor.film', 'film.performance.character']
    3. John Noble -> award.award_winner.awards_won -> m.09k3pgy -> award.award_honor.honored_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_winner.awards_won', 'award.award_honor.honored_for', 'film.film.starring', 'film.performance.character']
    4. John Noble -> award.award_nominee.award_nominations -> m.09k3q0p -> award.award_nomination.nominated_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_nominee.award_nominations', 'award.award_nomination.nominated_for', 'film.film.starring', 'film.performance.character']
    5. John Noble -> award.award_winner.awards_won -> m.0n7xsws -> award.award_honor.honored_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_winner.awards_won', 'award.award_honor.honored_for', 'film.film.starring', 'film.performance.character']
    6. John Noble -> award.award_nominee.award_nominations -> m.0b4d5rz -> award.award_nomination.nominated_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_nominee.award_nominations', 'award.award_nomination.nominated_for', 'film.film.starring', 'film.performance.character']
    """

    first_response = """
    The most direct and relevant way to determine John Noble’s “Lord of the Rings” character is via his actor–performance relationship, rather than detouring through award links. Paths #1 and #2 both use the same reasoning relations (“film.actor.film” → “film.performance.character”) and therefore are duplicates from a reasoning standpoint. The longer award-based paths (#3–#6) are not the most straightforward way to answer “What character did John Noble play?” and thus are less rational for this specific question.

    after deduplication, the rational path is:

    <Solution> The rational paths are:
    1. [film.actor.film, film.performance.character]
    """


    job_start = "Now, let's begin! Identify all the rational paths, and list below with explanations. "

    # Load training and validation data
    N_tr = len(train_data)
    N_val = len(val_data)
    train_samples = [dict_to_text(train_data[i][i]) for i in range(N_tr)]
    val_samples = [dict_to_text(val_data[i][i]) for i in range(N_val)]

    # Messages to pass for the GPT API
    messages = [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": job_start}
    ]

    cnt = 0
    SLEEP_TIME = 60  # Adjust this value if needed based on your rate limit
    # Process training set
    save_path = f"data_files/{dataset}/annotated_paths_GPT4o/train/"
    os.makedirs(save_path, exist_ok=True)
    split_size = N_tr // 3
    start_idx = split_size * train_split_id
    end_idx = split_size * (train_split_id + 1) if train_split_id < 2 else N_tr

    if train_split_id == -1:
        samples_to_process = train_samples
    else:
        samples_to_process = train_samples[start_idx:end_idx]

    for idx, sample in tqdm(enumerate(samples_to_process)):
        if os.path.exists(save_path + f'sample_{start_idx + idx}.txt'): 
            print(colored(f"sample_{start_idx + idx}.txt already exists, skipping", 'red'))
            continue
        print(colored(f"Processing sample {start_idx + idx}/{N_tr} \n", 'yellow'))
        
        # Prepare GPT messages
        copy_messages = dp(messages)
        copy_messages.append({"role": "user", "content": sample})

        # Request GPT-4o response
        try:
            # Request GPT-4o response
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Specify GPT-4o
                messages=copy_messages,
                temperature=0.,  # Adjust for deterministic output
            )

            # Extract and save the response
            s = response["choices"][0]["message"]["content"]
            with open(save_path + f'sample_{start_idx + idx}.txt', 'w') as f:
                f.write(s)
            cnt += 1
            
        except openai.error.RateLimitError as e:
            # Handle rate limit error: Wait and try again
            print(colored(f"Rate limit reached, retrying after {SLEEP_TIME} seconds...", 'red'))
            time.sleep(SLEEP_TIME)  # Sleep for the specified time
            continue  # Retry the current sample

        except Exception as e:
            # Handle other exceptions: Wait 180 seconds before retrying
            print(colored(f"Error encountered: {e}. Retrying after 60 seconds...", 'cyan'))
            time.sleep(60)
        print ('processed sample count:',cnt)


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='webqsp',
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('--train_part', type=int, default=-1,
                        choices=[-1,0,1,2], help='-1 mean make the entire train dataset')
    parser.add_argument('--which_key', type=int, default=0,
                        choices=[0], help='which openai key to use')
    args = parser.parse_args()
    chat(args)
    
    