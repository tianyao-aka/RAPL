import os
import sys
import pickle
import json
import time
import openai
from argparse import ArgumentParser
from tqdm import tqdm
from termcolor import colored

def chat(args):
    """
    This function reads three pickle files (train, val, test). Each pickle file is
    a list of dictionaries with the structure:
        {
            'question': str,
            'cand_relations': set(...),
            'gt_relations': set(...)
        }

    For each dictionary in the training data, we can optionally process only a subset
    of the dataset (1/3 at a time) if --train_part is set to 0, 1, or 2. If
    --train_part = -1, we process the entire training dataset.

    For each dictionary in these lists, it asks GPT to identify which of the candidate
    relations are relevant to the question, in the format of a Python list of strings.

    Inputs:
    ---------
    args : Parsed command line arguments:
        - which_key: An integer specifying which API key to use.
        - train_path, val_path, test_path: Paths to train, val, and test pickle files.
        - out_dir: The directory to save the output (JSON) files.
        - train_part: -1 for the entire training set; 0, 1, or 2 for splitting the
          training data into three parts.
    
    Outputs:
    --------
    The script saves three JSON files into out_dir:
        - train_results.json
        - val_results.json
        - test_results.json

    Each JSON file is a list of response dictionaries, for example:
        [
            {
                "question": <question_str>,
                "cand_relations": <candidate_relations_list>,
                "response_text": <raw GPT response>
            },
            ...
        ]
    """

    # Set your OpenAI API key
    key_id = args.which_key
    if key_id == 0:
        print ('type your openai key here')
        openai.api_key = "xxx"  # Replace with your actual API key

    # Create output directory if needed
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data from pickle files
    with open(args.train_path, "rb") as f:
        train_data = pickle.load(f)

    # If we only want a fraction of the training data, handle it here
    N_tr = len(train_data)
    if args.train_part in [0, 1, 2]:
        split_size = N_tr // 3
        start_idx = split_size * args.train_part
        # If it's the last part, take everything until the end
        end_idx = split_size * (args.train_part + 1) if args.train_part < 2 else N_tr
        train_data = train_data[start_idx:end_idx]
    elif args.train_part == -1:
        # Use the entire training set
        pass
    else:
        raise ValueError("train_part must be one of [-1, 0, 1, 2]")

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant for identifying relevant relations "
            "given a question and a set of candidate relations."
        )
    }

    def build_prompt(question, cand_relations):
        """
        Builds the prompt for GPT by taking in a question and a set of candidate relations.

        Inputs:
        question: str
            The input question for which we want relevant relations.
        cand_relations: iterable
            The set (or list) of candidate relations.

        Output:
        A string that we feed into the 'content' of a 'user' role message.
        """
        prompt_str = (   # try without examples
            f"We have the question:\n\n{question}\n\n"
            f"And a set of relations for this question:\n\n{cand_relations}\n\n"
            "List the relevant relations for this question. Please respond using the following format:\n\n"
            "<Solution:> [r1,r2,....] \n\n"
        )
        return prompt_str

    def gpt_call(prompt):
        """
        Sends a chat completion request to OpenAI's API using the messages
        in 'prompt'. This function continually retries on error.

        Input:
        ------
        prompt: List[dict]
            Example structure:
            [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."}
            ]

        Output:
        -------
        str
            The content of the assistant's response.
        """
        
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=prompt,
                    temperature=0.0,
                )
                return response["choices"][0]["message"]["content"]

            except openai.error.RateLimitError:
                print(colored("Rate limit reached. Waiting 60 seconds...", 'red'))
                time.sleep(60)
                # Retry with the same prompt in the next loop iteration

    def process_data(data_list):
        """
        Processes a list of data items. Each data item is a dictionary:
            {
                'question': str,
                'cand_relations': set(...),
                'gt_relations': set(...)
            }

        We build a prompt for GPT and then store GPT's response in a list.

        Inputs:
        data_list: list of dictionaries

        Output:
        A list of dictionaries, each containing the question, candidate relations,
        and the GPT response text.
        """
        results = []
        for item in tqdm(data_list, desc="Processing data"):
            q = item["question"]
            # Convert sets to lists for easier handling
            cand_rels = list(item["cand_relations"])
            gt_relations = list(item["gt_relations"])

            user_prompt = build_prompt(q, cand_rels)
            messages = [
                system_message,
                {"role": "user", "content": user_prompt}
            ]
            response_text = gpt_call(messages)
            

            results.append({
                "question": q,
                "cand_relations": cand_rels,
                "response_text": response_text,
                "gt_relations": gt_relations
            })

        return results

    print(colored(f"Processing training set (size={len(train_data)}).", 'green'))
    train_results = process_data(train_data)

    # Save all results to out_dir as pickle files
    with open(os.path.join(args.out_dir, "train_results.pkl"), "wb") as f:
        pickle.dump(train_results, f)


    print(colored("All results saved to output directory.", 'blue'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_path', type=str, default="data_files/webqsp/train_res_topic_relations.pkl", help='Path to the train pickle file')
    parser.add_argument('--out_dir', type=str, default="data_files/webqsp/topic_relation_candidates/", help='Directory to save the results')
    parser.add_argument('--which_key', type=int, default=0, choices=[0,1,2], help='Which OpenAI key to use')
    parser.add_argument('--train_part', type=int, default=-1,
                        choices=[-1,0,1,2],
                        help='-1 means use the entire training set; 0,1,2 for splitting the dataset into thirds')
    
    args = parser.parse_args()
    chat(args)

