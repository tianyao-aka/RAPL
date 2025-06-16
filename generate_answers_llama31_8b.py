import os
import sys
import torch
import pickle
import re
from tqdm import tqdm
from copy import deepcopy as dp
from termcolor import colored
import warnings
import time
warnings.filterwarnings("ignore")

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer



def convert_messages_to_response_text(messages):
    """
    Converts a list of structured chat messages into a single text prompt.
    Here we implement a simple user/assistant style conversation.
    
    Args:
        messages (list): A list of dictionaries, each with "role" (e.g. "system" or "user")
                         and "content" (string).
    
    Returns:
        str: A single string that represents the combined prompt for the model.
    """
    prompt_text = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt_text += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            prompt_text += f"User: {msg['content']}\n"
    return prompt_text.strip()

def extract_latest_answer(response_text):
    """
    Given the model's raw generation text, extracts or cleans up the final
    assistant answer. Adjust as needed for your own prompt style.
    
    Args:
        response_text (str): The full text output from the language model.
    
    Returns:
        str: The cleaned or final portion of the model's answer.
    """
    # Simple approach: just strip whitespace
    answer = response_text.strip()
    return answer



def chat(args):
    """
    Main function that:
      1) Lists all sample text files in args.file_dir whose names match `sample_*.txt`.
      2) Loads the Llama-3.1-8B-Instruct model and tokenizer.
      3) Iterates through each sample file, reads its content as the user message.
      4) Builds a properly formatted conversation prompt.
      5) Uses `model.generate()` to obtain a response.
      6) Saves the generated result to a new text file inside `llama31_8b_generated_answers/`.
    """
    # ---- Interpret the GPU device from args.device ----
    device_id = args.device  # e.g., 0
    device = f"cuda:{device_id}"  # "cuda:0", "cuda:1", etc.

    # ---- Load tokenizer and model onto specified device ----
    local_model_path = "hf_models/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct/"  
    print(f"Loading model from {local_model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation='sdpa'
    ).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    gen_answer_time = []

    # ---- List all sample_*.txt files in file_dir ----
    file_list = sorted(
        [f for f in os.listdir(args.file_dir) if f.startswith("sample_") and f.endswith(".txt")]
    )
    print(f"Found {len(file_list)} sample files in {args.file_dir}")

    # Create output directory if not exists
    output_dir = os.path.join(args.file_dir, "llama31_8b_generated_answers")
    os.makedirs(output_dir, exist_ok=True)
    post_str = """
    Answer the question, also tell me if you used the provided knowledge above.
    Now think step by step and answer this question. Use the following template for your final answers:
    <Some thinking process here>
    <Solution>
    1. The answer is: <Your answer here>
    2. I used the provided knowledge: <Yes/No>, followed by your explanation
    """
    # ---- Iterate through each file, generate response, and save ----
    s = time.time()
    cnt = 0
    for idx, fname in enumerate(file_list):
        file_path = os.path.join(args.file_dir, fname)
        out_file = os.path.join(output_dir, fname)

        print(colored(f"Processing file {idx+1}/{len(file_list)}: {fname}", 'yellow'))

        # Read the user's text from the file
        with open(file_path, "r", encoding="utf-8") as f:
            user_text = f.read().strip()

        # Construct the messages for Llama-3.1-8B-Instruct
        messages = [
            {"role": "user", "content": user_text +"\n" + post_str}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


        # Prepare input IDs correctly
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Generate text
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=500,
                    do_sample=False,
                    num_beams=5,
                    top_p=1.0,
                    repetition_penalty=1.25,
                    length_penalty=1.2
                )
        except torch.cuda.OutOfMemoryError:
            print(colored(f"Out of memory error occurred, skipping this file:{fname}", 'red'))
            torch.cuda.empty_cache()
            continue
        cnt += 1 
        # Decode generated output
        raw_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # print(f"Raw text:\n{raw_text}")

        # Save the model's generated answer to a new file
        
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(colored(f"Saved output to {out_file}\n", 'green'))
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='Which GPU device to use (0-based index).')
    parser.add_argument('--file_dir', type=str, required=True, help='Directory containing sample_xxx.txt files.')
    args = parser.parse_args()

    chat(args)




