import os
import re
import json
import argparse
import random
from tqdm import tqdm
from base_prompt import *
import NarrativeClarificationPrompting

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from jsonformer import Jsonformer  # Assuming this is a custom library you're using.


import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


json_schema_related_information = {
    "type": "object",
    "properties": {"related_information": {"type": "string"}}
    }

json_schema_narrative_clarification = {
    "type": "object",
    "properties": {"narrative_clarification": {"type": "string"}}
    }

json_schema_final_answer = {
    "type": "object",
    "properties": {
        "Correct Option": {"type": "string"},
        "choice value": {"type": "string"},
        "explanation": {"type": "string"}
        }
    }

def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt

def load_data(args):
    problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    qids = pid_splits['%s' % (args.test_split)]
    qids = qids[:args.test_number] if args.test_number > 0 else qids
    print(f"number of test problems: {len(qids)}\n")

    # pick up shot examples from the training set
    shot_qids = args.shot_qids
    train_qids = pid_splits['train']
    if shot_qids == None:
        assert args.shot_number >= 0 and args.shot_number <= 32
        shot_qids = random.sample(train_qids, args.shot_number)  # random sample
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            assert qid in train_qids  # check shot_qids
    print("training question ids for prompting: ", shot_qids, "\n")

    return problems, qids, shot_qids


def get_gpt3_result(prompt, args):
    response = openai.Completion.create(engine=args.engine,
                                        prompt=prompt,
                                        temperature=args.temperature,
                                        max_tokens=args.max_tokens,
                                        top_p=args.top_p,
                                        frequency_penalty=args.frequency_penalty,
                                        presence_penalty=args.presence_penalty,
                                        stop=["\n"])
    output = response["choices"][0]["text"].strip()

    # extract the answer
    pattern = re.compile(r'The answer is ([A-Z]).')
    res = pattern.findall(output)
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"

    return answer, output

def load_llama3_model(model_path, access_token):
    logging.info(f"Loading Model: {model_path} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token="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",
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_path, token="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")

    logging.info(f"Model: {model_path} Loaded ...")
    return model, tokenizer

def run_llama3_prompting(tokenizer, model, messages, json_schema):

    prompt = f"""
        Answer the following question.

        ### Question:
        {messages}
        """

    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_number_tokens=2000,
                            max_array_length=2000,
                            max_string_token_length=2000)
    response = jsonformer()
    return prompt, response

def run_prompting(tokenizer, model, messages, json_schema):

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_number_tokens=2000,
                            max_array_length=2000,
                            max_string_token_length=2000)
    response = jsonformer()

    return(prompt, response)

def get_llama3_result(prompt, model, tokenizer):
    json_schema = {
            "type": "object",
            "properties": {
                "Correct Option": {"type": "string"},
                }
            }  # Define your JSON schema here if needed
    print("Prompt: ",prompt)
    prompt, response = run_llama3_prompting(tokenizer, model, prompt, json_schema)
    output = response["Correct Option"]
    print("output: ",output)
    print("\n")
     # Adjusted regex pattern to capture the first letter and stop at non-letter characters (like a period or space)
    pattern = re.compile(r'^[A-Z]')  # Matches the first uppercase letter at the beginning of the string
    
    res = pattern.findall(output)
    
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"



    print("answer after regex is:",answer)
    return answer, output


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


def get_result_file(args):
    result_file = "{}/{}/{}_{}_{}_{}_seed_{}.json".format(args.output_root, args.model, args.label, args.test_split,
                                                          args.prompt_format, args.shot_number, args.seed)

    return result_file


def save_results(result_file, acc, correct, count, shot_qids, args, results, outputs):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['shot_qids'] = shot_qids
    data['args'] = vars(args)
    data['results'] = results
    data['outputs'] = outputs

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/scienceqa')
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--caption_file', type=str, default='../data/captions.json')
    parser.add_argument('--model', type=str, default='gpt3')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_number', type=int, default=10, help='GPT-3 is expensive. -1 for whole val/test set')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
                        ],
                        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=3, help='Number of n-shot training examples.')
    parser.add_argument('--shot_qids', type=list, default=None, help='Question indexes of shot examples')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    # Load the llama3 model and tokenizer
    model, tokenizer = load_llama3_model(args.model, "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")

    problems, qids, shot_qids = load_data(args)  # probelms, test question ids, shot example ids


    result_file = get_result_file(args)

    if not os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(result_file))
        acc = check_point['acc']
        correct = check_point['correct']
        results = check_point['results']
        outputs = check_point['outputs']
        print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%")
    else:
        correct = 0
        results = {}
        outputs = {}

    for i, qid in enumerate(qids[:10]):
        print("##New Question##")
        if qid in results:
            continue

        choices = problems[qid]["choices"]
        answer = problems[qid]["answer"]  # 0, 1, ..., 4
        label = args.options[answer]  # 'A', ..., 'E'
        # print("labels: ",label)
        # print("Choices: ",choices)
        # print("answer: ",answer)

        question = problems[qid]['question']
        choice = get_choice_text(problems[qid], args.options)

        import pprint
        # STEP 1    
        messages_1 = NarrativeClarificationPrompting.meta_narrative_prompting.llama_3_prompt_creation_step_1(question)
        prompt_1, response_1 = run_prompting(tokenizer, model, messages_1, json_schema_related_information)
        print("Step 1 Prompt: ")
        pprint.pprint(messages_1)
        print("Response 1")
        pprint.pprint(response_1)
        # STEP 2
        messages_2 = NarrativeClarificationPrompting.meta_narrative_prompting.llama_3_prompt_creation_step_2(question, response_1)
        prompt_2, response_2 = run_prompting(tokenizer, model, messages_2, json_schema_narrative_clarification)
        print("Step 2 Prompt: ")
        pprint.pprint(messages_2)
        print("Response 2")
        pprint.pprint(response_2)
        # STEP 3
        messages_3 = NarrativeClarificationPrompting.meta_narrative_prompting.llama_3_prompt_creation_step_3(question, choice, response_2)
        prompt_3, response_3 = run_prompting(tokenizer, model, messages_3, json_schema_final_answer)
        print("Step 3 Prompt: ")
        pprint.pprint(messages_3)
        print("Response 3")
        pprint.pprint(response_3)
   

        output = response_3["Correct Option"]
        print(" \n ###final output: ",output)
        print("\n")
        # Adjusted regex pattern to capture the first letter and stop at non-letter characters (like a period or space)
        pattern = re.compile(r'^[A-Z]')  # Matches the first uppercase letter at the beginning of the string
        
        res = pattern.findall(output)
        
        if len(res) == 1:
            predicted_option = res[0]  # 'A', 'B', ...
        else:
            predicted_option = "FAILED"

        #prompt = build_prompt(problems, shot_qids, qid, args)  # Assuming build_prompt returns messages

        # Generate prediction using llama3
        pred_idx = get_pred_idx(predicted_option, choices, args.options)

        results[qid] = pred_idx
        outputs[qid] = output
        if pred_idx == answer:
            correct += 1

        acc = correct / len(results) * 100

        if args.debug or i < 3:
            print("\n")
            # print("##################################")
            print("# labeled answer:", label)
            print("# predicted answer:", answer)
            print("# predicted index:", pred_idx)
            print("# predicted output:", output)

        if (i + 1) % args.save_every == 0 or (i + 1) == len(qids):
            print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
            save_results(result_file, acc, correct, i + 1, shot_qids, args, results, outputs)
