import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import multiprocessing as mp

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None, max_len=None):
    # truncate
    max_len = maxlen_map.get(model) if max_len is None else max_len
    assert max_len is not None, f"Model {model} not found in model2maxlen.json or --max_len is {max_len}."

    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
    tries = 0
    if model in model_map:
        model = model_map[model]
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, args):
    model = args.model
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model_map.get(model, model), trust_remote_code=True)
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )
    outputs = []
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=1024, max_len=args.max_len)
        else:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128, max_len=args.max_len)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128, max_len=args.max_len)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        outputs.append(item)
    
    return outputs

def aggreate_results(output_dir: str):
    files = os.listdir(output_dir)
    output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"]
    compensated = False

    for file in files:
        filename = os.path.join(output_dir, file)
        try:
            pred_data = json.load(open(filename, encoding='utf-8'))
        except Exception as e:
            pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
        easy, hard, short, medium, long = 0, 0, 0, 0, 0
        easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
        for pred in pred_data:
            acc = int(pred['judge'])
            if compensated and pred["pred"] == None:
                acc = 0.25
            if pred["difficulty"] == "easy":
                easy += 1
                easy_acc += acc
            else:
                hard += 1
                hard_acc += acc

            if pred['length'] == "short":
                short += 1
                short_acc += acc
            elif pred['length'] == "medium":
                medium += 1
                medium_acc += acc
            else:
                long += 1
                long_acc += acc

        if easy == 0:
            easy = 1
        if hard == 0:
            hard = 1
        if short == 0:
            short = 1
        if medium == 0:
            medium = 1
        if long == 0:
            long = 1
            
        name = '.'.join(file.split('.')[:-1])
        output.append(name+'\t'+str(round(100*(easy_acc+hard_acc)/len(pred_data), 1))+'\t'+str(round(100*easy_acc/easy, 1))+'\t'+str(round(100*hard_acc/hard, 1))+'\t'+str(round(100*short_acc/short, 1))+'\t'+str(round(100*medium_acc/medium, 1))+'\t'+str(round(100*long_acc/long, 1)))

    open(f'{output_dir}/result.txt', 'w', encoding='utf-8').write('\n'.join(output))


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    with mp.Pool(args.n_proc) as pool:
        for i in range(args.n_proc):
            processes.append(pool.apply_async(get_pred, args=(data_subsets[i], args)))
        for p in processes:
            results = p.get()
            for item in results:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    fout.close()

    aggreate_results(args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--tokenizer", "-t", type=str, default=None)
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--base_url", "-b", type=str, default=None)
    parser.add_argument("--api_key", "-k", type=str, default=None)
    parser.add_argument("--max_len", "-l", type=int, default=None)
    args = parser.parse_args()
    main(args)