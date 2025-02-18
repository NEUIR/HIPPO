import torch
import os
import json
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import shortuuid
from vllm import LLM, SamplingParams
import pandas as pd
import argparse
from prompt.multi_modal_prompt import wtq_template, tabmwp_template, tat_qa_template, hitab_template, fetaqa_template, infotabs_template, tabfact_template
def get_markdown_table(table, tokens, tokenizer):
    header = table["header"]
    rows = table["rows"]

    df = pd.DataFrame(rows, columns=header)

    markdown_table = df.to_markdown(index=False)

    token_count = len(tokenizer.tokenize(markdown_table))

    while token_count > tokens and len(rows) > 1:
        rows = rows[:-1]
        df = pd.DataFrame(rows, columns=header)
        markdown_table = df.to_markdown(index=False)
        token_count = len(tokenizer.tokenize(markdown_table))

    return markdown_table

def get_markdown_table_infotabs(table, tokens, tokenizer):
    df_vertical = pd.DataFrame({
        "Attribute": table.keys(),
        "Value": ["\n".join(values) if isinstance(values, list) else values for values in table.values()]
    })

    markdown_table = df_vertical.to_markdown(index=False)

    return markdown_table

def main(model_path, input_file, output_file, image_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=1,
        max_model_len=8192,
        tensor_parallel_size=1,
        dtype="bfloat16",
    )

    with open(input_file, "r", encoding="utf-8") as f:
        questions = [json.loads(q) for q in f]

    count = 0

    for line in tqdm(questions):
        count += 1

        if count <= 4500:
            continue

        idx = line["question_id"]
        image_file = line["image"]
        question = line["question"]
        table = line["table"]
        if "InfoTabs" in idx:
            markdown_table = get_markdown_table_infotabs(table, 7000, tokenizer)
        else:
            markdown_table = get_markdown_table(table, 7000, tokenizer)
        category = line["category"]

        template_mapping = {
            "WTQ_for_TQA": wtq_template,
            "TABMWP_for_TQA": tabmwp_template,
            "TAT-QA_for_TQA": tat_qa_template,
            "HiTab_for_TQA": hitab_template,
            "FeTaQA_for_TQA": fetaqa_template,
            "InfoTabs_for_TFV": infotabs_template,
            "TabFact_for_TFV": tabfact_template
        }

        template = next((template_mapping[key] for key in template_mapping if key in category), None)

        if "TFV" in category:
            cur_prompt = f"Markdown Table:\n{markdown_table}{template}Statement: {question}"
        else:
            cur_prompt = f"Markdown Table:\n{markdown_table}{template}Question: {question}"

        try:
            image = Image.open(os.path.join(image_path, image_file))
        except (FileNotFoundError, IOError) as e:
            tqdm.write(f"Failed to open image {image_file}: {e}")
            continue

        messages = [{
            "role": "user",
            "content": "(<image>./</image>)" + "\n" + cur_prompt
        }]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        }

        # 2.6
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

        sampling_params = SamplingParams(
            stop_token_ids=stop_token_ids,
            use_beam_search=True,
            temperature=0,
            best_of=3,
            max_tokens=2048
        )

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        outputs = outputs[0].outputs[0].text

        ans_id = shortuuid.uuid()
        res = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": "MiniCPM-V-2.6",
            "metadata": {},
            "category": category
        }

        with open(output_file, "a") as f:
            json.dump(res, f)
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the table image")
    args = parser.parse_args()

    main(args.model_path, args.input_file, args.output_file, args.image_path)