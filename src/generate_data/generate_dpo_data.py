import json
import os
import re
from collections import Counter
import argparse

def extract_tqa_answer_list(model_output):
    """
    Extract the answer list from the model output to compute accuracy
    """
    model_output = model_output.replace('\n',' ')
    ret = re.match('.*({[\"\']answer[\"\']\:.*}).*',model_output)
    if ret is not None:
        answer_str = ret.group(1)
        try:
            answer_str = re.sub('[\"\']+',"\"",answer_str)
            answer_item = eval(answer_str)
            predicted_answer = answer_item['answer']
            if type(predicted_answer) != list and type(predicted_answer) == str:
                predicted_answer = [predicted_answer]
            elif type(predicted_answer) != list and type(predicted_answer) in [float,int]:
                predicted_answer = [str(predicted_answer)]
            else:
                pass
        # The answer is considered to be wrong if we can not extract answer list from the json str
        except:
            predicted_answer = []
        return predicted_answer
    else:
        return []


def main(input_multi_modal_file, input_image_file, input_markdown_file, output_file):
    try:
        multi_modal_file_path = input_multi_modal_file
        image_file_path = input_image_file
        markdown_file_path = input_markdown_file

        with open(multi_modal_file_path, 'r', encoding='utf-8') as f:
            multi_modal_data = [json.loads(line) for line in f]
        with open(image_file_path, 'r', encoding='utf-8') as f:
            image_data = [json.loads(line) for line in f]
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_data = [json.loads(line) for line in f]

        image_data_dict = {record['question_id']: record for record in image_data}
        markdown_data_dict = {record['question_id']: record for record in markdown_data}

        processed_data = []
        for multi_modal_record in multi_modal_data:
            try:
                question_id = multi_modal_record['question_id']

                image_record = image_data_dict.get(question_id, {})
                markdown_record = markdown_data_dict.get(question_id, {})

                if not image_record or not markdown_record:
                    continue

                multi_modal_record['text'].extend(image_record.get('text', []))
                multi_modal_record['text'].extend(markdown_record.get('text', []))

                record = multi_modal_record

                correct_texts = []
                incorrect_texts = []

                correct_answer = record.get('answer', [])

                for text in record.get('text', []):
                    predicted_answer = extract_tqa_answer_list(text)
                    if predicted_answer == correct_answer:
                        correct_texts.append(text)
                    else:
                        incorrect_texts.append(text)

                if correct_texts:
                    correct_counter = Counter(correct_texts)
                    most_common_correct_text = correct_counter.most_common(1)[0][0]
                    record['correct_text'] = most_common_correct_text
                else:
                    record['correct_text'] = None
                    continue

                if incorrect_texts:
                    incorrect_answers = [tuple(extract_tqa_answer_list(text)) for text in incorrect_texts]
                    answer_counter = Counter(incorrect_answers)
                    most_common_answer = answer_counter.most_common(1)[0][0]
                    candidate_texts = [
                        text for text, answer in zip(incorrect_texts, incorrect_answers)
                        if answer == most_common_answer
                    ]
                    candidate_counter = Counter(candidate_texts)
                    most_common_incorrect_text = candidate_counter.most_common(1)[0][0]
                    record['incorrect_text'] = most_common_incorrect_text
                else:
                    record['incorrect_text'] = None
                    continue


                processed_data.append(record)

            except Exception as e:
                print(f"Error processing record {multi_modal_record.get('question_id')}: {e}")
                continue

        new_file_path = output_file
        with open(new_file_path, 'w', encoding='utf-8') as f_new:
            for record in processed_data:
                prompt = record['prompt']
                correct_text = record['correct_text']
                incorrect_text = record['incorrect_text']
                image_path = record['image_path']
                new_record = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"<image>\n{prompt}"},
                        {"role": "assistant", "content": correct_text}
                    ],
                    "rejected_response": incorrect_text,
                    "images": [f"{image_path}"]
                }
                f_new.write(json.dumps(new_record, ensure_ascii=False) + '\n')

    except Exception as e:
        print(e)
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("--input_multi_modal_file", type=str, required=True, help="Path to the multi modal file.")
    parser.add_argument("--input_image_file", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--input_markdown_file", type=str, required=True, help="Path to the markdown file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()

    main(args.input_multi_modal_file, args.input_image_file, args.input_markdown_file, args.output_file)