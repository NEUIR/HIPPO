# generate responses
# multi_modal
cd ../src/generate_data
CUDA_VISIBLE_DEVICES=0 python multi_modal_generate_response.py \
--model_path # the path of hippo model \
--input_file ../../data/train_data.jsonl \
--output_file output/multi_modal_response.jsonl \
--image_path  # the path of image

# image
CUDA_VISIBLE_DEVICES=0 python image_generate_response.py \
--model_path # the path of hippo model \
--input_file ../../data/train_data.jsonl \
--output_file output/image_response.jsonl \
--image_path  # the path of image

# markdown
CUDA_VISIBLE_DEVICES=0 python markdown_generate_response.py \
--model_path  # the path of hippo model \
--input_file ../../data/train_data.jsonl \
--output_file output/image_response.jsonl

# generate dpo data
python markdown_generate_response.py \
--input_multi_modal_file output/multi_modal_output.jsonl \
--input_image_file output/image_output.jsonl \
--input_markdown_file output/markdown_output.jsonl \
--output_file output/dpo_data.jsonl