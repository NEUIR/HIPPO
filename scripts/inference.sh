cd src/inference
CUDA_VISIBLE_DEVICES=0 python multi_modal_inference.py \
--model_path # the path of hippo model \
--input_file ../../data/test_data.jsonl \ # the path of input file
--output_file answer_multi_modal.jsonl \ # the path of output file
--image_path  ../all_test_image # the path of image