
# Base Model
# train
python src/main.py ctdet --houghnet --region_num 17 --vote_field_size 65 --exp_id coco_resdcn_101 --arch resdcn_101 --batch_size 44 --master_batch 8 --lr 1.75e-4 --gpus 0,1,2,3 --num_workers 16 --coco_path $COCO_PATH

###  for testing your own trainings, please remove '--model_v1'
# test
python src/test.py ctdet --houghnet --region_num 17 --vote_field_size 65 --exp_id coco_resdcn_101 --arch resdcn_101 --keep_res --resume --load_model ./models/ctdet_coco_resdcn101.pth --coco_path $COCO_PATH --model_v1
# flip test
python src/test.py ctdet --houghnet --region_num 17 --vote_field_size 65 --exp_id coco_resdcn_101 --arch resdcn_101 --keep_res --resume --load_model ./models/ctdet_coco_resdcn101.pth --coco_path $COCO_PATH --flip_test --model_v1
# multi scale test
python src/test.py ctdet --houghnet --region_num 17 --vote_field_size 65 --exp_id coco_resdcn_101 --arch resdcn_101 --keep_res --resume --load_model ./models/ctdet_coco_resdcn101.pth --coco_path $COCO_PATH --flip_test --test_scales 0.6,0.8,1,1.2,1.5,1.8 --model_v1

