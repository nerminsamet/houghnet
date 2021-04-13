
# train
python src/main.py ctdet --houghnet --region_num 9 --vote_field_size 17 --exp_id coco_hg104_extremenet --arch hourglass --batch_size 29 --master_batch 5 --lr 2.5e-4 --gpus 0,1,2,3 --num_workers 16 --coco_path $COCO_PATH --num_epochs 50 --lr_step 40 --load_model ./models/ExtremeNet_500000.pth

###  for testing your own trainings, please remove '--model_v1'
# test
python src/test.py ctdet --houghnet --region_num 9 --vote_field_size 17 --exp_id coco_hg104_extremenet --arch hourglass --keep_res --resume --load_model ./models/ctdet_coco_hg104_extremenet.pth --coco_path $COCO_PATH --model_v1
# flip test
python src/test.py ctdet --houghnet --region_num 9 --vote_field_size 17 --exp_id coco_hg104_extremenet --arch hourglass --keep_res --resume --load_model ./models/ctdet_coco_hg104_extremenet.pth --coco_path $COCO_PATH --flip_test --model_v1
# multi scale test
python src/test.py ctdet --houghnet --region_num 9 --vote_field_size 17 --exp_id coco_hg104_extremenet --arch hourglass --keep_res --resume --load_model ./models/ctdet_coco_hg104_extremenet.pth --coco_path $COCO_PATH --flip_test --test_scales 0.6,0.8,1,1.2,1.5,1.8 --model_v1



