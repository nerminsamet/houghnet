# train
python src/main.py ctdet --houghnet --region_num 17 --vote_field_size 65 --exp_id coco_res_101 --arch res_101 --batch_size 44 --master_batch 8 --lr 1.75e-4 --gpus 0,1,2,3 --num_workers 16 --coco_path $COCO_PATH
# test
python src/test.py ctdet --houghnet --region_num 17 --vote_field_size 65 --exp_id coco_res101 --arch res_101 --keep_res --resume --load_model ./models/ctdet_coco_res101.pth --coco_path $COCO_PATH
# flip test
python src/test.py ctdet --houghnet --region_num 17 --vote_field_size 65 --exp_id coco_res101 --arch res_101 --keep_res --resume --load_model ./models/ctdet_coco_res101.pth --coco_path $COCO_PATH --flip_test
# multi scale test
python src/test.py ctdet --houghnet --region_num 17 --vote_field_size 65 --exp_id coco_res101 --arch res_101 --keep_res --resume --load_model ./models/ctdet_coco_res101.pth --coco_path $COCO_PATH --flip_test --test_scales 0.6,0.8,1,1.2,1.5,1.8
