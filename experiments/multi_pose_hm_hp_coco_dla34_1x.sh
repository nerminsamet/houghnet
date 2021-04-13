
### voting for both 'hm' and 'hm_hp' heads - please be sure that voting head is 'hm' and 'hm_hp'

# train
python src/main.py multi_pose --dataset coco_hp --houghnet --region_num 9 --vote_field_size 17 --exp_id coco_mp_voting_hm_hp --arch dla_34 --batch_size 44 --master_batch_size 8 --lr 1.75e-4 --gpus 0,1,2,3 --num_workers 16 --load_model ./models/ctdet_coco_dla_2x.pth --coco_path $COCO_PATH

###  for testing your own trainings, please remove '--model_v1'
# test
python src/test.py multi_pose --houghnet --dataset coco_hp --exp_id coco_mp_voting_hm_hp --arch dla_34 --keep_res --resume --load_model ./models/multi_pose_hm_hp_coco_dla34_1x.pth --coco_path $COCO_PATH --model_v1
# flip test
python src/test.py multi_pose --houghnet --dataset coco_hp --exp_id coco_mp_voting_hm_hp --arch dla_34 --keep_res --resume --load_model ./models/multi_pose_hm_hp_coco_dla34_1x.pth --coco_path $COCO_PATH --flip_test --model_v1
