
### Segmentation Model with Voting

# train
python src/main.py ctseg --dataset coco_seg --houghnet --region_num 9 --vote_field_size 17 --exp_id ctseg_coco_resdcn_101_light --arch resdcn_101 --master_batch_size 5 --batch_size 32 --lr_step 50 --num_epochs 80 --lr 1.25e-4 --gpus 0,1,2,3 --num_workers 16 --coco_path $COCO_PATH
# test
python src/test.py ctseg --dataset coco_seg --houghnet --region_num 9 --vote_field_size 17 --exp_id ctseg_coco_resdcn_101_light --arch resdcn_101 --keep_res --resume --load_model ./models/ctseg_coco_resdcn101_light.pth --coco_path $COCO_PATH

