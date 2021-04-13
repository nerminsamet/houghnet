
# train
python src/main.py ddd --houghnet --dataset kitti --kitti_split subcnn --exp_id sub --arch dla_34 --batch_size 16 --master_batch 7 --num_epochs 70 --lr_step 45,60 --gpus 0,1

###  for testing your own trainings, please remove '--model_v1'
# test
python src/test.py ddd --houghnet --dataset kitti --kitti_split subcnn --exp_id sub --arch dla_34 --resume --load_model ./models/ddd_kitti_dla34.pth --model_v1
