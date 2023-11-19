TZ=America/New_York date;
python3 run_exp.py --method dedier -d CUB --model_type resnet18 --device 1 \
--teacher_type resnet50 --teacher_fname resnet50_CUB_GDRO.pt --batch_size 64;
TZ=America/New_York date;
python3 run_exp.py --method dedier -d CelebA --model_type resnet18 --device 1 \
--teacher_type resnet50 --teacher_fname resnet50_CelebA_GDRO.pt --batch_size 64;
TZ=America/New_York date;