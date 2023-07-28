echo "R18-pt <- R50-pt"; python3 run_exp.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet18 --model_state pretrained --teacher resnet50-pt --batch_size 64 > /dev/null;  # 
TZ=IST-5:30 date;

# echo "R18-pt"; python3 run_exp.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet18 --model_state pretrained > /dev/null;  # 
# TZ=IST-5:30 date;

echo "R18-pt"; python3 run_exp.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet18 --model_state pretrained > /dev/null;  # 
TZ=IST-5:30 date;