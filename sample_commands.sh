echo "R18-pt <- R50-pt"; python3 run_exp.py --method KD -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet18-pt --teacher resnet50-pt --batch_size 64;
TZ=IST-5:30 date;

# echo "R18-pt"; python3 run_exp.py --method ERM -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet18-pt;  # 
# TZ=IST-5:30 date;

echo "R18-pt"; python3 run_exp.py --method ERM -s confounder -d CelebA -t Blond_Hair -c Male --model resnet18-pt;  # 
TZ=IST-5:30 date;