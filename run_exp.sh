# python run_exp.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 128 --weight_decay 0.0001 --model resnet50 --n_epochs 50 --reweight_groups --robust --gamma 0.1 --generalization_adjustment 0 --log_every 50
python3 run_exp.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.0001 --lr 0.0001 --batch_size 128 --n_epochs 50 --save_best --save_last --log_every 50

# python3 run_exp.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.0001 --lr 0.001 --batch_size 128 --n_epochs 300 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --log_every 10
# python3 run_exp.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.0001 --lr 0.001 --batch_size 128 --n_epochs 300 --save_best --save_last --log_every 10

# TZ=IST-5:30 date; python3 run_exp.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.0001 --lr 0.0001 --batch_size 128 --n_epochs 50 --save_step 10 --log_every 50 --save_best --save_last > /dev/null ; TZ=IST-5:30 date;
# TZ=IST-5:30 date; python3 run_exp.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet18 --teacher resnet50 --weight_decay 0.0001 --lr 0.0001 --batch_size 200 --n_epochs 50 --save_step 10 --log_every 50 --save_best --save_last > /dev/null ; TZ=IST-5:30 date

python3 test_model.py  -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model_path logs/CUB/resnet18_0/best_model.pth --batch_size 128 --save_best --save_last --log_every 50 --lr 0.001 --weight_decay 0.0001


# TZ=IST-5:30 date; python3 run_exp.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --batch_size 128 --save_best --save_last > /dev/null; TZ=IST-5:30 date;
# TZ=IST-5:30 date; python3 run_exp.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet18 --batch_size 128 --save_best --save_last > /dev/null; TZ=IST-5:30 date;
# TZ=IST-5:30 date; python3 run_exp.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet18 --teacher resnet50 --batch_size 128 --save_best --save_last > /dev/null; TZ=IST-5:30 date;

# TZ=IST-5:30 date; python3 run_exp.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --batch_size 128 --save_best --save_last > /dev/null; TZ=IST-5:30 date;
# TZ=IST-5:30 date; python3 run_exp.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet18 --batch_size 128 --save_best --save_last > /dev/null; TZ=IST-5:30 date;
# TZ=IST-5:30 date; python3 run_exp.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet18 --teacher resnet50 --batch_size 128 --save_best --save_last > /dev/null; TZ=IST-5:30 date;