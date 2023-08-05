echo "R18-pt <- R50-pt"; python3 run_exp.py --method KD -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet18-pt --teacher resnet50-pt --batch_size 64;
TZ=IST-5:30 date;

# echo "R18-pt"; python3 run_exp.py --method ERM -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet18-pt;  # 
# TZ=IST-5:30 date;

echo "R18-pt"; python3 run_exp.py --method ERM -s confounder -d CelebA -t Blond_Hair -c Male --model resnet18-pt;  # 
TZ=IST-5:30 date;

python3 save_datasets.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert

python3 save_datasets.py -s confounder -d jigsaw -t toxicity -c identity_any --model bert-base-uncased --batch_size 24

#python run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --lr 2e-05 
#--batch_size 32 --weight_decay 0 --model bert --n_epochs 3 --reweight_groups --robust 
#--generalization_adjustment 0

python3 run_exp.py --method ERM -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --batch_size 32 

python3 run_exp.py --method ERM -s confounder -d jigsaw -t toxicity -c identity_any --model bert-base-uncased --batch_size 24

TZ=IST-5:30 date; python3 run_exp.py --method aux_wt -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet18-pt --teacher resnet50-pt_group_DRO --batch_size 64 --alpha 6 --beta 3 --n_epochs 20; TZ=IST-5:30 date; 