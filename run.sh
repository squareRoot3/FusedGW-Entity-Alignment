printf '==========full version==========\n'>>result.txt 
python main.py --dataset_id 0 --use_attr 1 --use_name 1 --use_trans 1 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 1 --use_attr 1 --use_name 1 --use_trans 1 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 2 --use_attr 1 --use_name 1 --use_trans 1 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 3 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 4 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 5 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.02 --gw_iter 2000
python main.py --dataset_id 6 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000

printf '==========only attributes==========\n'>>result.txt 
python main.py --dataset_id 0 --use_attr 1 --use_name 0 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 1 --use_attr 1 --use_name 0 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 2 --use_attr 1 --use_name 0 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000

printf '===========only translated name information============\n'>>result.txt 
python main.py --dataset_id 0 --use_attr 0 --use_name 1 --use_trans 1 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 1 --use_attr 0 --use_name 1 --use_trans 1 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 2 --use_attr 0 --use_name 1 --use_trans 1 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000

printf '===========only untranslated name information============\n'>>result.txt 
python main.py --dataset_id 0 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 1 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000
python main.py --dataset_id 2 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 2000

printf '===========ablation study==============\n'>>result.txt 
python main.py --dataset_id 0 --use_attr 1 --use_name 1 --use_trans 1 --use_stru 0 --use_rel 1 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 0 --use_attr 1 --use_name 1 --use_trans 1 --use_stru 1 --use_rel 0 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 0 --use_attr 1 --use_name 1 --use_trans 1 --use_stru 0 --use_rel 0 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 0 --use_attr 1 --use_name 1 --use_trans 1 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 3 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 0 --use_rel 1 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 3 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 0 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 3 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 0 --use_rel 0 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 3 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 5 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 0 --use_rel 1 --gw_ss 0.02 --gw_iter 0
python main.py --dataset_id 5 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 0 --gw_ss 0.02 --gw_iter 0
python main.py --dataset_id 5 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 0 --use_rel 0 --gw_ss 0.02 --gw_iter 0
python main.py --dataset_id 5 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.02 --gw_iter 0
python main.py --dataset_id 6 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 0 --use_rel 1 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 6 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 0 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 6 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 0 --use_rel 0 --gw_ss 0.01 --gw_iter 0
python main.py --dataset_id 6 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 1 --gw_ss 0.01 --gw_iter 0


printf '==========hard version==========\n'>>result.txt 
python main.py --dataset_id 0 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 0 --gw_ss 0.01 --gw_iter 2000 --unbaised_name 1
python main.py --dataset_id 1 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 0 --gw_ss 0.01 --gw_iter 2000 --unbaised_name 1
python main.py --dataset_id 2 --use_attr 0 --use_name 1 --use_trans 0 --use_stru 1 --use_rel 0 --gw_ss 0.01 --gw_iter 2000 --unbaised_name 1
