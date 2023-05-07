import numpy as np
from gwea_utils import *
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id", type=int, default=0,
        help="Dataset. 0: DBP15k/zh_en/, 1: DBP15k/ja_en/, 2: DBP15k/fr_en/, 3: SRPRS/FR_EN/, 4: SRPRS/DE_EN/, 5: MED-BBK-9K 6: D_W_15K_V2")
parser.add_argument("--use_attr", type=int, default=1, help="Use attribute information 1:True, 0:False")
parser.add_argument("--use_name", type=int, default=1, help="Use untranslated name information")
parser.add_argument("--unbaised_name", type=int, default=0, help="Use unbaised DBP15K and bert-base-multilingual-cased")
parser.add_argument("--use_trans", type=int, default=1, help="Use translated name information")
parser.add_argument("--use_stru", type=int, default=1, help="Use C_\{stru\} in multi-view OT Alignment")
parser.add_argument("--use_rel", type=int, default=1, help="Use C_\{rel\} in multi-view OT Alignment")
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--gw_ss", type=float, default=0.01)
parser.add_argument("--gw_iter", type=int, default=2000)
args = parser.parse_args()
print(args)

data_loc = 'data/'+ ['DBP15k/zh_en/','DBP15k/ja_en/','DBP15k/fr_en/', 'SRPRS/FR_EN/','SRPRS/DE_EN/','med9k/','D_W_15K_V2/'][args.dataset_id]
gwea = GWEA(data=EAData(loc=data_loc), use_attr=args.use_attr, use_trans=args.use_trans, hard_pair=args.unbaised_name)
print(len(gwea.data.seed_pair), len(gwea.test_pair))

time_st = time.time()
thre = 0.5/gwea.n
gwea.cal_cost_st(w_homo=0, w_rel=0, w_feat=args.use_name, w_attr=args.use_attr)  # stage 1: Semantic Alignment
X = gwea.ot_align()
for ii in range(args.epochs):
    print('iteration: {}, threshold: {}'.format(ii, thre))
    gwea.update_anchor(X, thre)
    gwea.rel_align(emb_w=1, seed_w=1)
    gwea.cal_cost_st(w_homo=args.use_stru, w_rel=args.use_rel, w_feat=args.use_name, w_attr=args.use_attr)  # stage 2: Multi-view OT Alignment
    X = gwea.ot_align()
gwea.update_anchor(X, thre)
if args.gw_iter > 0:
    X = gwea.gw_align(X, lr=args.gw_ss, iter=args.gw_iter)  # stage 3: Gromov-Wasserstein Refinement
time_ed = time.time()
a1, a10, mrr = test_align(X, gwea.test_pair)
p, r, f1 = gwea.update_anchor(X, 1e-5)

with open('result.txt', 'a+') as f:
    f.write('Dataset: {}; use_attr: {}; use_name: {}; use_trans: {}; use_stru: {}; use_rel: {}; GW_ss: {}; GW_iter: {} \n'.format(
        args.dataset_id, args.use_attr, args.use_name, args.use_trans, args.use_stru, args.use_rel, args.gw_ss, gwea.iters))
    f.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(a1,a10,mrr,time_ed-time_st,p,r,f1))
