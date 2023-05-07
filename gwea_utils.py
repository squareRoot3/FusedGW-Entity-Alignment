from sentence_transformers import SentenceTransformer
import numpy as np
import dgl
import torch
from bidict import bidict
from copy import deepcopy
import os
# from simcse import SimCSE


class KG:
    def __init__(self):
        self.rel_ids = bidict()
        self.ent_ids = bidict()
        self.er_e = dict()
        self.ee_r = dict()
        self.edges = set()

    def construct_graph(self):
        ent_graph = dgl.graph(list(self.edges))
        self.ent_graph = dgl.to_bidirected(ent_graph).to_simple()


class EAData:
    def __init__(self, loc, bi=True):
        self.kg = [KG(), KG()]
        self.seed_pair = bidict()
        self.test_pair = bidict()
        self.loc = loc
        if 'DBP' in loc or "SRPRS" in loc:
            self.load_dbp(bi)
        elif 'med' in loc:
            self.load_med(bi)
        else:
            self.load_DW15K(bi)
        self.test_pair.update(self.seed_pair)
        self.seed_pair = bidict()

    def load_DW15K(self, bi):
        with open(self.loc+'ent_links', 'r', encoding='UTF-8') as f:
            knt = 0
            t = f.readlines()
            n1 = len(t)
            for line in t:
                head, tail = line.strip().split('\t')
                self.kg[0].ent_ids[knt]=head.split('resource/')[-1].replace('_', ' ')
                self.kg[1].ent_ids[knt+n1]=tail.split('entity/')[-1].replace('_', ' ')
                self.test_pair[knt] = knt+n1
                knt += 1
        knt = 0
        for i in range(2):
            with open(self.loc+'rel_triples_{}'.format(i+1), 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    head, rel, tail = line.strip().split('\t')
                    if i==0:
                        head = head.split('resource/')[-1].replace('_', ' ')
                        tail = tail.split('resource/')[-1].replace('_', ' ')
                        rel = rel.split('ontology/')[-1]
                    else:
                        head = head.split('entity/')[-1].replace('_', ' ')
                        tail = tail.split('entity/')[-1].replace('_', ' ')
                        rel = rel.split('entity/')[-1] 
                    if rel not in self.kg[i].rel_ids.values():
                        self.kg[i].rel_ids[knt]=rel
                        knt+=1
                    head = self.kg[i].ent_ids.inv[head]
                    tail = self.kg[i].ent_ids.inv[tail]
                    rel = self.kg[i].rel_ids.inv[rel]
                    self.kg[i].edges.add((head, tail))
                    self.kg[i].er_e[(head, rel)] = tail
                    self.kg[i].ee_r[(head, tail)] = rel
                    if bi:
                        self.kg[i].er_e[(tail, rel)] = head
                self.kg[i].construct_graph()

    def load_med(self, bi):
        with open(self.loc+'ent_links', 'r', encoding='UTF-8') as f:
            knt = 0
            t = f.readlines()
            n1 = len(t)
            for line in t:
                head, tail = line.strip().split('\t')
                self.kg[0].ent_ids[knt]=head.split('resource/')[-1].replace('_', ' ')
                self.kg[1].ent_ids[knt+n1]=tail.split('resource/')[-1].replace('_', ' ')
                self.test_pair[knt] = knt+n1
                knt += 1
        knt = 0
        for i in range(2):
            with open(self.loc+'rel_triples_{}'.format(i+1), 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    head, rel, tail = line.strip().split('\t')
                    head = head.split('resource/')[-1].replace('_', ' ')
                    tail = tail.split('resource/')[-1].replace('_', ' ')
                    rel = rel.split('ontology/')[-1]
                    if rel not in self.kg[i].rel_ids.values():
                        self.kg[i].rel_ids[knt]=rel
                        knt+=1
                    head = self.kg[i].ent_ids.inv[head]
                    tail = self.kg[i].ent_ids.inv[tail]
                    rel = self.kg[i].rel_ids.inv[rel]
                    self.kg[i].edges.add((head, tail))
                    self.kg[i].er_e[(head, rel)] = tail
                    self.kg[i].ee_r[(head, tail)] = rel
                    if bi:
                        self.kg[i].er_e[(tail, rel)] = head
                self.kg[i].construct_graph()

    def load_dbp(self, bi, trans=False):
        for i in range(2):
            with open(self.loc+'rel_ids_{}'.format(i+1), encoding='UTF-8') as f:
                for line in f.readlines():
                    ids, rel = line.strip().split('\t')
                    self.kg[i].rel_ids[int(ids)] = rel.split('property/')[-1]

            with open(self.loc+'ent_ids_{}'.format(i+1), encoding='UTF-8') as f:
                if trans==True and i==0:
                    self.kg[i].ent_trans = {}
                    with open(self.loc+'translated_google.txt', encoding='UTF-8') as f2:
                        for line1, line2 in zip(f.readlines(), f2.readlines()):
                            ids, ent = line1.strip().split('\t')
                            ent_trans = line2.strip()
                            self.kg[i].ent_ids[int(ids)] = ent.split('resource/')[-1].replace('_', ' ')
                            self.kg[i].ent_trans[int(ids)] = ent_trans
                else:
                    for line in f.readlines():
                        ids, ent = line.strip().split('\t')
                        self.kg[i].ent_ids[int(ids)] = ent.split('resource/')[-1].replace('_', ' ')

            with open(self.loc+'triples_{}'.format(i+1), encoding='UTF-8') as f:
                for line in f.readlines():
                    head, rel, tail = line.strip().split('\t')
                    head, rel, tail = int(head), int(rel), int(tail)
                    # head, tail = self.kg[i].ent_ids.inv[head], self.kg[i].ent_ids.inv[tail]
                    self.kg[i].edges.add((head, tail))
                    self.kg[i].er_e[(head, rel)] = tail
                    self.kg[i].ee_r[(head, tail)] = rel
                    if bi:
                        self.kg[i].er_e[(tail, rel)] = head
                self.kg[i].construct_graph()

        with open(self.loc+'sup_pairs', 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                head, tail = line.strip().split('\t')
                self.seed_pair[int(head)] = int(tail)

        with open(self.loc+'ref_pairs', 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                head, tail = line.strip().split('\t')
                self.test_pair[int(head)] = int(tail)
        if os.path.exists(self.loc+'hard_pairs.txt'):
            self.hard_pair = {}
            with open(self.loc+'hard_pairs.txt', 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    head, tail = line.strip().split('\t')
                    self.hard_pair[int(head)] = int(tail)
        # self.test_entities = (set(self.test_pair.keys()), set(self.test_pair.values()))
        # self.cal_ent_emb(self.loc, encoder_name='bert-base-multilingual-cased')
        # self.cal_rel_emb(self.loc, encoder_name='bert-base-multilingual-cased')
        # self.cal_ent_emb(self.loc,'princeton-nlp/sup-simcse-roberta-base', trans)
        # self.cal_ent_emb(self.loc,'sentence-transformers/all-mpnet-base-v2', trans)        

    def cal_rel_emb(self,loc,encoder_name='sentence-transformers/LaBSE'):
        BERTencoder = SentenceTransformer(encoder_name)
        rel_num = len(self.kg[0].rel_ids) + len(self.kg[1].rel_ids)
        self.rel_emb = np.zeros([rel_num, 768])
        for i in range(2):
            kg = self.kg[i]
            for idx, rel in kg.rel_ids.items():
                self.rel_emb[idx] = BERTencoder.encode(rel)
        np.save('{}/rel_emb_bert'.format(loc),self.rel_emb)

    def cal_ent_emb(self,loc,encoder_name='sentence-transformers/LaBSE', trans=False):
        # if trans:
            # encoder = SimCSE(encoder_name)
        # else:
        encoder = SentenceTransformer(encoder_name)
        ent_num = len(self.kg[0].ent_ids) + len(self.kg[1].ent_ids)
        self.ent_emb = np.zeros([ent_num, 768])
        for i in range(2):
            kg = self.kg[i]
            if i == 0 and trans:
                for idx, ent in kg.ent_trans.items():
                    self.ent_emb[idx] = encoder.encode(ent)
            else:
                for idx, ent in kg.ent_ids.items():
                    self.ent_emb[idx] = encoder.encode(ent)
        np.save('{}/ent_emb_bert'.format(loc),self.ent_emb)


def NeuralSinkhorn(cost, p_s=None, p_t=None, trans=None, beta=0.1, outer_iter=20):
    if p_s is None:
        p_s = torch.ones([cost.shape[0],1],device=cost.device)/cost.shape[0]
    if p_t is None:
        p_t = torch.ones([cost.shape[1],1],device=cost.device)/cost.shape[1]
    if trans is None:
        trans = p_s @ p_t.T
    a = torch.ones([cost.shape[0],1],device=cost.device)/cost.shape[0]
    cost_new = torch.exp(-cost / beta)
    for oi in range(outer_iter):
        kernel = cost_new * trans
        b = p_t / (kernel.T@a)
        a = p_s / (kernel@b)
        trans = (a @ b.T) * kernel
    return trans


def test_align(pred, test_pair):
    ind = (-pred).argsort(axis=1)
    ind = ind.cpu().numpy()
    a1, a10, mrr = 0, 0, 0
    for k, v in test_pair.items():
        rank=np.where(ind[k]==v)[0][0]+1
        if rank==1:
            a1+=1
        if rank<=10:
            a10+=1
        mrr+=1/rank
    a1 /= len(test_pair)
    a10 /= len(test_pair)
    mrr /= len(test_pair)
    print('H@1 %.1f%% H@10 %.1f%% MRR %.1f%%' % (a1*100, a10*100, mrr*100))
    return a1, a10, mrr


class GWEA():
    def __init__(self, data, use_attr=True, use_trans=False, hard_pair=False):
        self.iters = 0
        self.data = data
        self.candi = self.data.test_pair.copy()
        self.graph1 = self.data.kg[0].ent_graph
        self.graph2 = self.data.kg[1].ent_graph
        self.rel_list = [list(self.data.kg[0].rel_ids),list(self.data.kg[1].rel_ids)]
        self.ent_ids1 = bidict()
        self.ent_ids2 = bidict()
        if 'D_W' not in self.data.loc:
            self.rel_emb = np.load(self.data.loc+'rel_emb.npy')
        if use_trans:
            self.ent_emb = torch.tensor(np.load(self.data.loc+'ent_emb_google.npy')).float()
            self.ent_emb = self.ent_emb/((self.ent_emb**2).sum(1)**0.5)[:,None] 
        elif hard_pair:
            self.ent_emb = torch.tensor(np.load(self.data.loc+'ent_emb_bert.npy')).float()
        else:
            self.ent_emb = torch.tensor(np.load(self.data.loc+'ent_emb.npy')).float()
        if use_attr:
            self.attr_emb = torch.tensor(np.load(self.data.loc+'attr_emb.npy')).float()
            # self.attr_emb = self.attr_emb/((self.attr_emb**2).sum(1)**0.5)[:,None] 
        self.ent_emb = self.ent_emb/((self.ent_emb**2).sum(1)**0.5)[:,None] 
        # self.rel_emb = self.rel_emb/((self.rel_emb**2).sum(1)**0.5)[:,None]
        # todo normalize embedding in advance
        rand_ind = np.random.permutation(len(data.test_pair))
        self.test_pair = {}
        for i, ind in enumerate(rand_ind):
            self.test_pair[i] = ind
        for i, (ent1, ent2) in enumerate(self.candi.items()):
            self.ent_ids1[i] = ent1
            self.ent_ids2[self.test_pair[i]] = ent2
        if hard_pair:
            self.hard_pair = {}
            for k, v in data.hard_pair.items():
                self.hard_pair[self.ent_ids1.inv[k]] = self.ent_ids2.inv[v]
        self.n = len(self.ent_ids1)
        self.ent_ids2 = bidict(sorted(self.ent_ids2.items()))
        self.cost_s = self.graph1.subgraph(list(self.ent_ids1.values())).adj().cuda()
        self.cost_t = self.graph2.subgraph(list(self.ent_ids2.values())).adj().cuda()
        self.cost_st_feat = 1-self.ent_emb[list(self.ent_ids1.values())]@self.ent_emb[list(self.ent_ids2.values())].T
        if use_attr:
            self.cost_st_attr = 1-self.attr_emb[list(self.ent_ids1.values())]@self.attr_emb[list(self.ent_ids2.values())].T

    def cal_cost_st(self, w_homo=1, w_rel=1, w_feat=1, w_attr=1, M=20):
        self.cost_st = torch.zeros(self.n, self.n)
        if w_homo>0:
            cost_st_homo = self.cal_cost_st_homo()
            cost_st_homo = cost_st_homo#+cost_st_homo.T)
            cost_st_homo[cost_st_homo>M]=M
            cost_st_homo = 1-cost_st_homo/cost_st_homo.max()
            self.cost_st += w_homo*cost_st_homo
        if w_rel>0:
            cost_st_rel = self.cal_cost_st_rel(bi=True)
            cost_st_rel = cost_st_rel#+cost_st_rel.T
            cost_st_rel[cost_st_rel>M]=M
            cost_st_rel = 1-cost_st_rel/cost_st_rel.max()
            self.cost_st += w_rel*cost_st_rel
        if w_feat>0:
            self.cost_st += w_feat*self.cost_st_feat
        if w_attr>0:
            self.cost_st += w_attr*self.cost_st_attr
        self.cost_st = self.cost_st.cuda()

    def cal_cost_st_homo(self):
        cost = torch.zeros(self.n,self.n)
        for i, (ent1, ent2) in enumerate(self.anchor.items()):
            idx1, idx2 = [],[]
            for ne1 in self.graph1.predecessors(ent1).numpy():
                if ne1 in self.ent_ids1.values():
                    idx1.append(self.ent_ids1.inv[ne1])
            for ne2 in self.graph2.predecessors(ent2).numpy():
                if ne2 in self.ent_ids2.values():
                    idx2.append(self.ent_ids2.inv[ne2])
            if len(idx1)>0 and len(idx2) > 0:
                idxx = np.ix_(idx1,idx2)
                cost[idxx] += 1
        return cost

    def cal_cost_st_rel(self, bi=True):
        cost = torch.zeros(self.n,self.n)
        for (head, rel), tail in self.data.kg[0].er_e.items():
            if head in self.anchor.keys() and rel in self.r2r.keys() and tail in self.ent_ids1.values():
                head2 = self.anchor[head]
                rel2 = self.r2r[rel]
                if head2 in self.anchor.values() and (head2, rel2) in self.data.kg[1].er_e.keys():
                    tail2 = self.data.kg[1].er_e[(head2, rel2)]
                    if tail2 in self.ent_ids2.values():
                        cost[self.ent_ids1.inv[tail]][self.ent_ids2.inv[tail2]] += 1
        if bi:
            for (head, rel), tail in self.data.kg[1].er_e.items():
                if head in self.anchor.values() and rel in self.r2r.values() and tail in self.ent_ids2.values():
                    head2 = self.anchor.inv[head]
                    rel2 = self.r2r[rel]
                    if head2 in self.anchor.keys() and (head2, rel2) in self.data.kg[0].er_e.keys():
                        tail2 = self.data.kg[0].er_e[(head2, rel2)]
                        if tail2 in self.ent_ids1.values():
                            cost[self.ent_ids1.inv[tail2]][self.ent_ids2.inv[tail]] += 1
        return cost

    def update_anchor(self, X, thre=None):
        if thre is None:
            thre = 0.5/self.n
        val, idx = X.cpu().topk(1)
        x_max = X.cpu().max()
        anchor = bidict()
        knt, total, pre, rec, f1 = 0, 0, 0, 0, 0
        for i in range(len(idx)):
            if val[i] > x_max-thre:
                if self.ent_ids1[i] not in anchor.keys() and self.ent_ids2[idx[i][0].item()] not in anchor.values():
                    anchor[self.ent_ids1[i]] = self.ent_ids2[idx[i][0].item()]
                    total += 1
                    if idx[i][0].item() == self.test_pair[i]:
                        knt += 1
        rec = knt/len(self.test_pair)
        if total > 0:
            pre = knt/total
            f1 = (2*pre*rec)/(pre+rec)
        print(knt, total, len(self.test_pair), "thre:{:.2e}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(thre,pre,rec,f1))
        self.anchor = anchor
        return pre, rec, f1

    def rel_align(self, emb_w=1, seed_w=1, M=20):
        # (1) name channel
        rel_n1 = len(self.rel_list[0])
        rel_sim = torch.zeros(len(self.rel_list[0]),len(self.rel_list[1]))
        if emb_w > 0 and 'D_W' not in self.data.loc:
            rel_emb = torch.tensor(self.rel_emb)
            emb_rel_sim = rel_emb[self.rel_list[0]]@rel_emb[self.rel_list[1]].T
            emb_rel_sim = 1-emb_rel_sim.float()
            rel_sim += emb_w*emb_rel_sim
        # (2) structure channel
        if seed_w > 0:
            anchor_rel_sim = torch.zeros_like(rel_sim)
            for (head, rel), tail in self.data.kg[0].er_e.items():
                if head in self.anchor.keys() and tail in self.anchor.keys():
                    head2 = self.anchor[head]
                    tail2 = self.anchor[tail]
                    if head2 in self.anchor.values() and (head2, tail2) in self.data.kg[1].ee_r.keys():
                        rel2 = self.data.kg[1].ee_r[(head2, tail2)]
                        anchor_rel_sim[rel][rel2-rel_n1] += 1
            print("anchor_rel_mat:", anchor_rel_sim.sum())
            anchor_rel_sim[anchor_rel_sim>M]=M
            anchor_rel_sim = 1- anchor_rel_sim/anchor_rel_sim.max()
            rel_sim += seed_w*anchor_rel_sim

        rel_mat = NeuralSinkhorn(rel_sim)
        self.r2r = {}
        for idx1, idx2 in enumerate(list(rel_mat.argmax(1).numpy())):
            self.r2r[idx1] = rel_n1 + idx2
        for idx2, idx1 in enumerate(list(rel_mat.argmax(0).numpy())):
            self.r2r[rel_n1 + idx2] = idx1

    def ot_align(self, initX=None, beta=0.1, iter=10):
        trans = NeuralSinkhorn(self.cost_st, beta=beta, trans=initX, outer_iter=iter)
        print("===OT align result===")
        test_align(trans, self.test_pair)
        return trans

    def gw_align(self, initX=None, lr=0.001, iter=200, alpha=1000):
        alpha = 2*self.n*self.n/(self.cost_s.to_dense().sum()+self.cost_t.to_dense().sum()).cpu().item()
        trans = self.gw_torch(self.cost_s, self.cost_t, alpha, trans=initX, beta=lr, outer_iter=iter, test_pair=self.test_pair)
        print("===GW align result===")
        test_align(trans, self.test_pair)
        return trans

    def gw_torch(self, cost_s, cost_t, alpha=None, p_s=None, p_t=None, trans=None, beta=0.001,
                outer_iter=1000, inner_iter=10, test_pair=None):
        device = cost_s.device
        last_fgw_score = 100
        knt = 0
        if p_s is None:
            p_s = torch.ones([cost_s.shape[0],1], device=device)/cost_s.shape[0]
        if p_t is None:
            p_t = torch.ones([cost_t.shape[0],1], device=device)/cost_t.shape[0]
        if trans is None:
            trans = p_s @ p_t.T
        for oi in range(outer_iter):
            cost = - 2 * cost_t @ (cost_s @ trans).T
            cost = cost.T  
            kernel = torch.exp(-cost / beta) * trans
            a = torch.ones_like(p_s)/p_s.shape[0]
            for ii in range(inner_iter):
                b = p_t / (kernel.T@a)
                a_new = p_s / (kernel@b)
                a = a_new
            trans = (a @ b.T) * kernel
            if oi % 20 == 0:
                test_align(trans, test_pair)
                gw_score = -torch.trace(cost_s.to_dense() @ trans @ cost_t.to_dense() @ trans.T).cpu().item()
                ot_score = (self.cost_st*trans).sum().cpu().item()
                fgw_score = alpha*gw_score + ot_score
                print(gw_score, ot_score, fgw_score)
                self.iters = oi
                if fgw_score - last_fgw_score > -0.00002:
                    knt += 1
                    if knt >= 2:
                        break
                last_fgw_score = fgw_score
        return trans