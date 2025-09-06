# -*- coding: UTF-8 -*-
"""
@Time : 07/07/2025 20:01
@Author : Xiaoguang Liang
@File : evaluation_metrics_v1.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
from functools import partial

import torch
import numpy as np
from tqdm import trange, tqdm
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed

from src.utils.common import timer


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


# Import CUDA version of approximate Structural Losses, from https://github.com/MaciejZamorski/pytorch_structural_losses
try:
    from StructuralLosses.nn_distance import nn_distance


    def distChamferCUDA(x, y):
        return nn_distance(x, y)
except:
    print("distChamferCUDA not available; fall back to slower version.")


    def distChamferCUDA(x, y):
        return distChamfer(x, y)


def emd_approx(x, y):
    bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    assert npts == mpts, "EMD only works if two point clouds are equal size"
    dim = x.shape[-1]
    x = x.reshape(bs, npts, 1, dim)
    y = y.reshape(bs, 1, mpts, dim)
    dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

    emd_lst = []
    dist_np = dist.cpu().detach().numpy()
    for i in range(bs):
        d_i = dist_np[i]
        r_idx, c_idx = linear_sum_assignment(d_i)
        emd_i = d_i[r_idx, c_idx].mean()
        emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    emd_torch = torch.from_numpy(emd).to(x)
    return emd_torch


try:
    from StructuralLosses.match_cost import match_cost


    def emd_approx_cuda(sample, ref):
        # sample.shape = [1, 1024, 3], ref.shape = [1, 1024, 3]
        B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
        assert N == N_ref, "Not sure what would EMD do in this case"
        emd = match_cost(sample, ref)  # (B,)
        emd_norm = emd / float(N)  # (B,)
        return emd_norm
except:
    print("emd_approx_cuda not available. Fall back to slower version.")


    def emd_approx_cuda(sample, ref):
        return emd_approx(sample, ref)


def EMD_CD(sample_pcs, ref_pcs, batch_size, accelerated_cd=True, reduced=True,
           accelerated_emd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []

    # cpu_num = 10
    # _compute_cd_cmd = partial(compute_cd_cmd, N_sample=N_sample, batch_size=batch_size,
    #                           sample_pcs=sample_pcs, ref_pcs=ref_pcs,
    #                           accelerated_cd=accelerated_cd,
    #                           accelerated_emd=accelerated_emd)
    # result = Parallel(n_jobs=cpu_num, backend='multiprocessing', verbose=10)(
    #     delayed(_compute_cd_cmd)(batch) for batch in tqdm(range(0, N_sample, batch_size)))
    for b_start in trange(0, N_sample, batch_size):
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        if accelerated_cd:
            dl, dr = distChamferCUDA(sample_batch, ref_batch)
        else:
            dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        if accelerated_emd:
            emd_batch = emd_approx_cuda(sample_batch, ref_batch)
        else:
            emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': round(cd.cpu().detach().item(), 3),
        'MMD-EMD': round(emd.cpu().detach().item(), 3),
    }
    return results


def compute_cd_cmd(b_start, N_sample=None, batch_size=None, sample_pcs=None, ref_pcs=None,
                   accelerated_cd=True,
                   accelerated_emd=True):
    b_end = min(N_sample, b_start + batch_size)
    sample_batch = sample_pcs[b_start:b_end]
    ref_batch = ref_pcs[b_start:b_end]

    if accelerated_cd:
        dl, dr = distChamferCUDA(sample_batch, ref_batch)
    else:
        dl, dr = distChamfer(sample_batch, ref_batch)
    cd_value = dl.mean(dim=1) + dr.mean(dim=1)

    if accelerated_emd:
        emd_batch = emd_approx_cuda(sample_batch, ref_batch)
    else:
        emd_batch = emd_approx(sample_batch, ref_batch)
    return cd_value, emd_batch


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, accelerated_cd=True,
                      accelerated_emd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    for sample_b_start in trange(N_sample):
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            if accelerated_cd and distChamferCUDA is not None:
                dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
            else:
                dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            if accelerated_emd:
                # sample_batch_exp.shape = [1, 1024, 3], ref_batch.shape = [1, 1024, 3]
                emd_batch = emd_approx_cuda(sample_batch_exp, ref_batch)
            else:
                emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd=True, simple_mode=True):
    if simple_mode:
        return EMD_CD(sample_pcs, ref_pcs, batch_size, accelerated_cd=accelerated_cd)
    results = {}
    # sample_pcs.shape = [5, 1024, 3], ref_pcs.shape = [5, 1024, 3]
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd)

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: round(v.item(), 3) for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({
        "%s-EMD" % k: round(v.item(), 3) for k, v in res_emd.items()
    })

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size, accelerated_cd=accelerated_cd)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd)

    # 1-NN results
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: round(v.item(), 3) for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    results.update({
        "1-NN-EMD-%s" % k: round(v.item(), 3) for k, v in one_nn_emd_res.items() if 'acc' in k
    })

    return results


if __name__ == "__main__":
    B, N = 5, 2 ** 10
    x = torch.rand(B, N, 3).cuda()
    y = torch.rand(B, N, 3).cuda()

    with timer("all tasks"):
        results = compute_all_metrics(x, y, 1)
        print(f'results: {results}')

    min_l, min_r = distChamferCUDA(x, y)
    # min_l, min_r = distChamfer(x, y)
    print(min_l.shape)
    print(min_r.shape)

    l_dist = min_l.mean().cpu().detach().item()
    r_dist = min_r.mean().cpu().detach().item()
    print(l_dist, r_dist)

    emd_batch = emd_approx_cuda(x, y)
    print(emd_batch.shape)
    print(emd_batch.mean().detach().item())
