# -*- coding: UTF-8 -*-
"""
@Time : 01/07/2025 11:20
@Author : Xiaoguang Liang
@File : evaluation_util.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
from typing import Dict

from pytorch3d.loss.chamfer import chamfer_distance
from EMD_module.emd_module import EMDModule
from spaghetti.custom_types import *
from spaghetti.models import models_utils
from spaghetti.utils import files_utils, mesh_utils

from configs.log_config import logger
from src.utils.common import timer
from configs.global_setting import device

EPS = 0.002
ITERATION = 10 ** 4


class Evaluator:

    def __init__(self, num_samples: int, batch_size: int):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device

        self.emd_evaluator = EMDModule()
        self.buffers = [], []

        self.chamfers = []
        self.emds = []

    def compute_all_metrics_by_buffer(self, mesh_a: T_Mesh, mesh_b: T_Mesh):
        scale = (mesh_b[0].max(0)[0] - mesh_b[0].min(0)[0]).norm(2)
        samples_a = mesh_utils.sample_on_mesh(mesh_a, self.num_samples, sample_s=mesh_utils.SampleBy.AREAS)[
            0]
        samples_b = mesh_utils.sample_on_mesh(mesh_b, self.num_samples, sample_s=mesh_utils.SampleBy.AREAS)[
            0]
        self.buffers[0].append(samples_a / scale)
        self.buffers[1].append(samples_b / scale)
        if len(self.buffers[0]) == self.batch_size:
            self.empty_buffer()

    def empty_buffer(self):
        if len(self.buffers[0]) > 0:
            pc_a, pc_b = torch.stack(self.buffers[0], dim=0).to(self.device), torch.stack(self.buffers[1],
                                                                                          dim=0).to(
                self.device)
            self.compute_all_metrics(pc_a, pc_b)
        self.buffers = [], []

    @models_utils.torch_no_grad
    def compute_all_metrics(self, pred_pc: T, gt_pc: T):
        chamfer_loss, _ = chamfer_distance(pred_pc, gt_pc, single_directional=False)
        dis, _ = self.emd_evaluator(pred_pc, gt_pc, EPS, ITERATION)
        emd = np.sqrt(dis.cpu()).mean()
        self.chamfers.append(chamfer_loss.detach().cpu())
        self.emds.append(emd)

    def save(self, path: str):
        self.empty_buffer()
        emds = torch.cat(self.emds)
        chamfers = torch.cat(self.chamfers)
        self.print_evaluation(emds, chamfers)
        files_utils.save_pickle({"emd": emds, "chamfer": chamfers}, path)

    @staticmethod
    def print_evaluation(emds: T, chamfers: T):
        emds = emds.sort()[0]
        chamfers = chamfers.sort()[0]
        median_index = chamfers.shape[0] // 2
        logger.info(f"{chamfers.mean() * 1000:.3f} & {chamfers[median_index] * 1000:.3f} &"
                    f" {emds.mean() * 1000:.2f} & {emds[median_index] * 1000:.2f}")
        logger.info(
            f"chamfer mean: {chamfers.mean() * 1000:.3f} chamfer median: {chamfers[median_index] * 1000:.3f}\n"
            f"emd mean:     {emds.mean() * 1000:.2f}     emd median:     {emds[median_index] * 1000:.2f}\n"
        )

    @staticmethod
    def prepare_emd(pc_a, pc_b):
        if pc_a.shape[1] > 1024:
            order = (torch.rand(pc_a.shape[1]).argsort()[:1024]).to(pc_a.device)
            arange = torch.arange(pc_a.shape[0], device=pc_a.device)
            pc_a, pc_b = pc_a[arange, order], pc_b[arange, order]
        return pc_a, pc_b


class EvaluatorIoU:

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.iou_surface_a = []
        self.iou_surface_b = []
        self.iou_random = []

    def print_evaluation(self):
        logger.info(f"experiment_name: {self.experiment_name}")
        for arr, name in zip((self.iou_surface_a, self.iou_surface_b, self.iou_random),
                             ('surface', 'surface', 'random')):
            values = torch.tensor(arr)
            mean, median = values.mean(), values.median()
            logger.info(f"{name} mean: {mean}   {name} median: {median}")

    def __call__(self, pred: T, labels: T):
        labels = labels.lt(0)
        pred = pred.lt(0)
        iou = (pred * labels).sum(-1) / (pred + labels).sum(-1)
        self.iou_surface_a.append(iou[0].item())
        self.iou_surface_b.append(iou[1].item())
        self.iou_random.append(iou[2].item())


class EvaluatorParts:

    def __init__(self):
        self.log = {"iou_parts_near_a": [], "iou_parts_near_b": [], "iou_parts_random": [],
                    "iou_parts_inside": [],
                    "iou_all_near_a": [], "iou_all_near_b": [], "iou_all_random": [], "iou_all_inside": []}

    def evaluate_all_parts(self, occ_all: T, split_out: Dict[int, T], supports: T, gmm_labels: T):
        coords_labels = gmm_labels[supports.argmax(-1)]
        union: TN = None
        for i, occ in split_out.items():
            self.evaluate_part(i, occ, occ_all.clone(), coords_labels, gmm_labels)
            if union is None:
                union = occ
            else:
                union = union + occ
        self.iou_occ(union, occ_all,
                     ("iou_all_near_a", "iou_all_near_b", "iou_all_random", "iou_all_inside"))

    def evaluate_part(self, i: int, occ: T, occ_all: T, coords_labels: T, gmm_labels: T):
        occ_all[~coords_labels.eq(i)] = 0
        self.iou_occ(occ_all, occ,
                     ("iou_parts_near_a", "iou_parts_near_b", "iou_parts_random", "iou_parts_inside"))
        # if coords_part_labels.shape[0] != 0:
        #     self.iou_parts.append((coords_part_labels.sum().float() / coords_part_labels.shape[0]).item())
        # coords_part_labels = coords_labels.eq(i)

    def iou_occ(self, occ_a: T, occ_b: T, keys: Tuple[str, ...]):
        occ_a, occ_b = occ_a.view(len(keys), -1), occ_b.view(len(keys), -1)
        iou = (occ_a * occ_b).sum(1).float() / (occ_b + occ_b).sum(1).float()
        for i, key in enumerate(keys):
            if not torch.isnan(iou[i]):
                self.log[key].append(iou[i])

    def save(self, path: str):
        log = self.get_log()
        files_utils.save_pickle(log, path)
        self.print_evaluation(log)

    def get_log(self):
        log = {}
        for key, item in self.log.items():
            log[key] = torch.tensor(item)
        return log

    @staticmethod
    def print_evaluation(log: Dict[str, T]):
        mean_part = torch.cat(
            (log["iou_parts_near_b"], log["iou_parts_random"], log["iou_parts_inside"])).mean()
        mean_all = torch.cat((log["iou_all_near_b"], log["iou_all_random"], log["iou_all_inside"])).mean()
        logger.info(f"& {mean_part:.3f} & {mean_all:.3f}")
        for key, item in log.items():
            logger.info(f"{key}: {item.mean():.3f} {key}: {item.median():.3f}")


if __name__ == '__main__':
    evaluator = Evaluator(2 ** 10, 1)

    x, y = torch.rand(5, 2 ** 10, 3).cuda(), torch.rand(5, 2 ** 10, 3).cuda()
    with timer("all tasks"):
        evaluator.compute_all_metrics(x, y)
        print(f"chamfers: {evaluator.chamfers}")
        print(f"emds: {evaluator.emds}")
