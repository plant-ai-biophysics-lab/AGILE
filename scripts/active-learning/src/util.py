import torch
import numpy as np
import torch.nn.functional as F
import math

from torchvision.ops import boxes as box_ops
from torch import nn
from collections import OrderedDict
from typing import List, Tuple
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.image_list import ImageList
from dataclasses import dataclass
from toma import toma
from tqdm.auto import tqdm
from sklearn.metrics import pairwise_distances

### CUSTOM FASTERRCNN CLASS ###
class CustomFasterRCNN(FasterRCNN):
    
    def __init__(self, backbone, num_classes, device):
        super(CustomFasterRCNN, self).__init__(backbone, num_classes)
        self.device = torch.device(device)
        self.to(self.device)
        
    def forward(self, images, targets=None, dropout=False):        
        # Check if the model is in training mode
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
                
            outputs = super(CustomFasterRCNN, self).forward(images, targets)
            return outputs
        
        # only during prediction
        if dropout:
            self.roi_heads.box_head.fc6 = nn.Sequential(
                self.roi_heads.box_head.fc6,
                self.fc6_dropout
            ).to(self.device)
            self.roi_heads.box_head.fc7 = nn.Sequential(
                self.roi_heads.box_head.fc7,
                self.fc7_dropout
            ).to(self.device)
        self.to(self.device)
        
        outputs = super(CustomFasterRCNN, self).forward(images, targets)
        
        # check images
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        images = ImageList(images, original_image_sizes)
        
        # extract features and metrics
        features = self.backbone(images.tensors.to(self.device))
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = self.rpn(images, features, targets)
        box_features = self.roi_heads.box_roi_pool(features, proposals, images.image_sizes).to(self.device)
        box_features = self.roi_heads.box_head(box_features).to(self.device)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)
        
        # post process class logits
        device = class_logits.device
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        class_logits_list = class_logits.split(boxes_per_image, 0)
        
        all_logits = []
        for i, (boxes, scores, logits, image_shape) in enumerate(zip(pred_boxes_list, pred_scores_list, class_logits_list, original_image_sizes)):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(20, device=device)  # Assuming 19 classes + 1 background class
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            if boxes.size(1) > 1:
                boxes = boxes[:, 1:]
                scores = scores[:, 1:]
                labels = labels[:, 1:]
                logits = logits[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            logits = logits.repeat_interleave(19, dim=0)

            # remove low scoring boxes
            score_inds = torch.where(scores > self.roi_heads.score_thresh)[0]
            boxes, scores, labels, logits = boxes[score_inds], scores[score_inds], labels[score_inds], logits[score_inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, logits = boxes[keep], scores[keep], labels[keep], logits[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.roi_heads.nms_thresh)
            boxes, scores, labels, logits = boxes[keep], scores[keep], labels[keep], logits[keep]

            # append filtered logits to all_logits
            all_logits.append(logits)

        return outputs, all_logits
    
### BatchBALD Functions and Classes ###
def compute_conditional_entropy(log_probs_N_K_C: list, batch_size: int) -> torch.Tensor:
    # N, K, C = log_probs_N_K_C.shape
    N = len(log_probs_N_K_C)
    entropies_N = []

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    for log_probs_n_K_C in log_probs_N_K_C:
        entropies_n_K = []
        for log_probs_K in log_probs_n_K_C:
            
            # average over all detections
            entropies_n = torch.stack([torch.mean(log_probs_n*torch.exp(log_probs_n), dim=0) for log_probs_n in log_probs_K])
            
            entropies_n = -torch.sum(entropies_n, dim=1) # sum over all classes
            entropies_n_K.append(entropies_n)
        
        entropies_n_K = torch.stack(entropies_n_K).mean(dim=0) # average over all MC samples
        entropies_N.extend(entropies_n_K)
        
        pbar.update(1)

    pbar.close()
    entropies_N = torch.stack(entropies_N)
    return entropies_N

def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N

@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]

def get_batchbald_batch(
    log_probs_N_K_C: list, n_train: int, num_samples: int, num_classes: int, batch_size: int, N: int, dtype=None, device=None
) -> CandidateBatch:
    # N, K, C = log_probs_N_K_C.shape (N: number of samples, K: number of MC samples, C: number of classes)

    # n_train = min(n_train, N)
    
    flattened_log_probs = []
    for log_probs in log_probs_N_K_C:
        max_length = max(len(lp) for lp in log_probs)
        batch_flattened = [[] for _ in range(max_length)]
        
        for i in range(len(log_probs)):
            for j in range(len(log_probs[i])):
                while len(batch_flattened[j]) <= i:
                    batch_flattened[j].append(None)
                batch_flattened[j][i] = log_probs[i][j]
        
        flattened_log_probs.append(batch_flattened)
    flattened_log_probs = [item for sublist in flattened_log_probs for item in sublist]
    
    candidate_indices = []
    candidate_scores = []

    if n_train == 0:
        return CandidateBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C, batch_size)

    batch_joint_entropy = DynamicJointEntropy(
        num_samples, n_train - 1, num_samples, num_classes-1, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(len(conditional_entropies_N), dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(n_train), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            # batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])
            batch_joint_entropy.add_variables(flattened_log_probs[latest_index : latest_index + 1])

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(
            log_probs_N_K_C, 
            B=len(log_probs_N_K_C), 
            K=num_samples,
            C=num_classes,
            output_entropies_B=scores_N
        )

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        for index in candidate_indices:
            scores_N[index] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return CandidateBatch(candidate_scores, candidate_indices)

class JointEntropy:
    """Random variables (all with the same # of categories $C$) can be added via `JointEntropy.add_variables`.

    `JointEntropy.compute` computes the joint entropy.

    `JointEntropy.compute_batch` computes the joint entropy of the added variables with each of the variables in the provided batch probabilities in turn."""

    def compute(self) -> torch.Tensor:
        """Computes the entropy of this joint entropy."""
        raise NotImplementedError()

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "JointEntropy":
        """Expands the joint entropy to include more terms."""
        raise NotImplementedError()

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None) -> torch.Tensor:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        raise NotImplementedError()

class ExactJointEntropy(JointEntropy):
    joint_probs_M_K: torch.Tensor

    def __init__(self, joint_probs_M_K: torch.Tensor):
        self.joint_probs_M_K = joint_probs_M_K

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> "ExactJointEntropy":
        return ExactJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    def compute(self) -> torch.Tensor:
        probs_M = torch.mean(self.joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(probs_M) * probs_M
        entropy = torch.sum(nats_M)
        return entropy

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "ExactJointEntropy":
        assert self.joint_probs_M_K.shape[1] == log_probs_N_K_C.shape[1]

        N, K, C = log_probs_N_K_C.shape
        joint_probs_K_M_1 = self.joint_probs_M_K.t()[:, :, None]

        probs_N_K_C = log_probs_N_K_C.exp()

        # Using lots of memory.
        for i in range(N):
            probs_i__K_1_C = probs_N_K_C[i][:, None, :].to(joint_probs_K_M_1, non_blocking=True)
            joint_probs_K_M_C = joint_probs_K_M_1 * probs_i__K_1_C
            joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))

        self.joint_probs_M_K = joint_probs_K_M_1.squeeze(2).t()
        return self

    def compute_batch(self, log_probs_B_K_C: list, B: int, K: int, C: int, output_entropies_B=None) -> torch.Tensor:

        entropies_B = []
        if output_entropies_B is None:
            output_entropies_B = torch.empty(B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device)

        pbar = tqdm(total=B, desc="ExactJointEntropy.compute_batch", leave=False)
        
        for log_probs_b_K_C in log_probs_B_K_C:
            entropies_b_K = []
            for log_probs_K in log_probs_b_K_C:
                
                # average over all detections
                entropies_b = torch.stack([torch.mean(log_probs_b*torch.exp(log_probs_b), dim=0) for log_probs_b in log_probs_K])
                entropies_b = entropies_b.exp()
                entropies_b_K.append(entropies_b)

            entropies_b_K = torch.stack(entropies_b_K, dim=1)
            b = entropies_b_K.shape[0]
            probs_b_M_C = torch.empty(
                (b, C-1),
                dtype=self.joint_probs_M_K.dtype,
                device=self.joint_probs_M_K.device,
            )
            
            for i in range(b):
                torch.matmul(
                    self.joint_probs_M_K,
                    entropies_b_K[i].to(self.joint_probs_M_K, non_blocking=True),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K

            entropies_B.extend(torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C, dim=1))

            pbar.update(1)

        pbar.close()
        entropies_B = torch.stack(entropies_B)
        return entropies_B

def batch_multi_choices(probs_b_C, M: int):
    """
    probs_b_C: Ni... x C

    Returns:
        choices: Ni... x M
    """
    # probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))
    probs_B_C = [probs.reshape((-1, probs.shape[-1])) for probs in probs_b_C]

    # samples: Ni... x draw_per_xx
    # choices = torch.multinomial(probs_B_C, num_samples=M, replacement=True)
    choices = [torch.multinomial(probs, num_samples=M, replacement=True) for probs in probs_B_C]

    # choices_b_M = choices.reshape(list(probs_b_C.shape[:-1]) + [M])
    choices_b_M = [choices[i].reshape(list(probs_b_C[i].shape[:-1]) + [M]).long() for i in range(len(choices))]
    
    # make long
    return choices_b_M


def gather_expand(data, dim, index):
    
    # trims data and index if unequal size
    if any(tensor.size() != data[0].size() for tensor in data):
        min_sizes = [max(1, min(tensor.size(d) for tensor in data)) for d in range(len(data[0].size()))]
        data = [tensor[:min_sizes[0], :min_sizes[1], :min_sizes[2]] for tensor in data]
        
    if any(tensor.size() != index[0].size() for tensor in index):
        min_sizes = [max(1, min(tensor.size(d) for tensor in index)) for d in range(len(index[0].size()))]
        index = [tensor[:min_sizes[0], :min_sizes[1], :min_sizes[2]] for tensor in index]
    
    data = torch.stack(data, dim=1)
    index = torch.stack(index, dim=1)
    
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)

    return torch.gather(data, dim, index)

class SampledJointEntropy(JointEntropy):
    """Random variables (all with the same # of categories $C$) can be added via `SampledJointEntropy.add_variables`.

    `SampledJointEntropy.compute` computes the joint entropy.

    `SampledJointEntropy.compute_batch` computes the joint entropy of the added variables with each of the variables in the provided batch probabilities in turn."""

    sampled_joint_probs_M_K: torch.Tensor

    def __init__(self, sampled_joint_probs_M_K: torch.Tensor):
        self.sampled_joint_probs_M_K = sampled_joint_probs_M_K

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> "SampledJointEntropy":
        return SampledJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    @staticmethod
    def sample(probs_N_K_C: torch.Tensor, M: int, K: int) -> "SampledJointEntropy":
        # K = len(probs_N_K_C)
        # K = probs_N_K_C.shape[1]

        # S: num of samples per w
        S = M // K

        choices_N_K_S = batch_multi_choices(probs_N_K_C, S)

        # expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_choices_N_1_K_S = [choices[None, :, :] for choices in choices_N_K_S]
        # expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]
        expanded_probs_N_K_1_C = [probs[:, None, :] for probs in probs_N_K_C]

        probs_N_K_K_S = gather_expand(expanded_probs_N_K_1_C, dim=-1, index=expanded_choices_N_1_K_S)
        # exp sum log seems necessary to avoid 0s?
        probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
        samples_K_M = probs_K_K_S.reshape((K, -1))

        samples_M_K = samples_K_M.t()
        return SampledJointEntropy(samples_M_K)

    def compute(self) -> torch.Tensor:
        sampled_joint_probs_M = torch.mean(self.sampled_joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(sampled_joint_probs_M)
        entropy = torch.mean(nats_M)
        return entropy

    def add_variables(self, log_probs_N_K_C: torch.Tensor, M2: int) -> "SampledJointEntropy":
        K = self.sampled_joint_probs_M_K.shape[1]
        assert K == log_probs_N_K_C.shape[1]

        sample_K_M1_1 = self.sampled_joint_probs_M_K.t()[:, :, None]

        new_sample_M2_K = self.sample(log_probs_N_K_C.exp(), M2).sampled_joint_probs_M_K
        new_sample_K_1_M2 = new_sample_M2_K.t()[:, None, :]

        merged_sample_K_M1_M2 = sample_K_M1_1 * new_sample_K_1_M2
        merged_sample_K_M = merged_sample_K_M1_M2.reshape((K, -1))

        self.sampled_joint_probs_M_K = merged_sample_K_M.t()

        return self

    def _compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None):
        assert self.sampled_joint_probs_M_K.shape[1] == log_probs_B_K_C.shape[1]

        B, K, C = log_probs_B_K_C.shape
        M = self.sampled_joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device)

        pbar = tqdm(total=B, desc="SampledJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)
        def chunked_joint_entropy(chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int):
            b = chunked_log_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.sampled_joint_probs_M_K.dtype,
                device=self.sampled_joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    self.sampled_joint_probs_M_K,
                    chunked_log_probs_b_K_C[i].to(self.sampled_joint_probs_M_K, non_blocking=True).exp(),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K

            q_1_M_1 = self.sampled_joint_probs_M_K.mean(dim=1, keepdim=True)[None]

            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C / q_1_M_1, dim=(1, 2)) / M,
                non_blocking=True,
            )

            pbar.update(end - start)

        pbar.close()

        return output_entropies_B
    
    def compute_batch(self, log_probs_B_K_C: list, B: int, K: int, C: int, output_entropies_B=None) -> torch.Tensor:

        M = self.sampled_joint_probs_M_K.shape[0]
        
        entropies_B = []
        if output_entropies_B is None:
            output_entropies_B = torch.empty(B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device)

        pbar = tqdm(total=B, desc="ExactJointEntropy.compute_batch", leave=False)
        
        for log_probs_b_K_C in log_probs_B_K_C:
            entropies_b_K = []
            for log_probs_K in log_probs_b_K_C:
                
                # average over all detections
                entropies_b = torch.stack([torch.mean(log_probs_b*torch.exp(log_probs_b), dim=0) for log_probs_b in log_probs_K])
                entropies_b = entropies_b.exp()
                entropies_b_K.append(entropies_b)

            entropies_b_K = torch.stack(entropies_b_K, dim=1)
            b = entropies_b_K.shape[0]
            probs_b_M_C = torch.empty(
                (b, C-1),
                dtype=self.sampled_joint_probs_M_K.dtype,
                device=self.sampled_joint_probs_M_K.device,
            )
            
            for i in range(b):
                torch.matmul(
                    self.sampled_joint_probs_M_K,
                    entropies_b_K[i].to(self.sampled_joint_probs_M_K, non_blocking=True),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K
            q_1_M_1 = self.sampled_joint_probs_M_K.mean(dim=1, keepdim=True)[None]

            entropies_B.extend(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C / q_1_M_1.mean(), dim=1) / M
            )

            pbar.update(1)

        pbar.close()
        entropies_B = torch.stack(entropies_B)
        return entropies_B

class DynamicJointEntropy(JointEntropy):
    inner: JointEntropy
    log_probs_max_N_K_C: torch.Tensor
    N: int
    M: int

    def __init__(self, M: int, max_N: int, K: int, C: int, dtype=None, device=None):
        self.M = M
        self.N = 0
        self.C = C
        self.max_N = max_N
        self.K = K

        self.inner = ExactJointEntropy.empty(K, dtype=dtype, device=device)
        # self.log_probs_max_N_K_C = torch.empty((max_N, K, C), dtype=dtype, device=device)
        self.log_probs_max_N_K_C = []

    def add_variables(self, log_probs_N_K_C: list) -> "DynamicJointEntropy":
        # C = self.log_probs_max_N_K_C.shape[2]
        # add_N = log_probs_N_K_C[0].shape[0]
        add_N = len(log_probs_N_K_C)

        # assert self.log_probs_max_N_K_C.shape[0] >= self.N + add_N
        # assert self.log_probs_max_N_K_C.shape[2] == C
        
        # self.log_probs_max_N_K_C[self.N : self.N + add_N] = log_probs_N_K_C
        self.log_probs_max_N_K_C.append(log_probs_N_K_C[0])
        self.N += add_N

        num_exact_samples = self.C ** self.N
        if num_exact_samples > self.M:
            temp = [self.log_probs_max_N_K_C[:self.N][i][j].exp() for i in range(len(self.log_probs_max_N_K_C[:self.N])) for j in range(len(self.log_probs_max_N_K_C[:self.N][i]))]
            # self.inner = SampledJointEntropy.sample(self.log_probs_max_N_K_C[: self.N].exp(), self.M)
            self.inner = SampledJointEntropy.sample(temp, self.M, self.K)
        else:
            self.inner.add_variables(log_probs_N_K_C)

        return self

    def compute(self) -> torch.Tensor:
        return self.inner.compute()

    def compute_batch(self, log_probs_B_K_C: list, B:int, K:int, C:int, output_entropies_B=None) -> torch.Tensor:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        return self.inner.compute_batch(log_probs_B_K_C, B, K, C, output_entropies_B)
    
### Core-sets by k Greedy Centers ###
class kCenterGreedy:
    def __init__(self, X, already_selected, metric='euclidean'):
        self.X = X
        self.metric = metric
        self.min_distances = None
        self.already_selected = already_selected

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
        if len(cluster_centers) > 0: 
            x = self.X[cluster_centers]
            dist = pairwise_distances(self.X, x, metric=self.metric)
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch(self, N):
        self.update_distances(self.already_selected, only_new=False, reset_dist=True)
        new_batch = []
        for _ in range(N):
            if len(self.already_selected) == 0:
                ind = np.random.choice(np.arange(self.X.shape[0]))
            else:
                # Temporarily set distances of already selected indices to -infinity
                distances = self.min_distances.copy()
                distances[self.already_selected] = -np.inf
                ind = np.argmax(distances)  # Now correctly using argmax
            if ind in self.already_selected:
                raise ValueError("Selected index is already in 'already_selected'. This should not happen.")
            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
            self.already_selected.append(ind)  # Ensure the selected index is marked as selected
        return new_batch