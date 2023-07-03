# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_joint: float = 1, cost_action: float = 1, cost_identity: float = 1, num_queries: int = 100):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_joint: This is the relative weight of the L1 error of the joint coordinates in the matching cost
            cost_action: This is the relative weight of the CE error of the activity categories in the matching cost
            cost_identity: This is the relative weight of the L1 error of the identity categories in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_joint = cost_joint
        self.cost_action = cost_action
        self.cost_identity = cost_identity
        assert cost_class != 0 or cost_joint != 0 or cost_action != 0 or cost_identity != 0, "all costs cant be 0"
        self.num_queries = num_queries
        
    def computeCE(self, pred, tar):
        n = pred.shape[0]
        ans = torch.rand(n, n)
        for i in range(n):
            for j in range(n):
                if pred.shape[1]>1:
                    tar = torch.tensor([0], dtype=torch.long)
                    ans[i, j] = F.cross_entropy(pred[i].unsqueeze(0), tar)
                else:
                    ans[i, j] = F.binary_cross_entropy(pred[i], tar[j])
        return ans
        
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_conf": Tensor of dim [batch_size, num_queries, 1] with the classification logits
                 "pred_joint": Tensor of dim [batch_size, num_queries, 3*num_joint] with the predicted joint coordinates
                 "pred_action": Tensor of dim [batch_size, num_queries, 8] with the activity category logits
                 "pred_identity": Tensor of dim [batch_size, num_queries, 10] with the identity category logits
            targets: This is a ground truth dict with the same shape as outputs

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target)
        """
        bs, num_queries = outputs["pred_conf"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_conf"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, 1]
        out_joint = outputs["pred_joint"].flatten(0, 1)  # [batch_size * num_queries, 3*num_joint]
        out_action = outputs["pred_action"].flatten(0, 1)  # [batch_size * num_queries, num_action]
        out_identity = outputs["pred_identity"].flatten(0, 1)  # [batch_size * num_queries, num_identity]

        # Also flatten the target labels
        tar_prob = targets["conf"].unsqueeze(-1).flatten(0, 1)  # [batch_size * num_queries, 1]
        tar_joint = targets["joint"].flatten(0, 1)  # [batch_size * num_queries, 3*num_joint]
        tar_action = targets["action"].unsqueeze(-1).flatten(0, 1)  # [batch_size * num_queries, 1]
        tar_identity = targets["identity"].unsqueeze(-1).flatten(0, 1)  # [batch_size * num_queries, 1]


        # Compute the cost between queries
        C_prob = self.computeCE(out_prob, tar_prob)
        C_joint = torch.cdist(out_joint, tar_joint, p=1)
        C_action = self.computeCE(out_action, tar_action)
        C_identity = self.computeCE(out_identity, tar_identity)
        

        # Final cost matrix
        C = self.cost_class*C_prob + self.cost_joint*C_joint + self.cost_action*C_action + self.cost_identity*C_identity
        C = C.view(bs, num_queries, -1).cpu()
        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(self.num_queries, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class, cost_joint=args.set_cost_joint,
        cost_action=args.set_cost_action, cost_identity=args.set_cost_identity,
        num_queries=100
        )
