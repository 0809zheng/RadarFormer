
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer import build_transformer
from src.matcher import build_matcher
from pytorch_metric_learning import losses as MetricLoss




def SinusoidalEncoding(seq_len, d_model):
    pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)] 
        for pos in range(seq_len)])
    pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                
    pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                
    return torch.FloatTensor(pos_table)  

def LearnableEncoding(seq_len, d_model):            
    return torch.randn(seq_len, d_model, dtype=torch.float, requires_grad=True)  

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    
class RadarFormer(nn.Module):
    """ This is the RadarFormer module that performs Radar-based Human Perception """
    def __init__(self, transformer, PosEmb, num_joint, num_action, num_identity, num_queries, input_dim):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         RadarFormer can detect in a detection area. We recommend 100 queries.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        
        # task-settings
        self.num_joint = num_joint
        self.num_action = num_action
        self.num_identity = num_identity
        
        # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.conf_embed = MLP(self.hidden_dim, -1, output_dim=1, num_layers=1)
        self.joint_embed = MLP(self.hidden_dim, self.hidden_dim, output_dim=self.num_joint*3, num_layers=3)
        self.action_embed = MLP(self.hidden_dim, -1, output_dim=self.num_action, num_layers=1)
        self.identity_embed = MLP(self.hidden_dim, -1, output_dim=self.num_identity, num_layers=1)
        
        
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.curr_proj = nn.Linear(input_dim, self.hidden_dim)
        self.hist_proj = nn.Linear(input_dim, self.hidden_dim)
        self.PosEmb = PosEmb

    def forward(self, curr_echo, hist_echo):
        """ The forward expects a NestedTensor, which consists of:
               - curr_echo: batched current echos, of shape [batch_size x 32 x 640]
               - hist_echo: batched history echos, of shape [batch_size x 32 x 640]

            It returns a dict with the following elements:
               - "pred_conf": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x 1]
               - "pred_joint": The 3d joint coordinates for all queries.
                                 Shape= [batch_size x num_queries x 3*num_joint]
               - "pred_action": The activity category for all queries.
                                  Shape= [batch_size x num_queries x num_action]
               - "pred_identity": The identity category for all queries.
                                    Shape= [batch_size x num_queries x num_identity]
        """
        n = curr_echo.shape[1]
        curr_echo = self.curr_proj(curr_echo)
        hist_echo = self.hist_proj(hist_echo)
        
        pos = self.PosEmb(n, self.hidden_dim)
        pos = pos.unsqueeze(0)
        
        hs = self.transformer(curr_echo, hist_echo, self.query_embed.weight, pos)[0]

        outputs_conf = self.conf_embed(hs)
        outputs_joint = self.joint_embed(hs)
        outputs_action = self.action_embed(hs)
        outputs_identity = self.identity_embed(hs)
        out = {
            'pred_conf': outputs_conf, 'pred_joint': outputs_joint,
            'pred_action': outputs_action, 'pred_identity': outputs_identity,
            'pred_feature': hs,
            }
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction
    """
    def __init__(self, matcher, losses):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        self.triplet_loss = MetricLoss.TripletMarginLoss()
        self.N = 0
        
        
    def loss_conf(self, outputs, targets, indices):
        """Classification loss (NLL) with class-balanced loss
        """
        assert 'pred_conf' in outputs
        src_logits = outputs['pred_conf'].sigmoid()
        # print('src_logits', src_logits.shape)

        idx = self._get_src_permutation_idx(indices)
        # print(idx)
        # target_classes_o = torch.cat([t["conf"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:], 0.,
                                    dtype=torch.float32, device=src_logits.device)

        target_classes[idx] = torch.ones(src_logits.shape[2:],
                                          dtype=torch.float32, device=src_logits.device)
        # print('target_classes', target_classes.shape)
        n = torch.sum(target_classes)
        self.N = self.N + n
        beta = (self.N - 1) / self.N
        E = (1 - beta ** n) / (1 - beta)

        loss_ce = F.binary_cross_entropy(src_logits, target_classes)
        loss_ce = loss_ce / E
        losses = {'loss_ce': loss_ce}

        return losses


    def loss_joint(self, outputs, targets, indices):
        """Compute the losses related to the 3d pose estimation.
        """
        assert 'pred_joint' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_joint = outputs['pred_joint'][idx]
        target_joint = targets['joint']
        target_joint = target_joint.view(src_joint.shape)

        loss_joint = F.l1_loss(src_joint, target_joint, reduction='none')

        losses = {}
        losses['loss_joint'] = loss_joint.mean()

        return losses
    

    def loss_action(self, outputs, targets, indices):
        """Compute the losses related to the activity recognition.
        """
        assert 'pred_action' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_action = outputs['pred_action'][idx]
        target_action = targets['action'].view(-1)

        loss_action = F.cross_entropy(src_action, target_action)
        losses = {'loss_action': loss_action}

        return losses


    def loss_identity(self, outputs, targets, indices):
        """Compute the losses related to the identity recognition.
        """
        assert 'pred_identity' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_identity = outputs['pred_identity'][idx]
        target_identity = targets['identity'].view(-1)

        loss_identity = F.cross_entropy(src_identity, target_identity)
        losses = {'loss_identity': loss_identity}

        return losses  
    

    def loss_triplet(self, outputs, targets, indices):
        """Compute the triplet losses related to the ReID.
        """
        assert 'pred_feature' in outputs
        d = outputs['pred_feature'].shape[-1]
        pred_feature = outputs['pred_feature'].reshape(-1, d)
        label = targets['identity'].view(-1)

        loss_triplet = self.triplet_loss(pred_feature, label)
        losses = {'loss_triplet': loss_triplet}

        return losses  
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'conf': self.loss_conf,
            'joint': self.loss_joint,
            'action': self.loss_action,
            'identity': self.loss_identity,
            'triplet': self.loss_triplet,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'pred_feature'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        return losses

    
def build(args):
    device = torch.device(args.device)

    transformer = build_transformer(args)
    # PosEmb = SinusoidalEncoding
    PosEmb = LearnableEncoding

    model = RadarFormer(
        transformer,
        PosEmb,
        num_joint=args.num_joint,
        num_action=args.num_action,
        num_identity=args.num_identity,
        num_queries=args.num_queries,
        input_dim=args.input_dim,
    )
    
    matcher = build_matcher(args)
    losses = ['conf', 'joint', 'action', 'identity', 'triplet']

    criterion = SetCriterion(matcher=matcher, losses=losses)
    criterion.to(device)

    return model, criterion

def build_test(args):

    transformer = build_transformer(args)
    # PosEmb = SinusoidalEncoding
    PosEmb = LearnableEncoding

    model = RadarFormer(
        transformer,
        PosEmb,
        num_joint=args.num_joint,
        num_action=args.num_action,
        num_identity=args.num_identity,
        num_queries=args.num_queries,
        input_dim=args.input_dim,
    )

    return model