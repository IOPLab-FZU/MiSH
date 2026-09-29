from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .hra_adapter import inject_hra, collect_ortho_reg, collect_beta_reg
import torch
import torch.nn as nn
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class MiSH(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        # ---- MiSH: Minor-Subspace Householder Reflection Adaptation ----
        # Freeze the whole CLIP backbone, then inject orthogonal HRA adapters
        # (confined to each weight's minor singular subspace) and unfreeze
        # LayerNorm. The TSE heads above stay trainable. This drives the
        # trainable-param count from 153M (full fine-tune) down to ~4-6M.
        self.freeze_clip = getattr(args, 'freeze_clip', True)
        self.hra_lambda = getattr(args, 'hra_lambda', 1e-3)
        # MSC: confine reflections to the minor singular subspace (protect span(V_p)).
        self.hra_subspace = getattr(args, 'hra_subspace', 0.9)
        # DGM: per-axis learnable reflection coefficient beta in the minor subspace.
        self.dgm = getattr(args, 'dgm', True)
        self.dgm_lambda = getattr(args, 'dgm_lambda', 1e-4)
        if self.freeze_clip:
            for p in self.base_model.parameters():
                p.requires_grad_(False)
            hra_r = getattr(args, 'hra_r', 32)
            n_hra = inject_hra(self.base_model,
                               r=hra_r,
                               lam=self.hra_lambda,
                               where=tuple(getattr(args, 'hra_where', 'mlp').split('+')),
                               subspace_energy=self.hra_subspace,
                               dgm=self.dgm)
            # keep LayerNorm trainable (cheap, standard PETL practice)
            for name, p in self.base_model.named_parameters():
                if 'ln_' in name or '.ln' in name:
                    p.requires_grad_(True)
            print(f'[MiSH] frozen CLIP, injected HRA into {n_hra} linear layers '
                  f'(r={hra_r}, where={getattr(args, "hra_where", "mlp")}, '
                  f'subspace={self.hra_subspace}, dgm={self.dgm})')

        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})

        # MiSH: orthogonality regulariser on Householder reflection vectors.
        # do_train sums every key containing "loss", so this is added to total.
        if self.freeze_clip and self.hra_lambda > 0:
            ortho = collect_ortho_reg(self.base_model)
            ret.update({'ortho_loss': self.hra_lambda * ortho})

        # DGM: regularise learnable reflection coefficients toward 2 (pure HRA).
        # Keyed with "loss" -> summed into total_loss by do_train.
        if self.freeze_clip and self.dgm and self.dgm_lambda > 0:
            beta = collect_beta_reg(self.base_model)
            ret.update({'beta_loss': self.dgm_lambda * beta})

        return ret


def build_model(args, num_classes=11003):
    model = MiSH(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
