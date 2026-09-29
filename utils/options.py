import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiSH Args")

    parser.add_argument("--noisy_rate", default=0.0, type=float)
    parser.add_argument("--noisy_file", default='', type=str)
    parser.add_argument("--tau", default=0.015, type=float)
    parser.add_argument("--select_ratio", default=0.3, type=float)
    parser.add_argument("--margin", default=0.1, type=float)

    ######################## MiSH: Minor-Subspace Householder Reflection Adaptation ########################
    parser.add_argument("--freeze_clip", default=True, action='store_true',
                        help="freeze CLIP backbone and use HRA adapters (lightweight mode)")
    parser.add_argument("--hra_r", default=32, type=int,
                        help="number of Householder reflections per adapted linear")
    parser.add_argument("--hra_where", default='mlp', type=str,
                        help="where to inject HRA: 'mlp', 'attn', or 'mlp+attn'")
    parser.add_argument("--hra_lambda", default=1e-3, type=float,
                        help="orthogonality regularisation strength (0 disables)")
    parser.add_argument("--hra_subspace", default=0.9, type=float,
                        help="MSC (Minor-Subspace Confinement): principal-subspace energy to protect "
                             "in (0,1); reflections are confined to the minor singular subspace. "
                             "0 disables (recovers vanilla HRA).")
    parser.add_argument("--dgm", default=True, action='store_true',
                        help="DGM (Directional Gain Modulation): make the reflection coefficient "
                             "per-axis learnable (beta_j, init 2). Protects span(V_p) exactly while "
                             "unlocking directional gain. With MSC disabled, axes span the full input space.")
    parser.add_argument("--dgm_lambda", default=1e-4, type=float,
                        help="DGM regularisation strength pulling beta toward 2 (pure HRA). "
                             "0 disables the regulariser.")

    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="MiSH", help="experiment name to save")
    parser.add_argument("--output_dir", default="run_logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')
    parser.add_argument("--txt_aug", default=True, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='TAL+sr0.3_tau0.015_margin0.1_nr0.0', help="which loss to use ['TAL', 'TRL', 'InfoNCE', 'SDM']")

    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="RSTPReid", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="/root/autodl-tmp/Datasets")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args
