import argparse

def load_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datasets', type=str, default='PaviaU')
    
    parser.add_argument('--c_pca', type=int, default=30)
    
    parser.add_argument('--ratio_train', type=float, default=0.003) ##0.0003
    parser.add_argument('--windowsize', type=int, default=27)
    parser.add_argument('--batch_size', type=int, default=32)
    
    parser.add_argument('--feature_dim', type=int, default=128)
    
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--ckpt_interval', default=100, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    
    parser.add_argument('--mask_ratio', type=int, default=0.7)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--nexp', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='result')
    parser.add_argument('--log_dir', type=str, default='logs')

    parser.add_argument('--hid_chans', type=int, default=64)
    parser.add_argument('--embed_dim_e', type=int, default=128)
    parser.add_argument('--embed_dim_a', type=int, default=128)
    parser.add_argument('--encoder_depth', type=int, default=4)
    parser.add_argument('--decoder_depth', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--decoder_num_heads', type=int, default=8)
    parser.add_argument('--decoder_embed_dim', type=int, default=128)

    parser.add_argument('--cls_loss_ratio', type=float, default=0.005)
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='label smoothing')
    parser.add_argument('--kl_ratio', type=float, default=0.05,
                        help='kl loss ration')
  
    args = parser.parse_args()
    return args
