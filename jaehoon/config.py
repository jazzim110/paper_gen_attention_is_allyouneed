import argparse

def get_args():

    argp = argparse.ArgumentParser(description='Stock Movement Prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Main setting
    argp.add_argument('--sentence_model_gen', type=str, default="False")
    argp.add_argument('--input_file_dir', type=str,
                      default="/home/jaehoon/JupyterNotebook/1.python3/5.2019_2_paper/dataset/de-en/train.en")
    argp.add_argument('--output_file_dir', type=str,
                      default="/home/jaehoon/JupyterNotebook/1.python3/5.2019_2_paper/dataset/de-en/train.de")
    argp.add_argument('--batch_size', type=int, default=32)




    """
    argp.add_argument('--price_data_dir', type=str, default="./data/price/snp500")
    argp.add_argument('--financial_data_dir', type=str, default="./data/bloomberg/processed")
    argp.add_argument('--save_dir',  type=str, default="./out")
    argp.add_argument('--model_dir', type=str, default="./model")
    argp.add_argument('--model_name', type=str, default="basic")
    argp.add_argument('--save_log', action='store_true', default=False)
    argp.add_argument('--max_to_keep', type=int, default=10)
    argp.add_argument('--label_proportion', nargs='+', type=int, required=True)

    # Main Control
    argp.add_argument('--preprocess', action='store_true', default=False)
    argp.add_argument('--debug', action='store_true', default=False)
    argp.add_argument('--normal_train', action='store_true', default=False)
    argp.add_argument('--train_period', type=str, default=['2013','2014','2015'])
    argp.add_argument('--valid_period', type=str, default=['2016'])
    argp.add_argument('--test_period', type=str, default=['2017'])
    argp.add_argument('--quarter_point', default=['03-31','06-30', '9-30', '12-31']) # deprecated
    argp.add_argument('--test_start_date', type=str, default='2017-01-03')
    argp.add_argument('--non_use_feat', action='store_true', default=False)
    argp.add_argument('--feature_list', nargs='+', type=str, help='return, volume, close')
    argp.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'cluster'])
    argp.add_argument('--min_train_period', type=int, default=300)
    argp.add_argument('--use_fin', action='store_true', default=False)
    argp.add_argument('--n_fin_feats', type=int, default=19)
    argp.add_argument('--n_topk', type=int, default=10)
    argp.add_argument('--lookback', type=int, default=60)
    argp.add_argument('--n_epochs', type=int, default=200)
    argp.add_argument('--eval_step', type=int, default=50)
    argp.add_argument('--print_step', type=int, default=10)
    argp.add_argument('--early_stop_type', type=str, default='acc', choices=['acc', 'loss', 'f1', 'mcc'])
    argp.add_argument('--use_tech', action='store_true', default=False)
    argp.add_argument('--n_tech_feats', type=int, default=14)
    argp.add_argument('--use_cluster', action='store_true', default=False)
    argp.add_argument('--cluster_num', type=int, default=3)

    argp.add_argument('--sort_run', action='store_true', default=False)
    argp.add_argument('--sort_quarter', type=str, default='2013-1')
    argp.add_argument('--sort_key', type=str, default='ROE')

    ## model general
    argp.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'cnn'])
    argp.add_argument('--rnn_models', type=list, default=['rnn', 'lstm', 'gru'])
    argp.add_argument('--num_layer', type=int, default=4)
    # MLP model
    argp.add_argument('--mlp_layers', nargs='+', type=int, help='[32, 16, 8]')
    # CNN model
    argp.add_argument('--fin_conv_out', type=int, default=16)
    argp.add_argument('--fin_conv_w', type=int, default=4)
    argp.add_argument('--fin_conv_pool', type=int, default=2)
    argp.add_argument('--prc_conv_out', type=int, default=16)
    argp.add_argument('--prc_conv_w', type=int, default=4)
    argp.add_argument('--prc_conv_pool', type=int, default=2)

    ## optimizer
    argp.add_argument('--optimizer', type=str, default='Adam')
    argp.add_argument('--lr', type=float, default=1e-5)
    argp.add_argument('--weight_decay', type=float, default=1e-5)
    argp.add_argument('--dropout', type=float, default=0.1)
    argp.add_argument('--momentum', type=float, default=0.9)
    argp.add_argument('--grad-max-norm', type=float, default=2.0)
    
    """
    return argp.parse_args()