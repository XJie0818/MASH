import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="MASH")
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--mode', default='train', type=str, help='train/test')
    parser.add_argument('--dataset', default='NYCcat', type=str, help='NYCcat/TKYcat/CAcat/Exp-volume')
    parser.add_argument('--model_path', default='./Model/model_NYCcat.pkl', type=str, help='model path')
    parser.add_argument('--model_dir', default='./Model', type=str, help='model dir')
    parser.add_argument('--database', default='./Datasets', type=str, help='database dir')
    parser.add_argument('--max_sequence_length', default=20, type=int, help='max sequence length')
    parser.add_argument('--long_sequence_length', default=200, type=int, help='long sequence length')
    parser.add_argument('--min_neighborhood_num', default=200, type=int, help='candidate num')
    parser.add_argument('--mask', default=True, type=bool, help='GBM mask')

    parser.add_argument('--batch_size', default=16, help='batch size.')
    parser.add_argument('--hidden_size', default=128, type=int, help='hidden size.')  # if TKY,d=256
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate')  # if TKY,lr=0.0002
    parser.add_argument('--epochs', default=50, type=int, help='train epoch')
    parser.add_argument('--alpha', default=0.1, type=float, help='core poi rate')
    parser.add_argument('--beta', default=0.1, type=float, help='core region rate')
    parser.add_argument('--drop_ratio', default=0, type=float, help='drop ratio for Exp-robustness')

    args = parser.parse_args()
    args.model_path = Path(args.model_path)
    args.model_dir = Path(args.model_dir)
    args.database = Path(args.database)
    args.dataset_train_path = args.database / f'dataset_{args.dataset}_train.pkl'
    args.dataset_test_path = args.database / f'dataset_{args.dataset}_test.pkl'

    if args.dataset == 'NYCcat':
        args.loss_rate = 0.3
        args.dataset_file = './Datasets/checkins-nyccat.txt'
    elif args.dataset == 'TKYcat':
        args.loss_rate = 0.3
        args.dataset_file = './Datasets/checkins-tkycat.txt'
    elif args.dataset == 'CAcat':
        args.loss_rate = 0.3
        args.dataset_file = './Datasets/checkins-cacat.txt'
    elif args.dataset == 'shortnyc':
        args.loss_rate = 0.3
        args.dataset_file = './Datasets/checkins-nyccat_low30.txt'
    elif args.dataset == 'midnyc':
        args.loss_rate = 0.3
        args.dataset_file = './Datasets/checkins-nyccat_mid40.txt'
    elif args.dataset == 'highnyc':
        args.loss_rate = 0.3
        args.dataset_file = './Datasets/checkins-nyccat_high30.txt'
    elif args.dataset == 'shorttky':
        args.loss_rate = 0.3
        args.dataset_file = './Datasets/checkins-tkycat_low30.txt'
    elif args.dataset == 'midtky':
        args.loss_rate = 0.3
        args.dataset_file = './Datasets/checkins-tkycat_mid40.txt'
    elif args.dataset == 'hightky':
        args.loss_rate = 0.3
        args.dataset_file = './Datasets/checkins-tkycat_high30.txt'
    return args
