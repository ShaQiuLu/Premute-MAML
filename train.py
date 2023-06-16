import numpy as np
import scipy.stats
from torch.utils.data import DataLoader
import argparse
from meta import Meta
import torch, gc

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    # print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gc.collect()
    torch.cuda.empty_cache()

    maml = Meta(args).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(maml)
    print('Total trainable tensors:', num)

    if args.dataset == 'MiniImagenet':
        from premute.Data.MiniImagenet import MiniImagenet
        # batchsz here means total episode number
        train_set = MiniImagenet('../Data/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz=100, resize=args.imgsz)
        test_set = MiniImagenet('../Data/', mode='test', n_way=args.t_way, k_shot=args.t_spt,
                                 k_query=args.t_qry,
                                 batchsz=10, resize=args.imgsz)
    elif args.dataset == 'Data_ISIC':
        from other_py.isic import ISIC
        # batchsz here means total episode number
        train_set = ISIC('../Data_ISIC/', mode='train', n_way=args.isic_n_way, k_shot=args.isic_k_spt,
                            k_query=args.isic_k_qry,
                            batchsz=100, resize=args.imgsz)
        test_set = ISIC('../Data_ISIC/', mode='test', n_way=args.isic_t_way, k_shot=args.isic_t_spt,
                                 k_query=args.isic_t_qry,
                                 batchsz=10, resize=args.imgsz)
    elif args.dataset == 'defect_G':
        from premute.Data.defect_G import defect
        train_set = defect('../Data_G/', mode='train', n_way=args.g_n_way, k_shot=args.g_k_spt,
                         k_query=args.g_k_qry,
                         batchsz=100, resize=args.g_imgsz)
        test_set = defect('../Data_G/', mode='test', n_way=args.g_t_way, k_shot=args.g_t_spt,
                        k_query=args.g_t_qry,
                        batchsz=10, resize=args.g_imgsz)
    elif args.dataset == 'defect_C':
        from premute.Data.defect_C import defect
        train_set = defect('../Data_C/', mode='train', n_way=args.c_n_way, k_shot=args.c_k_spt,
                         k_query=args.c_k_qry,
                         batchsz=120, resize=args.c_imgsz)
        test_set = defect('../Data_C/', mode='test', n_way=args.c_t_way, k_shot=args.c_t_spt,
                        k_query=args.c_t_qry,
                        batchsz=10, resize=args.c_imgsz)

    for epoch in range(1, args.epoch+1):
        # fetch meta_batchsz num of episode each time
        train_loader = DataLoader(train_set, 3, shuffle=True, num_workers=1, pin_memory=False)

        for step, (support, query, support_y, query_y) in enumerate(train_loader):
            if torch.cuda.is_available():
                support = support.cuda()
                query = query.cuda()
                support_y = support_y.cuda()
                query_y = query_y.cuda()

            accs = maml(support, support_y, query, query_y, epoch)

            if step % 5 == 0:
                print('epoch:', epoch, '\tstep:', step, '\ttraining acc:', accs)

            if step % 10 == 0:  # evaluation
                test_loader = DataLoader(test_set, 1, shuffle=True, num_workers=1, pin_memory=False)
                accs_all_test = []
                for support, query, support_y, query_y in test_loader:
                    support = support.squeeze(0)
                    query = query.squeeze(0)
                    support_y = support_y.squeeze(0)
                    query_y = query_y.squeeze(0)

                    if torch.cuda.is_available():
                        support = support.cuda()
                        query = query.cuda()
                        support_y = support_y.cuda()
                        query_y = query_y.cuda()

                    accs = maml.finetunning(support, support_y, query, query_y)
                    accs_all_test.append(accs)

                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)

                print('epoch:', epoch, '\tstep:', step, '\tTest acc:', accs)

            if step == 39:
                print('epoch:', epoch, '\tstep:', step, '\ttraining acc:', accs)

                test_loader = DataLoader(test_set, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                for support, query, support_y, query_y in test_loader:
                    support = support.squeeze(0)
                    query = query.squeeze(0)
                    support_y = support_y.squeeze(0)
                    query_y = query_y.squeeze(0)

                    if torch.cuda.is_available():
                        support = support.cuda()
                        query = query.cuda()
                        support_y = support_y.cuda()
                        query_y = query_y.cuda()

                    accs = maml.finetunning(support, support_y, query, query_y)
                    accs_all_test.append(accs)

                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)

                print('epoch:', epoch, '\tstep:', step, '\tTest acc:', accs)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=600)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--t_way', type=int, help='n way', default=5)
    argparser.add_argument('--t_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--t_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--episodes_per_epoch', type=int, default=100)
    argparser.add_argument('--backbone_class', type=str, default='Conv-4', choices=['Res12', 'Conv-4', 'Res18', 'vit', 'swin-transformer'])
    argparser.add_argument('--dataset', type=str, default='defect_C', choices=['Data_ISIC', 'MiniImagenet',
                                                                                   'defect_C', 'defect_G'])
    argparser.add_argument('--number_of_training_steps_per_iter', type=int, help='', default=6)
    argparser.add_argument('--multi_step_loss_num_epochs', type=int, help='', default=10)

    # defect_G dataset
    argparser.add_argument('--g_imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--g_n_way', type=int, help='n way', default=3)
    argparser.add_argument('--g_k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--g_k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--g_t_way', type=int, help='n way', default=3)
    argparser.add_argument('--g_t_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--g_t_qry', type=int, help='k shot for query set', default=5)

    # defect_C dataset
    argparser.add_argument('--c_imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--c_n_way', type=int, help='n way', default=3)
    argparser.add_argument('--c_k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--c_k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--c_t_way', type=int, help='n way', default=3)
    argparser.add_argument('--c_t_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--c_t_qry', type=int, help='k shot for query set', default=5)

    # Conv-4
    argparser.add_argument('--c_hid_dim', type=int, default=64)
    argparser.add_argument('--c_z_dim', type=int, default=128)

    # vit
    argparser.add_argument('--patch_size', type=int, default=6)
    argparser.add_argument('--dim', type=int, default=512)
    argparser.add_argument('--depth', type=int, default=8)
    argparser.add_argument('--heads', type=int, default=8)
    argparser.add_argument('--mlp_dim', type=int, default=768)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--emb_dropout', type=float, default=0.5)

    # swin-transformer
    argparser.add_argument('--s_patch_size', type=int, default=4)
    argparser.add_argument('--s_window_size', type=int, default=7)
    argparser.add_argument('--s_embed_dim', type=int, default=96)
    argparser.add_argument('--s_depths', type=tuple, default=(2, 2, 6, 2))
    argparser.add_argument('--s_num_heads', type=tuple, default=(3, 6, 12, 24))

    args = argparser.parse_args()

    print('-------------train--------------')
    main()
    print('-------------end--------------')
