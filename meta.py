import torch
from  torch import nn
from  torch.nn import functional as F
from  torch import optim
import  numpy as np
from collections import OrderedDict

from copy import deepcopy


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args.dataset == 'defect_C':
            self.channel = 1
            self.num_classes = 3
        elif self.args.dataset == 'defect_G':
            self.channel = 1
            self.num_classes = 3
        elif self.args.dataset == 'MiniImagenet':
            self.channel = 3
            self.num_classes = 5

        if self.args.backbone_class == 'Res12':
            hdim = 460*6*6
            from premute.neural_network.resnet import ResNet
            self.encoder = ResNet(in_channel = self.channel)
        elif self.args.backbone_class == 'Conv-4':
            hdim = self.args.c_z_dim
            from premute.neural_network.convnet import ConvNet
            self.encoder = ConvNet(in_channels = self.channel, hid_dim=self.args.c_hid_dim, z_dim=self.args.c_z_dim)
        elif self.args.backbone_class == 'Res18':
            hdim = 512
            from premute.neural_network.resnet18 import ResNetMAML
            self.encoder = ResNetMAML(inplanes = self.channel)
        elif self.args.backbone_class == 'vit':
            hdim = self.args.dim
            from premute.neural_network.vit import ViT
            self.encoder = ViT(image_size=self.args.imgsz,
                               patch_size=self.args.patch_size,
                               dim=self.args.dim,
                               depth=self.args.depth,
                               heads=self.args.heads,
                               mlp_dim=self.args.mlp_dim,
                               channels=self.channel,
                               dropout=self.args.dropout,
                               emb_dropout=self.args.emb_dropout)
        elif self.args.backbone_class == 'swin-transformer':
            hdim = int(self.args.s_embed_dim * 2 ** (len(self.args.s_depths) - 1))
            from premute.neural_network.swin_transformer import SwinTransformer
            self.encoder = SwinTransformer(in_chans=self.channel,
                                           patch_size=self.args.s_patch_size,
                                           window_size=self.args.s_window_size,
                                           embed_dim=self.args.s_embed_dim,
                                           depths=self.args.s_depths,
                                           num_heads=self.args.s_num_heads)

        else:
            raise ValueError('not find')

        self.update_lr = self.args.update_lr   # inner learning rate
        self.meta_lr = self.args.meta_lr       # outer learning rate

        self.hdim = hdim
        self.update_step = self.args.update_step
        self.update_step_test = self.args.update_step_test

        if self.args.dataset == 'MiniImagenet':
            self.encoder.fc = nn.Linear(self.hdim, self.num_classes)
            self.fcone = nn.Linear(self.hdim, 1)
        elif self.args.dataset == 'defect_G':
            self.encoder.fc = nn.Linear(self.hdim, self.num_classes)
            self.fcone = nn.Linear(self.hdim, 1)
        elif self.args.dataset == 'defect_C':
            self.encoder.fc = nn.Linear(self.hdim, self.num_classes)
            self.fcone = nn.Linear(self.hdim, 1)

        # self.net = Learner(config, args.imgc, args.imgsz)
        # self.meta_optim = optim.Adam(self.encoder.parameters(), lr=self.meta_lr)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.meta_optim,
        #                                                       T_max=100, eta_min=0.00001)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def get_per_step_loss_importance_vector(self, args, epoch):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        用于计算损失加权平均值的张量，对 MSL（多步损失）机制很有用。
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * \
                       (1.0 / self.args.number_of_training_steps_per_iter)     # 初始均等，number_of_training_steps_per_iter多少训练步数
        decay_rate = 1.0 / 5 / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def update_param(self, loss, params, acc_gradients, step_size=0.5, first_order=True):
        name_list, tensor_list = zip(*params.items())
        # print(tensor_list)
        grads = torch.autograd.grad(loss, tensor_list)
        # print(grads)
        updated_params = OrderedDict()
        for name, param, grad in zip(name_list, tensor_list, grads):
            updated_params[name] = param - self.update_lr * grad
            # accumulate gradients for final updates
            if name == 'fc.weight':
                acc_gradients[0] = acc_gradients[0] + grad
            if name == 'fc.bias':
                acc_gradients[1] = acc_gradients[1] + grad

        return updated_params, acc_gradients

    def forward(self, support, label_s, query, label_q, epoch):   #
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        args = self.args
        task_num, setsz, c_, h, w = support.size()
        querysz = query.size(1)

        losses_q = [0 for _ in range(self.update_step)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step)]
        for i in range(task_num):
            if args.dataset == 'MiniImagenet':
                self.encoder.fc.weight.data = self.fcone.weight.data.repeat(args.n_way, 1)
                self.encoder.fc.bias.data = self.fcone.bias.data.repeat(args.n_way)
            elif args.dataset == 'Data_ISIC':
                self.encoder.fc.weight.data = self.fcone.weight.data.repeat(args.isic_n_way, 1)
                self.encoder.fc.bias.data = self.fcone.bias.data.repeat(args.isic_n_way)
            elif args.dataset == 'defect_G':
                self.encoder.fc.weight.data = self.fcone.weight.data.repeat(args.g_n_way, 1)
                self.encoder.fc.bias.data = self.fcone.bias.data.repeat(args.g_n_way)
            elif args.dataset == 'defect_C':
                self.encoder.fc.weight.data = self.fcone.weight.data.repeat(args.c_n_way, 1)
                self.encoder.fc.bias.data = self.fcone.bias.data.repeat(args.c_n_way)

            updated_params = OrderedDict(self.encoder.named_parameters())
            meta_optim = optim.Adam(self.encoder.parameters(), lr=args.meta_lr)
            acc_gradients = [torch.zeros_like(updated_params['fc.weight']), torch.zeros_like(updated_params['fc.bias'])]

            # inner train step
            logits = self.encoder(support[i], params=updated_params)
            # print(logits.shape)
            loss = F.cross_entropy(logits, label_s[i])
            updated_params, acc_gradients = self.update_param(loss, updated_params, acc_gradients,
                                                          step_size=self.update_lr, first_order=True)
            if args.dataset == 'MiniImagenet':
                updated_params['fc.weight'] = self.fcone.weight.repeat(args.n_way, 1) - \
                                              self.update_lr * acc_gradients[0]
                updated_params['fc.bias'] = self.fcone.bias.repeat(args.n_way) - \
                                            self.update_lr * acc_gradients[1]
            elif args.dataset == 'Data_ISIC':
                updated_params['fc.weight'] = self.fcone.weight.repeat(args.isic_n_way, 1) - \
                                              self.update_lr * acc_gradients[0]
                updated_params['fc.bias'] = self.fcone.bias.repeat(args.isic_n_way) - \
                                            self.update_lr * acc_gradients[1]
            elif args.dataset == 'defect_G':
                updated_params['fc.weight'] = self.fcone.weight.repeat(args.g_n_way, 1) - \
                                              self.update_lr * acc_gradients[0]
                updated_params['fc.bias'] = self.fcone.bias.repeat(args.g_n_way) - \
                                            self.update_lr * acc_gradients[1]
            elif args.dataset == 'defect_C':
                updated_params['fc.weight'] = self.fcone.weight.repeat(args.c_n_way, 1) - \
                                              self.update_lr * acc_gradients[0]
                updated_params['fc.bias'] = self.fcone.bias.repeat(args.c_n_way) - \
                                            self.update_lr * acc_gradients[1]

            logits_q = self.encoder(query[i], params=updated_params)
            loss_q = F.cross_entropy(logits_q, label_q[i])
            losses_q[0] += loss_q
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, label_q[i]).sum().item()
                corrects[0] = corrects[0] + correct

            for k in range(1, self.update_step):
                logits = self.encoder(support[i], params=updated_params)
                loss = F.cross_entropy(logits, label_s[i])
                updated_params, acc_gradients = self.update_param(loss, updated_params, acc_gradients,
                                                              step_size=self.update_lr, first_order=True)

                if args.dataset == 'MiniImagenet':
                    updated_params['fc.weight'] = self.fcone.weight.repeat(args.n_way, 1) - \
                                                  self.update_lr * acc_gradients[0]
                    updated_params['fc.bias'] = self.fcone.bias.repeat(args.n_way) - \
                                                self.update_lr * acc_gradients[1]
                elif args.dataset == 'Data_ISIC':
                    updated_params['fc.weight'] = self.fcone.weight.repeat(args.isic_n_way, 1) - \
                                                  self.update_lr * acc_gradients[0]
                    updated_params['fc.bias'] = self.fcone.bias.repeat(args.isic_n_way) - \
                                                self.update_lr * acc_gradients[1]
                elif args.dataset == 'defect_G':
                    updated_params['fc.weight'] = self.fcone.weight.repeat(args.g_n_way, 1) - \
                                                  self.update_lr * acc_gradients[0]
                    updated_params['fc.bias'] = self.fcone.bias.repeat(args.g_n_way) - \
                                                self.update_lr * acc_gradients[1]
                elif args.dataset == 'defect_C':
                    updated_params['fc.weight'] = self.fcone.weight.repeat(args.c_n_way, 1) - \
                                                  self.update_lr * acc_gradients[0]
                    updated_params['fc.bias'] = self.fcone.bias.repeat(args.c_n_way) - \
                                                self.update_lr * acc_gradients[1]

                logits_q = self.encoder(query[i], params=updated_params)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, label_q[i])
                losses_q[k] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, label_q[i]).sum().item()  # convert to numpy
                    corrects[k] = corrects[k] + correct
            # print(corrects)

        weight = Meta.get_per_step_loss_importance_vector(self, self.args, epoch + 1)
        c = zip(weight, losses_q)
        loss_q = 0
        for i, j in c:
            loss_q = loss_q + i * j
        # print(weight)
        # print(loss_q)

        # optimize theta parameters
        meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        meta_optim.step()
        # self.scheduler.step()

        accs = np.array(corrects) / (task_num * querysz)

        return accs

    def finetunning(self, support, label_s, query, label_q):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        args = self.args
        assert len(support.shape) == 4
        querysz = query.size(0)

        corrects = [0 for _ in range(self.update_step_test)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.encoder)

        if args.dataset== 'MiniImagenet':
            net.fc.weight.data = self.fcone.weight.data.repeat(args.n_way, 1)
            net.fc.bias.data = self.fcone.bias.data.repeat(args.n_way)
        elif args.dataset == 'Data_ISIC':
            net.fc.weight.data = self.fcone.weight.data.repeat(args.isic_n_way, 1)
            net.fc.bias.data = self.fcone.bias.data.repeat(args.isic_n_way)
        elif args.dataset == 'defect_G':
            net.fc.weight.data = self.fcone.weight.data.repeat(args.g_n_way, 1)
            net.fc.bias.data = self.fcone.bias.data.repeat(args.g_n_way)
        elif args.dataset == 'defect_C':
            net.fc.weight.data = self.fcone.weight.data.repeat(args.c_n_way, 1)
            net.fc.bias.data = self.fcone.bias.data.repeat(args.c_n_way)
        # 1. run the i-th task and compute loss for k=0
        updated_params = OrderedDict(net.named_parameters())
        acc_gradients = [torch.zeros_like(updated_params['fc.weight']), torch.zeros_like(updated_params['fc.bias'])]

        self.train()
        logits = net(support, params=updated_params)
        loss = F.cross_entropy(logits, label_s)
        updated_params, acc_gradients = self.update_param(loss, updated_params, acc_gradients,
                                                      step_size=self.update_lr, first_order=True)

        if args.dataset == 'MiniImagenet':
            updated_params['fc.weight'] = self.fcone.weight.repeat(args.n_way, 1) - \
                                          self.update_lr * acc_gradients[0]
            updated_params['fc.bias'] = self.fcone.bias.repeat(args.n_way) - \
                                        self.update_lr * acc_gradients[1]
        elif args.dataset == 'defect_G':
            updated_params['fc.weight'] = self.fcone.weight.repeat(args.g_n_way, 1) - \
                                          self.update_lr * acc_gradients[0]
            updated_params['fc.bias'] = self.fcone.bias.repeat(args.g_n_way) - \
                                        self.update_lr * acc_gradients[1]
        elif args.dataset == 'defect_C':
            updated_params['fc.weight'] = self.fcone.weight.repeat(args.c_n_way, 1) - \
                                          self.update_lr * acc_gradients[0]
            updated_params['fc.bias'] = self.fcone.bias.repeat(args.c_n_way) - \
                                        self.update_lr * acc_gradients[1]

        # this is the loss and accuracy after the first update
        self.eval()
        # [setsz, nway]
        logits_q = net(query, params=updated_params)
        with torch.no_grad():
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, label_q).sum().item()
            corrects[0] = corrects[0] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            self.train()
            logits = net(support, params=updated_params)
            loss = F.cross_entropy(logits, label_s)
            updated_params, acc_gradients = self.update_param(loss, updated_params, acc_gradients,
                                                          step_size=self.update_lr, first_order=True)

            if args.dataset == 'MiniImagenet':
                updated_params['fc.weight'] = self.fcone.weight.repeat(args.n_way, 1) - \
                                              self.update_lr * acc_gradients[0]
                updated_params['fc.bias'] = self.fcone.bias.repeat(args.n_way) - \
                                            self.update_lr * acc_gradients[1]
            elif args.dataset == 'defect_G':
                updated_params['fc.weight'] = self.fcone.weight.repeat(args.g_n_way, 1) - \
                                              self.update_lr * acc_gradients[0]
                updated_params['fc.bias'] = self.fcone.bias.repeat(args.g_n_way) - \
                                            self.update_lr * acc_gradients[1]
            elif args.dataset == 'defect_C':
                updated_params['fc.weight'] = self.fcone.weight.repeat(args.c_n_way, 1) - \
                                              self.update_lr * acc_gradients[0]
                updated_params['fc.bias'] = self.fcone.bias.repeat(args.c_n_way) - \
                                            self.update_lr * acc_gradients[1]

            self.eval()
            logits_q = net(query, params=updated_params)
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, label_q).sum().item()  # convert to numpy
                corrects[k] = corrects[k] + correct

        del net

        accs = np.array(corrects) / querysz

        return accs

def main():
    pass


if __name__ == '__main__':
    main()
