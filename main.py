from dataloader import H36motion3D
from torch.utils.data import DataLoader
import DS4model as Model
import torch
from utils import loss_funcs
from torch.autograd import Variable
import os
import numpy as np
import utils.data_utils as data_utils
import time
from opt import Options


def main():
    opt = Options().parse()
    date = time.strftime('%m_%d', time.localtime())
    input_n = opt.input_n
    output_n = opt.output_n
    train_data = opt.train_data
    train_batch = opt.train_batch
    linear_size = opt.linear_size
    dropout = opt.dropout
    DS_stage = opt.DS_stage
    node_n = opt.node_n
    lr = opt.lr
    test_interval = opt.test_interval
    save_interval = opt.save_interval
    loss_interval = opt.loss_interval
    actions = ['walking', 'eating', 'smoking', 'discussion', 'directions', 'greeting', 'phoning', 'posing', 'purchases',
               'sitting', 'sittingdown', 'takingphoto', 'waiting', 'walkingdog', 'walkingtogether']
    test_root = opt.test_root
    test_batch = opt.test_batch
    Pretrain = True
    pre_path = 'ckpt/12_09_0/reg_ckpt/ckpt14.pth.tar'

    ckpt_root = ''
    for i in range(10):
        if not os.path.exists('ckpt/' + date + '_' + str(i)):
            ckpt_root = 'ckpt/' + date + '_' + str(i) + '/'
            os.makedirs(ckpt_root)
            break
    reg_ckpt = 'reg_ckpt/'
    if not os.path.exists(ckpt_root + reg_ckpt):
        os.makedirs(ckpt_root + reg_ckpt)
    logpath = ckpt_root + date + '.txt'
    epochs = opt.epochs

    is_cuda = torch.cuda.is_available()
    global eval_frame, t_3d
    # =============模型实例化=============
    model = Model.InceptionGCN(linear_size, dropout, is_cuda, DS_stage=DS_stage, node_n=66, opt=opt)
    if Pretrain:
        model.load_state_dict(torch.load(pre_path)['state_dict'])
    if is_cuda:
        model.cuda()
    # =============优化器实例化===========
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if Pretrain:
        optimizer.load_state_dict(torch.load(pre_path)['optimizer'])
    # =============数据集实例化===========
    train_dataset = H36motion3D(path_to_data=train_data, actions='all', input_n=input_n, output_n=output_n,
                                split=0, sample_rate=2)
    # ==========逐batch加载数据集=========
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    f = open(logpath, 'w+')
    err1 = [np.inf]
    err3 = [np.inf]
    err7 = [np.inf]
    err9 = [np.inf]
    err13 = [np.inf]
    err24 = [np.inf]

    for epoch in range(epochs):
        start_time = time.time()
        for i, (inputs, targets, all_seq) in enumerate(train_loader):
            model.train()
            batch_size = inputs.shape[0]
            if batch_size == 1:
                continue
            if is_cuda:
                inputs = Variable(inputs.cuda()).float()
                all_seq = Variable(all_seq.cuda()).float()
            else:
                inputs = Variable(inputs).float()
                all_seq = Variable(all_seq).float()
            # =============模型运算==============
            outputs = model(inputs)
            # outputs = outputs.transpose(0, 2, 1)
            loss = mpjpe_loss(outputs, all_seq, train_dataset.dim_used)
            if i % loss_interval == 0:
                print('Training loss: {}'.format(loss.item()))
            # =============梯度反传==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ==========每test_interval个batch进行模型测试==============
            if i % test_interval == 0 and i != 0:
                model.eval()
                ave_err = np.zeros(4)
                for action in actions:
                    test_data = os.path.join(test_root, action + '.npy')
                    test_dataset = H36motion3D(path_to_data=test_data, actions=action, input_n=input_n,
                                               output_n=output_n,
                                               split=0, sample_rate=2)
                    test_loader = DataLoader(
                        dataset=test_dataset,
                        batch_size=test_batch,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)
                    N = 0
                    if output_n == 25:
                        eval_frame = [1, 3, 7, 9, 13, 24]
                        t_3d = np.zeros(len(eval_frame))
                    elif output_n == 10:
                        eval_frame = [1, 3, 7, 9]
                        t_3d = np.zeros(len(eval_frame))

                    for t_i, (inputs, targets, all_seq) in enumerate(test_loader):
                        if is_cuda:
                            inputs = Variable(inputs.cuda()).float()
                            all_seq = Variable(all_seq.cuda()).float()
                        else:
                            inputs = Variable(inputs).float()
                            all_seq = Variable(all_seq).float()

                        # inputs = Variable(inputs).float()
                        # all_seq = Variable(all_seq).float()

                        outputs = model(inputs)
                        # ============无梯度反传===============
                        n, seq_len, dim_full_len = all_seq.data.shape
                        dim_used_len = len(train_dataset.dim_used)
                        # ===========计算MPJPE之前将22还原为32==========
                        pred_3d = all_seq.clone()
                        dim_used = np.array(train_dataset.dim_used)
                        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
                        index_to_ignore = np.concatenate(
                            (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
                        joint_equal = np.array([13, 19, 22, 13, 27, 30])
                        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

                        pred_3d[:, input_n:, dim_used] = outputs.permute(0, 2, 1)
                        pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
                        pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
                        targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]

                        for k in np.arange(0, len(eval_frame)):
                            j = eval_frame[k]
                            # ========计算第j帧预测结果与真实姿态的平均每关节的L2(MPJPE)========
                            t_3d[k] += torch.mean(torch.norm(
                                targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(
                                    -1,
                                    3),
                                2,
                                1)).cpu().unsqueeze(0).data.numpy()[0] * n

                        N += n

                    ave_err += t_3d / N
                    f.write(action + ':' + '\n')
                    for t_l in range(len(t_3d)):
                        f.write(str(t_3d[t_l] / N) + '\n')
                # ===========计算平均每类动作MPJPE============
                ave_err = ave_err / 15
                # ==========每当ave_err出现更优值保存相应模型==========
                if ave_err[0] < np.min(err1):
                    state = {'epoch': epoch + 1,
                             'lr': lr,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    torch.save(state, ckpt_root + 'ckpt' + '_80best' + '.pth.tar')
                if ave_err[1] < np.min(err3):
                    state = {'epoch': epoch + 1,
                             'lr': lr,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    torch.save(state, ckpt_root + 'ckpt' + '_160best' + '.pth.tar')
                if ave_err[2] < np.min(err7):
                    state = {'epoch': epoch + 1,
                             'lr': lr,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    torch.save(state, ckpt_root + 'ckpt' + '_320best' + '.pth.tar')
                if ave_err[3] < np.min(err9):
                    state = {'epoch': epoch + 1,
                             'lr': lr,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    torch.save(state, ckpt_root + 'ckpt' + '_400best' + '.pth.tar')
                f.write('Ep' + str(epoch) + ' average:' + '\n')
                for a_l in range(len(ave_err)):
                    f.write(str(ave_err[a_l]) + '\n')
                err1.append(ave_err[0])
                err3.append(ave_err[1])
                err7.append(ave_err[2])
                err9.append(ave_err[3])
                prog = 100 * (i * batch_size) / train_dataset.__len__()
                print('Epoch {}: {:.2f}% Err: {:.2f} {:.2f} {:.2f} {:.2f}'
                      .format(epoch, prog, ave_err[0], ave_err[1], ave_err[2], ave_err[3]))
        end_time = time.time()
        epoch_cost = end_time - start_time
        print('cost time: ' + str(epoch_cost))
        # ============每隔save_interval自动保存模型===========
        if epoch % save_interval == 0:
            state = {'epoch': epoch + 1,
                     'lr': lr,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, ckpt_root + reg_ckpt + 'ckpt' + str(epoch) + '.pth.tar')


def mpjpe_loss(outputs, all_seq, dim_used):
    """
    :param outputs:
    :param all_seq:
    :param dct_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used = np.array(dim_used)
    dim_used_len = len(dim_used)

    pred_3d = outputs
    targ_3d = all_seq[:, :, dim_used][:, 10:, :]
    pred_3d = pred_3d.permute(0, 2, 1)
    diff = pred_3d - targ_3d

    mean_3d_err = torch.mean(torch.norm(diff.reshape(-1, dim_used_len).reshape(-1, 3), 2, 1))

    loss = mean_3d_err

    return loss


if __name__ == "__main__":
    main()
