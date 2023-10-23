import time
from tqdm import tqdm
import copy
import math

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
from scipy.fft import dct, idct
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from options import args_parser,exp_details
from dataset import get_dataset
from models import MLP,CNNMnist,CNNFashion_Mnist,CNNCifar
from localupdate import LocalUpdate
from globalupdate import average_weights, test_inference


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'

    train_dataset, test_dataset, user_samples = get_dataset(args)

    global_model = None
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # 图像大小
        img_size = train_dataset[0][0].shape
        # 输入特征数
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in,dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()
    global_weights_ = copy.deepcopy(global_weights)
    # print('---------- global_weights ----------')
    # print(global_weights)
    # for key,_ in global_weights.items():
    #     print('key: ', key, '\t shape: ', global_weights[key].shape)


    train_loss,train_accuracy = [],[]
    val_acc_list,net_list = [],[]
    cv_loss,cv_acc = [],[]
    # print_every = 2     # 每 print_every 轮打印训练后的全局信息
    val_loss_pre,counter = 0,0

    # 记录各客户端本地模型每一层的参数个数
    weights_numbers = copy.deepcopy(global_weights)


    for epoch in tqdm(list(range(args.epochs))):
        local_weights, local_losses = [], []
        local_weights_ = []     # 记录压缩前的本地参数

        print(f'\n | Global Training Round : {epoch + 1} |\n')

        '''Step-1 : 服务器随机选择客户端执行本地训练'''
        global_model.train()

        m = max(int(args.frac * args.num_users),1)
        index_of_users = np.random.choice(list(range(args.num_users)), m, replace=False)

        '''Step-2 : 客户端本地训练，用 for 循环模拟客户端“并行化”训练'''
        index = 0   # 客户端序号（只是为了方便）
        for k in index_of_users:
            local_model = LocalUpdate(args=args,dataset=train_dataset,
                                      index_of_samples=user_samples[k])

            w,loss = local_model.local_train(
                model=copy.deepcopy(global_model),global_round=epoch)
            # for key,_ in w.items():
            #     print(w[key].shape)

            local_weights_.append(copy.deepcopy(w))


            if args.model == 'mlp':
                for key,_ in list(w.items()):
                    '''Step-2.2 : 压缩感知'''
                    # # 获取当前层的参数个数
                    # N = 1
                    # for i in range(len(w[key].shape)):
                    #     N *= w[key].shape[i]
                    #
                    # # 记录每一层的参数个数
                    # weights_numbers[key] = torch.tensor(N)
                    #
                    # # M << N
                    # M = int(N * args.compression_ratio)
                    # # 测量矩阵
                    # Phi = math.sqrt(1 / M) * torch.randn(M, N)
                    # # 记录每一层使用的测量矩阵
                    # Phi_log[key] = Phi
                    #
                    # # 压缩
                    # y = torch.mm(Phi, w[key].reshape((-1,1)))
                    #
                    #
                    # '''Step-2.3 : 自适应参数扰动'''
                    # # epsilon 张量化
                    # epsilon = epsilon_user + torch.zeros_like(y)
                    #
                    # # 获取每一层参数中的最小值和最大值
                    # min_weight = torch.min(y)
                    # max_weight = torch.max(y)
                    # # 每一层权重或偏置的变化范围 [c-r,c+r]
                    # center = (max_weight - min_weight) / 2  # + torch.zeros_like(y)
                    # radius = max_weight - center  # + torch.zeros_like(y)
                    # # 参数与 center 之间的距离 μ
                    # miu = y - center
                    #
                    # # 伯努利采样概率 Pr[u=1]
                    # # Pr = ((y - center) * (torch.exp(epsilon) - 1) + radius * (torch.exp(epsilon) + 1)) / 2 * radius * (torch.exp(epsilon) + 1)
                    # # 伯努利变量
                    # u = torch.bernoulli(torch.rand(y.shape))
                    #
                    # # 自适应扰动
                    # for i in range(len(y)):
                    #     if u[i] == 1.:
                    #         y[i] = center + miu[i] * (math.exp(epsilon_user) + 1) / (math.exp(epsilon_user) - 1)
                    #     elif u[i] == 0.:
                    #         y[i] = center - miu[i] * (math.exp(epsilon_user) + 1) / (math.exp(epsilon_user) - 1)
                    #
                    # # 更新每一层扰动后的参数
                    # w[key] = y


                    ###################################################### 一行一行地处理
                    # 获取当前层的参数张量的维度
                    dim = w[key].ndim

                    # 获取当前层的行数 rows 、每一行所拥有的参数个数 N
                    if 1 < dim:
                        rows = w[key].shape[0]
                        N = w[key].shape[-1]
                    elif 1 == dim:
                        rows = 1
                        N = len(w[key])
                    # 记录 N
                    weights_numbers[key] = torch.tensor(N)

                    # M << N
                    M = int(N * args.compression_ratio)
                    # 测量矩阵
                    Phi = torch.randn(M, N) #* math.sqrt(1 / M)
                    # 记录每一层使用的测量矩阵（每一行使用相同的测量矩阵，只记录一次）
                    # Phi_log[key] = Phi

                    # 一行一行地处理
                    if 1 < rows:
                        # 堆叠每一行的 y
                        y_matrix = None
                        for row in range(rows):
                            # 压缩
                            y = torch.mm(Phi, w[key][row].reshape((-1, 1)))     # (M, 1)


                            '''Step-2.3 : 自适应参数扰动'''
                            # epsilon 张量化
                            epsilon = epsilon_user + torch.zeros_like(y)

                            # 获取每一层参数中的最小值和最大值
                            min_weight = torch.min(y)
                            max_weight = torch.max(y)
                            # 每一层权重或偏置的变化范围 [c-r,c+r]
                            center = (max_weight - min_weight) / 2  # + torch.zeros_like(y)
                            radius = max_weight - center  # + torch.zeros_like(y)
                            # 参数与 center 之间的距离 μ
                            miu = y - center

                            # 伯努利采样概率 Pr[u=1]
                            # Pr = ((y - center) * (torch.exp(epsilon) - 1) + radius * (torch.exp(epsilon) + 1)) / 2 * radius * (torch.exp(epsilon) + 1)
                            # 伯努利变量
                            u = torch.bernoulli(torch.rand(y.shape))

                            # 自适应扰动
                            for i in range(len(y)):
                                if u[i] == 1.:
                                    y[i] = center + miu[i] * (math.exp(epsilon_user) + 1) / (math.exp(epsilon_user) - 1)
                                elif u[i] == 0.:
                                    y[i] = center - miu[i] * (math.exp(epsilon_user) + 1) / (math.exp(epsilon_user) - 1)


                            if y_matrix is None:
                                y_matrix = copy.deepcopy(y)
                            else:
                                y_matrix = torch.hstack((y_matrix, y))

                        # 更新每一层扰动后的参数
                        w[key] = y_matrix.t()   # (rows, M)

                    elif 1 == rows:
                        y = torch.mm(Phi, w[key].reshape((-1, 1)))
                        # print(y)


                        '''Step-2.3 : 自适应参数扰动'''
                        # epsilon 张量化
                        epsilon = epsilon_user + torch.zeros_like(y)

                        # 获取每一层参数中的最小值和最大值
                        min_weight = torch.min(y)
                        max_weight = torch.max(y)
                        # 每一层权重或偏置的变化范围 [c-r,c+r]
                        center = (max_weight - min_weight) / 2  # + torch.zeros_like(y)
                        radius = max_weight - center  # + torch.zeros_like(y)
                        # 参数与 center 之间的距离 μ
                        miu = y - center

                        # 伯努利采样概率 Pr[u=1]
                        # Pr = ((y - center) * (torch.exp(epsilon) - 1) + radius * (torch.exp(epsilon) + 1)) / 2 * radius * (torch.exp(epsilon) + 1)
                        # 伯努利变量
                        u = torch.bernoulli(torch.rand(y.shape))

                        # 自适应扰动
                        for i in range(len(y)):
                            if u[i] == 1.:
                                y[i] = center + miu[i] * (math.exp(epsilon_user) + 1) / (math.exp(epsilon_user) - 1)
                            elif u[i] == 0.:
                                y[i] = center - miu[i] * (math.exp(epsilon_user) + 1) / (math.exp(epsilon_user) - 1)

                        # 更新每一层扰动后的参数
                        w[key] = y.t()

            elif args.model == 'cnn':
                for key,_ in list(w.items()):
                    # break
                    # print('key: ', key, 'w: ', w[key])
                    # 每一层的权重个数
                    N = w[key].numel()

                    weights_numbers[key] = torch.tensor(N)

                    # M << N
                    M = max(int(args.compression_ratio * N), 1)

                    ''''''
                    # y = w[key].numpy().reshape((-1, 1))[:M, :]
                    ''''''
                    w_dct = dct(w[key].numpy().reshape((-1, 1)))
                    e = epoch
                    if e >= int(N / M):
                        e = e - int(N / M) * int(epoch / int(N / M))
                    y = w_dct[e * M:min((e + 1) * M, N), :]

                    # print('key: ', key, '\t y: ', y[-1, :])

                    ''' DP加噪 '''
                    # epsilon 张量化
                    epsilon_user = args.epsilon + np.zeros_like(y)

                    min_weight = min(y)
                    max_weight = max(y)
                    # 每一层权重或偏置的变化范围 [c-r,c+r]
                    center = (max_weight + min_weight) / 2  # + torch.zeros_like(y)
                    radius = (max_weight - center) if (max_weight - center) != 0. else 1  # + torch.zeros_like(y)
                    # print(center, radius)
                    # 参数与 center 之间的距离 μ
                    miu = y - center
                    # print('miu: ', miu)

                    # 伯努利采样概率 Pr[u==1]
                    # CAFL
                    # Pr = (np.exp(epsilon_user) - 1) / (2 * np.exp(epsilon_user))
                    # 参考文献[19]
                    Pr = ((y - center) * (np.exp(epsilon_user) - 1) + radius * (np.exp(epsilon_user) + 1)) / \
                         (2 * radius * (np.exp(epsilon_user) + 1))
                    # print('Pr: ', Pr)
                    # 伯努利变量
                    u = np.zeros_like(y)
                    for i in range(len(y)):
                        # CAFL
                        # u[i, 0] = np.random.binomial(1, Pr[i, :])
                        # 参考文献[19]
                        u[i, 0] = np.random.binomial(1, Pr[i, :])
                    # print('u: ', u)


                    for i in range(len(y)):
                        # 自适应扰动：miu
                        # if u[i, 0] > 0:
                        #     y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) + 1) / (np.exp(epsilon_user[i, :]) - 1))
                        # else:
                        #     y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) - 1) / (np.exp(epsilon_user[i, :]) + 1))
                        # 对比方案：radius
                        if u[i, 0] > 0:
                            y[i, :] = center + radius * ((np.exp(epsilon_user[i, :]) + 1) / (np.exp(epsilon_user[i, :]) - 1))
                        else:
                            y[i, :] = center - radius * ((np.exp(epsilon_user[i, :]) + 1) / (np.exp(epsilon_user[i, :]) - 1))


                    w[key] = torch.from_numpy(y)
                ######################################################


            # break
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            print(f'| client-{index + 1} : {k} finished!!! |')
            index += 1
        # break
        print(f'\n | Client Training End!!! | \n')


        '''Step-3 : 服务器端更新全局模型'''
        '''Step-3.1 : 平均聚合本地模型'''
        # 获取聚合前的全局模型每一层的形状
        shapes_global = copy.deepcopy(global_weights)
        for key,_ in list(shapes_global.items()):
            shapes_global[key] = shapes_global[key].shape
            # print(shapes_global[key])
        # break

        # print(local_weights)

        ''''''
        # global_weights_ = average_weights(local_weights_)
        # print('未压缩的全局模型：', global_weights_)
        partial_global_weights = average_weights(local_weights)
        # for key,_ in partial_global_weights.items():
        #     print(partial_global_weights[key].shape)
        # print('压缩后、重构前的全局模型：', partial_global_weights)
        # break

        # 计算本轮通信的平均损失
        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)


        '''Step-3.2 : 重构'''
        for key,_ in partial_global_weights.items():
            # break
            N = weights_numbers[key].item()
            M = max(int(args.compression_ratio * N), 1)
            # 重构矩阵
            rec_matrix = np.zeros((N, 1))
            e = epoch
            if e >= int(N / M):
                e = e - int(N / M) * int(epoch / int(N / M))
            rec_matrix[e * M:min((e + 1) * M, N), :] = partial_global_weights[key]
            x_rec = idct(rec_matrix)
            global_weights_1D = global_weights[key].numpy().reshape((-1, 1))
            global_weights_1D[e * M:min((e + 1) * M, N), :] = \
                (global_weights_1D[e * M:min((e + 1) * M, N), :] + x_rec[e * M:min((e + 1) * M, N), :]) / 2
            global_weights[key] = torch.from_numpy(global_weights_1D.reshape(shapes_global[key]))

            # global_weights[key] = partial_global_weights[key].reshape(shapes_global[key])

            print('key: ', key, '\t global_weights: ', global_weights[key].shape)


        # break
        # print('压缩后、重构后的全局模型：', global_weights)
        global_model.load_state_dict(global_weights)
        # global_model.load_state_dict(partial_global_weights)
        print(f'\n | Global Training Round : {epoch + 1} finished!!!!!!!!|\n')


        '''评估一轮通信后的全局模型'''
        list_acc, list_loss = [], []
        global_model.eval()
        # 在所有客户端上计算每一轮通信的平均训练精确率
        for k in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      index_of_samples=user_samples[k])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        # 计算所有客户端上评估后的平均精确率
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # 打印每一轮通信后的平均全局训练损失和精确率
        print(f'\nAvg Training States after {epoch + 1} global rounds:')
        print(f'Avg Training Loss : {train_loss[-1]}')
        print('Avg Training Accuracy : {:.2f}% \n'.format(100 * train_accuracy[-1]))

        if math.isnan(train_loss[-1]):
            train_loss.pop()
            train_accuracy.pop()
            break



    '''通信结束'''
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    runtime = time.time() - start_time
    print(('\n Total Run Time: {0:0.4f}'.format(runtime)))


    # 保存数据到 log
    data_log= {'Train Loss' : train_loss, 'Train Accuracy' : train_accuracy,
               'Test Loss' : test_loss, 'Test Accuracy' : test_acc}
    record = pd.DataFrame(data_log)
    record.to_csv('../log/MNIST/test_fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}]_time[{}].csv'.
                format(args.dataset, args.model, args.epochs, args.iid, args.unequal,
                       args.compression_ratio, args.epsilon, runtime))




    matplotlib.use('Agg')
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_E[{}]_iid[{}]_CR[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.iid, args.compression_ratio))

    # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(list(range(len(train_accuracy))), train_accuracy)
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/CIFAR/p_fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].png'.
    #             format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio, args.epsilon))
