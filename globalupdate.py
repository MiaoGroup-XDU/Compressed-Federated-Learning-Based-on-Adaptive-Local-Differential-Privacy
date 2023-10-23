import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy import linalg


def average_weights(local_weights):
    """
    :param local_weights: 客户端本地模型参数
    :return:
    """
    # 此时只有一个客户端的模型
    avg_weights = copy.deepcopy(local_weights[0])
    # 服务器平均聚合时，是一层一层地聚合
    for key in avg_weights.keys():
        # 对剩余的客户端进行聚合
        for i in range(1,len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        # 做除法，得平均值
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))

    return avg_weights


def test_inference(args, model, test_dataset):
    """
    :param args:
    :param model:
    :param test_dataset:
    :return: 全局模型对测试集的精确率和损失
    """

    model.eval()
    loss, total, correct = 0.0, 0.0 ,0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        # 测试
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # 预测
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


def cs_omp_1(y, D, N):
    global a, tmp

    # M, N = A.shape[0], A.shape[1]
    # # 待重构的向量
    # theta = np.zeros((N))
    # # 记录基向量的矩阵
    # Base_t = A[:, 0].reshape((-1, 1))
    # # 残差值
    # r_n = y
    # product = np.zeros((N))
    # erro_rn = np.zeros((K))
    # pos_array = np.zeros((K))
    # for i in range(K):
    #     for col in range(N):
    #         product[col] = np.abs(np.dot(A[:, col], r_n))
    #     pos = np.argmax(product)
    #     if i == 0:
    #         Base_t = A[:, pos].reshape((-1, 1))
    #     else:
    #         Base_t = np.hstack((Base_t, A[:, pos].reshape((-1, 1))))
    #     A[:, pos] = np.zeros((M))
    #     aug_y = np.dot(np.dot(((np.dot(Base_t.T, Base_t)) ** -1), Base_t.T), y)
    #     # print(Base_t, aug_y)
    #     # print(Base_t.shape, aug_y.shape)
    #     r_n = y - np.dot(Base_t, aug_y)
    #     erro_rn[i] = np.linalg.norm(r_n)
    #     pos_array[i] = pos
    #     if erro_rn[i] < 1e-6:
    #         break
    # theta[pos_array.astype(int)] = aug_y.reshape((N))

    ''''''
    # L = max(math.floor(len(y) / 2), 1)
    L = N
    # L = D.shape[0]
    # 初始化残差
    residual = y
    index = np.zeros((N), dtype=int)
    for i in range(L):
        index[i] = -1
    result = np.zeros((N, 1))
    D_new = copy.deepcopy(D)
    # D_orth = Gram_Schmidt_Orthogonality(D)
    erro_rn = np.zeros((L))
    for j in range(L):
        product = np.dot(D.T, residual)  # (N, 1)
        # product = np.dot(D_orth.T, residual)
        # print('product: ', product)
        pos = np.argmax(np.fabs(product))
        index[pos] = 1
        my = np.linalg.pinv(D_new[:, index >= 0])   # (row -> N, M)
        # print('my: ', my.shape)
        a = np.dot(my, y)   # (row -> N, 1)
        # print('a: ', a.shape)

        residual = y - np.dot(D_new[:, index >= 0], a)      # (M, 1)
        # print('residual: ', residual.shape)
        D[:, pos] = np.zeros((D.shape[0]))
        # D_orth[:, pos] = np.zeros((D_orth.shape[0]))

        # print(j + 1)
        # if not np.any(residual):
        #     break
        # erro_rn[j] = np.linalg.norm(residual)
        # if erro_rn[j] < 1e-15:
        #     break
    # print(index, index.shape)
    # print(a, a.shape)
    result[index >= 0] = a

    ''''''
    # L = max(math.floor(len(y) / 4), 1)
    # # L = N
    # # print('L: ', L)
    # residual = y
    # index = np.zeros((N), dtype=int)
    # for i in range(N):
    #     index[i] = -1
    # result = np.zeros((N, 1))
    # for j in range(L):
    #     product = np.fabs(np.dot(D.T, residual))
    #     # print('product: ', product.shape)
    #     pos = np.argmax(product)
    #     # print('pos: ', pos)
    #     index[j] = pos
    #     tmp = []
    #     for tt in range(len(index)):
    #         if (index[tt] > 0) or (index[tt] == 0):
    #             tmp.append(tt)
    #     # print('tmp: ', tmp)
    #     tmp1 = D[:, tmp]
    #     my = np.linalg.pinv(D[:, tmp])
    #     # print('my: ', my.shape)
    #     a = np.dot(my, y)
    #     # print('a: ', a)
    #     # print('a.shape: ', a.shape)
    #     residual = y - np.dot(D[:, tmp], a)
    #     # print('residual: ', residual.shape)
    # # print('end tmp: ', tmp)
    # # print('end a: ', a.shape)
    # result[tmp] = a

    return result

def cs_omp_2(y, D, N):
    L = N
    residual = y
    index = np.zeros(N, dtype=int)
    for i in range(N):
        index[i] = -1
    result = np.zeros((N, 1))
    D_new = np.zeros(D.shape)
    # D_orth = Gram_Schmidt_Orthogonality(D)
    erro_rn = np.zeros((L))
    for j in range(L):
        product = np.dot(D.T, residual)
        pos = np.argmax(np.fabs(product))
        index[pos] = 1
        D_new[:, j] = D[:, pos].reshape(-1)
        my = np.linalg.pinv(D_new[:, :j+1])
        a = np.dot(my, y)
        result[index >= 0] = a
        residual = y - np.dot(D_new[:, :j+1], a)
        D[:, pos] = np.zeros((D.shape[0]))

        print(j + 1)

        # erro_rn[j] = np.linalg.norm(residual)
        # if erro_rn[j] < 1e-15:
        #     break
    return result


# 施密特正交化
def matmul_mulelms(*matrixs):
    if len(matrixs)<2:
        raise ValueError('Please input more than one parameters.')
    res = matrixs[0]
    for i in range(1,len(matrixs)):
        res = np.matmul(res, matrixs[i])
    return res

def One_Col_Matrix(array):
    mat = np.mat(array)
    if mat.shape[0] == 1:
        return mat.T
    elif mat.shape[1] == 1:
        return mat
    else:
        raise ValueError('Please input 1 row array or 1 column array')

def Transfor_Unit_Vector(matrix):
    col_num = matrix.shape[1]
    # 初始化为零矩阵
    unit_mat = np.zeros((matrix.shape))
    for col in range(col_num):
        vector = matrix[:,col]
        unit_vector = vector / np.linalg.norm(vector)
        unit_mat[:,col] = unit_vector.T
    return unit_mat

def Gram_Schmidt_Orthogonality(matrix):
    col_num = matrix.shape[1]
    # 第一列无需变换
    gram_schmidt_mat = One_Col_Matrix(matrix[:,0])
    for col in range(1,col_num):
        raw_vector = One_Col_Matrix(matrix[:,col])
        orthogonal_vector = One_Col_Matrix(matrix[:,col])
        if len(gram_schmidt_mat.shape)==1:
            # 当矩阵为列向量是，shape的返回值为“(row,)”，没有col的值
            gram_schmidt_mat_col_num = 1
        else:
            gram_schmidt_mat_col_num = gram_schmidt_mat.shape[1]
        for base_vector_col in range(gram_schmidt_mat_col_num):
            base_vector = gram_schmidt_mat[:,base_vector_col]
            prejective_vector = matmul_mulelms(base_vector, np.linalg.pinv(np.matmul(base_vector.T,base_vector)), base_vector.T, raw_vector)
            orthogonal_vector = orthogonal_vector - prejective_vector
        gram_schmidt_mat = np.hstack((gram_schmidt_mat,orthogonal_vector))
    # print(gram_schmidt_mat)
    return Transfor_Unit_Vector(gram_schmidt_mat)