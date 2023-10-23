import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset


class DatasetSplit(Dataset):
    """
    实现 PyTorch 中的 Dataset 抽象类
    """
    def __init__(self,dataset,index_of_samples):
        self.dataset = dataset
        self.index_of_samples = [int(index) for index in index_of_samples]

    def __len__(self):
        return len(self.index_of_samples)

    def __getitem__(self,item):
        image,label = self.dataset[self.index_of_samples[item]]
        return torch.tensor(image),torch.tensor(label)


class LocalUpdate(object):
    def __init__(self,args,dataset,index_of_samples):
        self.args = args
        self.trainloader,self.validloader,self.testloader = self.train_valid_test(
            dataset,list(index_of_samples))
        self.device = 'cuda' if args.gpu else 'cpu'
        # 默认以 NLL 损失函数为评估标准
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_valid_test(self, dataset, index_of_samples):
        """
        生成训练集、验证集、测试集数据加载器
        :param dataset:
        :param index_of_samples:
        :return:
        """
        # 将本地样本按照 8:1:1 的比例划分训练集、验证集、测试集
        index_of_train = index_of_samples[:int(0.8 * len(index_of_samples))]
        index_of_valid = index_of_samples[int(0.8 * len(index_of_samples)) : int(0.9 * len(index_of_samples))]
        index_of_test = index_of_samples[int(0.9 * len(index_of_samples)) :]

        # 小批量数据加载器
        train_loader = DataLoader(DatasetSplit(dataset,index_of_train),
                                  batch_size=self.args.local_bs,shuffle=True)
        valid_loader = DataLoader(DatasetSplit(dataset, index_of_valid),
                                  batch_size=max(int(len(index_of_valid) / 10), 1), shuffle=False)
        test_loader = DataLoader(DatasetSplit(dataset, index_of_test),
                                  batch_size=max(int(len(index_of_test) / 10), 1), shuffle=False)

        return train_loader,valid_loader,test_loader

    def local_train(self,model,global_round):
        """
        :param model: 从服务器下载的最新全局模型，将其作为本地模型
        :param global_round: 当前的全局通信轮数
        :return:
        """
        # 将本地模型设置为训练模式
        model.train()
        # 记录本地一个 epoch 的训练损失
        epoch_loss = []

        # 优化器
        optimizer = None
        if self.args.optimizer =='sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=self.args.lr,
                                         weight_decay=1e-4)

        # 本地迭代训练
        for iter in range(self.args.local_ep):
            # 记录本地迭代一个小批量的训练损失
            batch_loss = []
            for index_of_batch,(images,labels) in enumerate(self.trainloader):
                images,labels = images.to(self.device),labels.to(self.device)
                # 将模型参数的梯度置零
                model.zero_grad()
                # 模型预测值
                labels_hat = model(images)
                # 计算损失
                loss = self.criterion(labels_hat,labels)
                # 计算梯度
                loss.backward()
                # 更新参数
                optimizer.step()

                # 每处理 10 个小批量数据便记录训练情况
                if self.args.verbose and (index_of_batch % 10 == 0):
                    # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     global_round + 1, iter + 1, index_of_batch * len(images),
                    #     len(self.trainloader.dataset),
                    #     100. * index_of_batch / len(self.trainloader),loss.item()
                    # ))
                    batch_loss.append(loss.item())

            # 一个本地 epoch 完成之后，记录该 epoch 的平均训练损失
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 返回本地模型的参数，以及本地训练的平均损失
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """
        :param model: 服务器分发的全局模型
        :return: 本地评估后的结果
        """

        model.eval()
        loss, total, correct = 0.0, 0.0 , 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # 测试
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # 预测
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            # 获取正确的个数
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            total += len(labels)

        accuracy = correct / total
        return accuracy, loss

    # def compressive_sensing(self,weights):