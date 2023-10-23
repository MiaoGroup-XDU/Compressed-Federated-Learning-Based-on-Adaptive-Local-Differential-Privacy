import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        """
        :param dim_in: 输入特征数
        :param dim_hidden: 隐藏单元数
        :param dim_out: 输出类别数
        """
        super(MLP,self).__init__()
        # 隐藏层
        self.layer_hidden = nn.Linear(dim_in,dim_hidden)
        # print(self.layer_hidden.state_dict().get('weight').shape)
        # print(self.layer_hidden.state_dict().get('bias').shape)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        # 输出层
        self.layer_out = nn.Linear(dim_hidden,dim_out)
        # print(self.layer_out.state_dict().get('weight').shape)
        # print(self.layer_out.state_dict().get('bias').shape)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,X):
        X = X.view(-1,X.shape[1] * X.shape[-2] * X.shape[-1])
        X = self.layer_hidden(X)
        X = self.dropout(X)
        X = self.relu(X)
        X = self.layer_out(X)
        return self.softmax(X)

# global_model = MLP(dim_in=28 * 28,dim_hidden=64,dim_out=10)
# print(global_model)
# X = torch.rand(1,28 * 28)
# print(global_model(X))


# class CNNMnist(nn.Module):
#     def __init__(self,args):
#         super(CNNMnist,self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels,10,kernel_size=(5,5))
#         # print(self.conv1.state_dict().get('weight').shape)
#         # print(self.conv1.state_dict().get('bias').shape)
#         self.conv2 = nn.Conv2d(10,20,kernel_size=(5,5))
#         # print(self.conv2.state_dict().get('weight').shape)
#         # print(self.conv2.state_dict().get('bias').shape)
#         self.conv2_dropout = nn.Dropout2d()
#         self.full_connection1 = nn.Linear(320,50)
#         # print(self.full_connection1.state_dict().get('weight').shape)
#         # print(self.full_connection1.state_dict().get('bias').shape)
#         self.full_connection2 = nn.Linear(50,args.num_classes)
#         # print(self.full_connection2.state_dict().get('weight').shape)
#         # print(self.full_connection2.state_dict().get('bias').shape)
#
#     def forward(self,X):
#         X = F.relu(F.max_pool2d(self.conv1(X),2))
#         X = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(X)),2))
#         X = X.view(-1,X.shape[1] * X.shape[2] * X.shape[3])
#         X = F.relu(self.full_connection1(X))
#         X = F.dropout(X,training=self.training)
#         X = self.full_connection2(X)
#         return F.log_softmax(X,dim=1)


class CNNMnist(nn.Module):
    def __init__(self,args):
        # super(CNNMnist,self).__init__()
        # self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=(5, 5))
        # # print(self.conv1.state_dict().get('weight').shape)
        # # print(self.conv1.state_dict().get('bias').shape)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        # self.full_connection1 = nn.Linear(1024, args.num_classes)

        super(CNNMnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.full_connection = nn.Linear(64 * 4 * 4, args.num_classes)

        # super(CNNMnist, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(32, 32, 5)
        # self.fc1 = nn.Linear(32 * 4 * 4, 120)
        # self.fc2 = nn.Linear(120, 64)
        # self.fc3 = nn.Linear(64, args.num_classes)

    def forward(self, X):
        # X = self.conv1(X)
        # # print(X.shape)
        # X = F.max_pool2d(X, 2)
        # # print(X.shape)
        # X = self.conv2(X)
        # # print(X.shape)
        # X = F.max_pool2d(X, 2)
        # # print(X.shape)
        # X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        # # print(X.shape)
        # X = self.full_connection1(X)
        # # print(X.shape)
        # X = F.relu(X)
        # # print(X.shape)

        X = self.layer1(X)
        X = self.layer2(X)
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        X = self.full_connection(X)
        X = F.relu(X)
        return F.softmax(X)

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return F.log_softmax(x, dim=1)


# import options
# global_model = CNNMnist(args=options.args_parser())
# print(global_model)
# X = torch.rand((1,1,28,28))
# print(global_model(X))


class CNNFashion_Mnist(nn.Module):
    def __init__(self,args):
        super(CNNFashion_Mnist,self).__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1,16,kernel_size=(5,5),padding=2),
        #     # 数据归一化处理，参数为特征的个数
        #     # nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2))
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16,32,kernel_size=(5,5),padding=2),
        #     # nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2))
        # self.full_connection = nn.Linear(7 * 7 * 32, 10)

        self.layer1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.full_connection = nn.Linear(64 * 4 * 4, args.num_classes)

        # self.conv1 = nn.Conv2d(1, 32, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(32, 32, 5)
        # self.fc1 = nn.Linear(32 * 4 * 4, 120)
        # self.fc2 = nn.Linear(120, 64)
        # self.fc3 = nn.Linear(64, args.num_classes)

    def forward(self,X):
        # out = self.layer1(X)
        # out = self.layer2(out)
        # out = out.view(out.size(0),-1)
        # out = self.full_connection(out)
        # return out

        X = self.layer1(X)
        X = self.layer2(X)
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        X = self.full_connection(X)
        X = F.relu(X)
        return F.softmax(X)

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return F.log_softmax(x, dim=1)



# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         # self.conv1 = nn.Conv2d(3, 32, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(32, 32, 5)
#         # self.fc1 = nn.Linear(32 * 5 * 5, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, args.num_classes)
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=(5, 5)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2)))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=(5, 5)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2)))
#         self.layer3 = nn.Sequential(
#             nn.Linear(64 * 5 * 5, 120),
#             nn.Sigmoid(),
#             nn.Linear(120, 84),
#             nn.Sigmoid(),
#             nn.Linear(84, args.num_classes))
#
#     def forward(self, x):
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         # x = x.view(-1, 32 * 5 * 5)
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
#         # return F.log_softmax(x, dim=1)
#
#         x = self.layer1(x)
#         # print(x.shape)
#         x = self.layer2(x)
#         x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
#         return self.layer3(x)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 4 * 4, 1028)
        self.fc2 = nn.Linear(1028, args.num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

