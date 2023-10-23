import numpy as np
from torchvision import datasets,transforms


def mnist_iid(dataset,num_users):
    """
    分配 MNIST 数据集，模拟 IID 场景
    每个客户端拥有大小相同的数据集
    :param dataset:
    :param num_users:
    :return:
    """
    # 每个客户端拥有的样本数量
    num_samples = int(len(dataset) / num_users)
    # 数据集中的样本索引
    index_of_samples = [i for i in range(len(dataset))]
    # {客户端 k : 分配给客户端 k 的样本的索引}
    user_samples = {}

    for k in range(num_users):
        # 依次为每个客户端 k 随机分配 num_samples 个样本
        user_samples[k] = set(np.random.choice(index_of_samples,num_samples,replace=False))
        # 删除已经被选择过的样本的索引
        index_of_samples = list(set(index_of_samples) - user_samples[k])

    return user_samples


def mnist_noniid(dataset,num_users):
    """
    分配 MNIST 数据集，模拟 non-IID 场景
    公平分配：每个客户端拥有的分片数量相同
    :param dataset:
    :param num_users:
    :return:
    """
    # 6,0000 个训练样本  -->  共划分为 300 个分片，每个分片包含 200 张图像
    num_shards,num_samples = 300,200
    # 分片索引
    index_of_shards = [i for i in range(num_shards)]
    # {客户端 k : 分配给客户端 k 的样本的索引}
    user_samples = {k : np.array([]) for k in range(num_users)}
    # 数据集所有样本的索引
    index_of_samples = np.arange(num_shards * num_samples)
    # 训练集样本标签
    labels = dataset.train_labels.numpy()

    # 第一行为样本索引，第二行为标签
    samples_labels = np.vstack((index_of_samples,labels))
    # 对训练集样本的标签排序
    samples_labels = samples_labels[:,samples_labels[1,:].argsort()]
    # 排序后标签对应的样本索引
    index_of_samples = samples_labels[0,:]

    for k in range(num_users):
        # 每个客户端随机分配 2 ==> int(num_shards / num_users) 个分片
        random_2_shards = set(np.random.choice(index_of_shards, int(num_shards / num_users), replace=False))
        # 删除已被选择的分片的索引
        index_of_shards = list(set(index_of_shards) - random_2_shards)
        # 将这 2 个分片拼接
        for shard in random_2_shards:
            user_samples[k] = np.concatenate(
                (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

    return user_samples


def mnist_noniid_unequal(dataset,num_users):
    """
    分配 MNIST 数据集，模拟 non-IID 场景
    不公平分配：每个客户端拥有的分片数量不同
    :param dataset:
    :param num_users:
    :return:
    """
    # 6,0000 个训练样本  -->  共划分为 1200 个分片，每个分片包含 50 张图像
    num_shards,num_samples = 1200,50
    # 分片索引
    index_of_shards = [i for i in range(num_shards)]
    # {客户端 k : 分配给客户端 k 的样本的索引}
    user_samples = {k : np.array([]) for k in range(num_users)}
    # 数据集所有样本的索引
    index_of_samples = np.arange(num_shards * num_samples)
    # 训练集样本标签
    labels = dataset.train_labels.numpy()

    # 第一行为样本索引，第二行为标签
    samples_labels = np.vstack((index_of_samples, labels))
    # 对训练集样本的标签排序
    samples_labels = samples_labels[:, samples_labels[1, :].argsort()]
    # 排序后标签对应的样本索引
    index_of_samples = samples_labels[0, :]

    # 客户端可以拥有的最小和最大分片数量
    min_shards = 1
    max_shards = 30

    # 为每个客户端预分配随机数量的分片
    # 注意：此时分片的总数可能 != num_shards
    random_shards_size = np.random.randint(min_shards,max_shards + 1,size=num_users)
    # 按比例 random_shards_size / sum(random_shards_size) 预分配分片
    # 注意：此时分片的总数仍然可能 != num_shards
    random_shards_size = np.around(random_shards_size / sum(random_shards_size)
                                   * num_shards)
    random_shards_size = random_shards_size.astype(int)

    # 为客户端随机分配分片
    # 当预分配的分片总数 > num_shards 时
    if sum(random_shards_size) > num_shards:
        # 首先给每个客户端都分配 1 个分片
        # 保证每个客户端至少有 1 个分片的样本数据
        for k in range(num_users):
            random_1_shard = set(np.random.choice(index_of_shards,1,replace=False))
            # 删除已被选择的分片
            index_of_shards = list(set(index_of_shards) - random_1_shard)
            for shard in random_1_shard:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

        # 预分配的分片数量减 1
        random_shards_size = random_shards_size - 1

        # 然后，根据 random_shards_size 对剩余的分片随机分配
        for k in range(num_users):
            # 当剩余分片数量为 0 时，不再为剩余的客户端分配分片
            if len(index_of_shards) == 0:
                continue
            # 当前客户端 k 还能获得的分片数量
            shards_size = random_shards_size[k]
            # 当 shards_size 超过剩余分片总数时，将剩余所有分片分配给该客户端
            if shards_size > len(index_of_shards):
                shards_size = len(index_of_shards)
            # 随机分配分片
            random_shards = set(np.random.choice(index_of_shards,shards_size,replace=False))
            # 删除已选分片
            index_of_shards = list(set(index_of_shards) - random_shards)
            # 分配分片对应的样本
            for shard in random_shards:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

    # 当与分配的分片总数 <= num_shards 时
    else:
        for k in range(num_users):
            # 分配给当前客户端的分片数量
            shards_size = random_shards_size[k]
            random_shards = set(np.random.choice(index_of_shards,shards_size,replace=False))
            index_of_shards = list(set(index_of_shards) - random_shards)
            for shard in random_shards:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

        # 若还剩余有分片时，将剩余的分片全部分配给样本数最少的客户端
        if len(index_of_shards) > 0:
            # 剩余的分片数量
            shards_size = len(index_of_shards)
            # 样本数最少的客户端索引
            k = min(user_samples,key=lambda x: len(user_samples.get(x)))
            random_shards = set(np.random.choice(index_of_shards,shards_size,replace=False))
            index_of_shards = list(set(index_of_shards) - random_shards)
            for shard in random_shards:
                user_samples[k] = np.concatenate(
                    (user_samples[k],index_of_samples[shard * num_samples : (shard + 1) * num_samples]),axis=0)

    return user_samples


def cifar_iid(dataset,num_users):
    """
    分配 CIFAR10 数据集，模拟 IID 场景
    每个客户端拥有大小相同的数据集
    :param dataset:
    :param num_users:
    :return:
    """
    # 每个客户端拥有的样本数量
    num_samples = int(len(dataset) / num_users)
    # 数据集中的样本索引
    index_of_samples = [i for i in range(len(dataset))]
    # {客户端 k : 分配给客户端 k 的样本的索引}
    user_samples = {}

    for k in range(num_users):
        # 依次为每个客户端 k 随机分配 num_samples 个样本
        user_samples[k] = set(np.random.choice(index_of_samples, num_samples, replace=False))
        # 删除已经被选择过的样本的索引
        index_of_samples = list(set(index_of_samples) - user_samples[k])

    return user_samples


def cifar_noniid(dataset,num_users):
    """
    分配 CIFAR10 数据集，模拟 non-IID 场景
    公平分配：每个客户端拥有的分片数量相同
    :param dataset:
    :param num_users:
    :return:
    """
    # 5,0000 个训练样本  -->  共划分为 200 个分片，每个分片包含 250 张图像
    num_shards, num_samples = 200, 250
    # 分片索引
    index_of_shards = [i for i in range(num_shards)]
    # {客户端 k : 分配给客户端 k 的样本的索引}
    user_samples = {k: np.array([]) for k in range(num_users)}
    # 数据集所有样本的索引
    index_of_samples = np.arange(num_shards * num_samples)
    # 训练集样本标签
    labels = np.array(dataset.targets)
    # print('labels: ',labels)

    # 第一行为样本索引，第二行为标签
    samples_labels = np.vstack((index_of_samples, labels))
    # 对训练集样本的标签排序
    samples_labels = samples_labels[:, samples_labels[1, :].argsort()]
    # 排序后标签对应的样本索引
    index_of_samples = samples_labels[0, :]

    for k in range(num_users):
        # 每个客户端随机分配 2 个分片
        random_2_shards = set(np.random.choice(index_of_shards, 2, replace=False))
        # 删除已被选择的分片的索引
        index_of_shards = list(set(index_of_shards) - random_2_shards)
        # 将这 2 个分片拼接
        for shard in random_2_shards:
            user_samples[k] = np.concatenate(
                (user_samples[k], index_of_samples[shard * num_samples: (shard + 1) * num_samples]), axis=0)

    return user_samples


def get_dataset(args):
    """
    获取训练集和测试集，以及“客户端：样本”字典
    其中，“客户端：样本”字典表示每个客户端所分配到样本集合
    """

    train_dataset,test_dataset,user_samples = None,None,None

    if args.dataset == 'cifar':
        # 存储下载的数据集的目录地址
        data_dir = '../../CAFL-conf/data/cifar'

        # 图像预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        # 获取数据集
        train_dataset = datasets.CIFAR10(data_dir,train=True,download=True,
                                         transform=transform)
        test_dataset = datasets.CIFAR10(data_dir,train=False,download=True,
                                        transform=transform)
        # print(train_dataset[0][0].shape)

        # 为客户端分配样本
        if args.iid:
            user_samples = cifar_iid(train_dataset,args.num_users)
        else:
            if args.unequal:    # 不公平分配
                raise NotImplementedError()
            else:   # 公平分配
                user_samples = cifar_noniid(train_dataset,args.num_users)


    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../../CAFL-conf/data/mnist'
        else:
            data_dir = '../../CAFL-conf/data/fmnist'

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081))])

        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir,train=True,download=True,
                                           transform=transform)
            test_dataset = datasets.MNIST(data_dir,train=False,download=True,
                                          transform=transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                          transform=transform)

        # print('train_dataset: ', train_dataset[0])

        if args.iid:
            user_samples = mnist_iid(train_dataset,args.num_users)
        else:
            if args.unequal:
                user_samples = mnist_noniid_unequal(train_dataset,args.num_users)
            else:
                user_samples = mnist_noniid(train_dataset,args.num_users)

    return train_dataset,test_dataset,user_samples


# from options import args_parser
# args = args_parser()
# train_dataset,test_dataset,user_samples = get_dataset(args)
# print(train_dataset[5000][0])
# print(train_dataset.train_labels)   # MNIST 数据集是无序的

# user_samples = mnist_noniid(train_dataset,args.num_users)
# print(user_samples)

# users_samples = mnist_noniid_unequal(train_dataset,args.num_users)
# print(users_samples)