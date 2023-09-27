import numpy as np
import torch




def record_net_data_stats(y_train, net_dataidx_map, classes):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {classes[unq[i]]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp


    # print('mean:', np.mean(data_list))
    # print('std:', np.std(data_list))

    return net_cls_counts





def partition_data(y_train, partition, n_parties, classes, beta=0.4):

    n_train = y_train.shape[0]

    if partition == "random_split":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}




    elif partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 100


        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]

            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, classes)
    return ( net_dataidx_map, traindata_cls_counts)



def client_subset_creation(partition,trainSet,splits,n_clients,beta, batch_size, mode, common_dataset_size):


    if mode == 'distillation':


        total_len= int(len(trainSet)*(1-common_dataset_size))
        list_server=[int(len(trainSet)*common_dataset_size)]




    else:
        list_server = [int(len(trainSet) * 0.01)]
        total_len = int(len(trainSet) * 0.99)

    if partition=='random_split':

        train_set, valid_set, valid_dataset_server=random_split(n_clients, trainSet, splits, total_len, mode, list_server)



    else:

        train_set, valid_set, valid_dataset_server = noniid_split(total_len, mode, trainSet, list_server, n_clients, partition, beta)



    train_loader_list, valid_loader_list, valid_loader_server=dataloader_creation(n_clients, train_set, valid_set, batch_size, mode, common_dataset_size,
                                                                                  valid_dataset_server)



    X_train, y_train = trainSet.data, np.array(trainSet.targets)




    return train_loader_list, valid_loader_list, y_train, valid_loader_server, train_set, valid_set, valid_dataset_server

def dataloader_creation(n_clients, train_set, valid_set, batch_size, mode, common_dataset_size, valid_dataset_server):


    train_loader_list = []
    valid_loader_list = []

    net_dataidx_map_tr = {}
    net_dataidx_map_v = {}

    for client in range(n_clients):
        train_loader_list.append(
            torch.utils.data.DataLoader(train_set[client], batch_size=batch_size, shuffle=True))
        valid_loader_list.append(
            torch.utils.data.DataLoader(valid_set[client], batch_size=batch_size, shuffle=False))

        net_dataidx_map_tr[client] = np.array(train_loader_list[client].dataset.indices)
        net_dataidx_map_v[client] = np.array(valid_loader_list[client].dataset.indices)

    if mode == 'distillation':

        if common_dataset_size > 0:

            valid_loader_server = torch.utils.data.DataLoader(valid_dataset_server, batch_size=batch_size,
                                                              shuffle=False)


        else:

            valid_loader_server = 0

    else:

        valid_loader_server = torch.utils.data.DataLoader(valid_dataset_server, batch_size=batch_size, shuffle=False)

    return train_loader_list, valid_loader_list, valid_loader_server








def random_split(n_clients, trainSet, splits, total_len, mode, list_server):


    split_tr = []
    split_ev = []

    for i in range(n_clients):
        if np.mod(i, 2) == 0:
            split_tr.append(int(np.floor(len(trainSet) * splits[i])))
            split_ev.append(int(np.floor(len(trainSet) * splits[n_clients + i])))
        else:
            split_tr.append(int(np.ceil(len(trainSet) * splits[i])))
            split_ev.append(int(np.ceil(len(trainSet) * splits[n_clients + i])))


    if sum(split_tr) > (total_len * 0.9):
        split_tr[-1] = int(split_tr[-1] - (sum(split_tr) - total_len * 0.9))
    elif sum(split_tr) < (total_len * 0.9):
        split_tr[-1] = int(split_tr[-1] - (sum(split_tr) - total_len * 0.9))

    if sum(split_ev) > (total_len * 0.1):
        split_ev[-1] = int(split_ev[-1] - (sum(split_ev) - total_len * 0.1))
    elif sum(split_ev) < (total_len * 0.1):
        split_ev[-1] = int(split_ev[-1] - (sum(split_ev) - total_len * 0.1))





    if mode == 'distillation':

        if list_server[0]>0:

            tot_dataset = torch.utils.data.random_split(trainSet,
                                                        split_tr + split_ev+list_server,
                                                        generator=torch.Generator().manual_seed(0))

        else:

            tot_dataset = torch.utils.data.random_split(trainSet,
                                                        split_tr + split_ev,
                                                        generator=torch.Generator().manual_seed(0))

    else:

        tot_dataset = torch.utils.data.random_split(trainSet,
                                                    split_tr + split_ev + list_server,
                                                    generator=torch.Generator().manual_seed(0))

    train_set = []
    valid_set = []

    if mode == 'distillation':

        if list_server[0]>0:

            valid_dataset_server = tot_dataset[-1]
            temp_dataset = tot_dataset[:-1]

        else:

            valid_dataset_server = 0
            temp_dataset = tot_dataset

    else:

        valid_dataset_server = tot_dataset[-1]
        temp_dataset = tot_dataset[:-1]




    for i in range(n_clients):
        train_set.append(temp_dataset[i])
        valid_set.append(temp_dataset[n_clients + i])




    return train_set, valid_set, valid_dataset_server





def noniid_split(total_len, mode, trainSet, list_server, n_clients, partition, beta ):

    total_len = [total_len]

    if mode == 'distillation':

        if list_server[0] > 0:

            tot_dataset = torch.utils.data.random_split(trainSet,
                                                        total_len + list_server,
                                                        generator=torch.Generator().manual_seed(0))

        else:

            tot_dataset = torch.utils.data.random_split(trainSet,
                                                        total_len,
                                                        generator=torch.Generator().manual_seed(0))


    else:

        tot_dataset = torch.utils.data.random_split(trainSet,
                                                    total_len + list_server,
                                                    generator=torch.Generator().manual_seed(0))

    if mode == 'distillation':

        if list_server[0] > 0:

            valid_dataset_server = tot_dataset[-1]
            temp_dataset = tot_dataset[0]

        else:

            valid_dataset_server = 0
            temp_dataset = tot_dataset[0]

    else:

        valid_dataset_server = tot_dataset[-1]
        temp_dataset = tot_dataset[0]


    temp = np.array(trainSet.targets)
    y_train = temp[np.array(temp_dataset.indices)]

    classes=trainSet.classes
    net_dataidx_map_init, traindata_cls_counts = partition_data(y_train, partition, n_clients, classes, beta=beta)

    net_dataidx_map = {}

    for i in range(len(net_dataidx_map_init)):
        net_dataidx_map[i]=[]




    for i in range(len(net_dataidx_map_init)):
        for j in range(len(net_dataidx_map_init[i])):
            net_dataidx_map[i].append(temp_dataset.indices[net_dataidx_map_init[i][j]])


    train_set = []
    valid_set = []

    for client in range(n_clients):
        idxs = np.random.permutation(net_dataidx_map[client])
        #idxs = net_dataidx_map[client]
        delimiter = int(np.ceil(0.9 * len(idxs)))

        train_set.append(torch.utils.data.Subset(trainSet, idxs[:delimiter]))
        valid_set.append(torch.utils.data.Subset(trainSet, idxs[delimiter:]))


        # import collections
        #
        # trainset = train_set[client]
        # valset=valid_set[client]
        # targets_np = np.array(trainset.dataset.targets)
        # inds_np=np.concatenate((trainset.indices, valset.indices))
        #
        # targets = targets_np[inds_np]
        #
        # train_counter = collections.Counter(targets)
        # classes = trainset.dataset.classes
        #
        # for i, c in enumerate(classes):
        #     print(f"{c}: {train_counter[i]} images")


    return train_set, valid_set, valid_dataset_server


