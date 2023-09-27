
from partition import *
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets
from copy import copy

class CIFAR100_withIndex(datasets.CIFAR100):

    def __getitem__(self, index):
        img, label = super(CIFAR100_withIndex, self).__getitem__(index)

        return (img, label, index)
def loader_build(dataset,mode, partition,splits,n_clients,beta,batch_size, common_dataset_size):

    transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if dataset=='cifar10':

        trainSet = torchvision.datasets.CIFAR10(root='./CifarTrainData', train=True,
                                                 download=True, transform=transform)


        trainSet_val= torchvision.datasets.CIFAR10(root='./CifarTrainData', train=True,
                                                 download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))]))

        testSet = torchvision.datasets.CIFAR10(root='./CifarTrainData', train=False,
                                                download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))]))


    elif dataset=='cifar100':

        trainSet = CIFAR100_withIndex(root='./CifarTrainData', train=True,
                                                download=True, transform=transform)

        trainSet_val = CIFAR100_withIndex(root='./CifarTrainData', train=True,
                                                    download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))


        testSet = CIFAR100_withIndex(root='./CifarTrainData', train=False,
                                                 download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))


    if dataset=='cifar10':
        n_classes=10
    elif dataset=='cifar100':
        n_classes=100


    test_loader = torch.utils.data.DataLoader(testSet)


    if mode == 'distillation':


            train_loader_list, valid_loader_list, y_train, _, train_set, valid_set, valid_dataset_server = \
                client_subset_creation(partition, trainSet, splits, n_clients , beta, batch_size, mode, common_dataset_size)

            _, _, _, valid_loader_server,_,_,_ = \
                client_subset_creation(partition, trainSet_val, splits, n_clients , beta, batch_size, mode,common_dataset_size)

    elif mode=='traditional':

            train_loader_list, valid_loader_list, y_train, _, train_set, valid_set, valid_dataset_server = \
                client_subset_creation(partition,trainSet_val,splits,n_clients,beta, batch_size, mode, common_dataset_size)

            _, _, _, valid_loader_server, _, _, _ = \
                client_subset_creation(partition, trainSet, splits, n_clients , beta, batch_size, mode, common_dataset_size)







    return train_loader_list, valid_loader_list, test_loader, n_classes, valid_loader_server, train_set, valid_set, valid_dataset_server,testSet
