import numpy as np
import torch

from VGG import vgg
from resnet import *

import time
import random
from utils import *
from partition import *
from loaders import *
from torch.utils.tensorboard import SummaryWriter
from traintest import *
from aggregation import *
import os


def print_cuda_info():
    print("Is cuda available?", torch.cuda.is_available())
    print("Is cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN enabled? ", torch.backends.cudnn.enabled)
    print("Device count?", torch.cuda.device_count())
    print("Current device?", torch.cuda.current_device())
    print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))


def seed_everything(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(SEED)


def remove_unwanted(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            if ('test' in f):
                pass
            else:
                if 'pth.tar' in f:
                    os.remove(f)
            print(f)


def main(total_epochs, evaluate, path, splits, t_round, wd, normalization, n_clients, seed, count, lr, dataset,
         partition, mod, depth, mode, coef_t, coef_d,
         external_distillation, save_path, layer, average, common_dataset_size,
         loss_type, path_s, beta,
         multiloss, scale, multiloss_type, incremental_step, mu, temperature,
         incremental_rounds,  model_buffer_size, exemplar_percentage):
    writer = SummaryWriter(log_dir='./' + save_path)
    print(splits)

    seed_everything(seed)

    print('Local epochs are ' + str(t_round))
    checkpoint_prev = 0
    batch_size = 64

    lr_next = np.zeros(n_clients)

    train_loader_list, valid_loader_list, test_loader, n_classes, valid_loader_server, train_set, valid_set, valid_dataset_server, testSet = \
        loader_build(dataset, mode, partition, splits, n_clients, beta, batch_size, common_dataset_size)

    server_exemplar_size = int(len(valid_loader_server.dataset.indices) * exemplar_percentage)
    client_train_exemplar_size = int(len(train_loader_list[0].dataset.indices) * exemplar_percentage)
    client_valid_exemplar_size = int(len(valid_loader_list[0].dataset.indices) * exemplar_percentage)

    if evaluate == True:
        evaluate(test_loader, path, mod, normalization, n_classes, depth)

    momentum = 0.9
    weight_decay = wd

    if incremental_step > 0:
        total_epochs = t_round * incremental_rounds * n_classes // incremental_step

    n_rounds = total_epochs // t_round

    if n_rounds * t_round < total_epochs:
        n_rounds += 1
        total_epochs = n_rounds * t_round

    if external_distillation:

        extracted_logits = external_distillation(mod, normalization, depth, layer, n_classes, average, n_clients,
                                                 train_loader_list, multiloss, scale)



    else:

        extracted_logits = [0]

    best_prec, best_round, model, lr_next, round_n, test_r_loss, test_r_prec = communication_round_train(writer, n_rounds,n_clients,
                                                                                                         n_classes,normalization,
                                                                                mod, dataset,depth,momentum,weight_decay,lr, t_round,
                                                                                save_path,mode,train_loader_list,valid_loader_list,total_epochs,
                                                                                count, coef_t,coef_d,lr_next,checkpoint_prev,extracted_logits,
                                                                                average, external_distillation,valid_loader_server,
                                                                                common_dataset_size,layer,loss_type,
                                                                                path_s, multiloss,scale,multiloss_type,incremental_step,
                                                                                train_set,valid_set,batch_size,valid_dataset_server,server_exemplar_size,
                                                                                client_train_exemplar_size,client_valid_exemplar_size,mu,temperature,
                                                                                incremental_rounds, model_buffer_size,
                                                                                testSet,test_loader)

    writer.close()

    if (mode == 'traditional'):

        model.cuda()

        checkpoint = load_checkpoint(best_prec, str(count) + '_' + str(best_round), n_clients, save_path)

        model.load_state_dict(checkpoint['state_dict'])

        loss_server, prec_server = test(model, test_loader, False)

        save_checkpoint({
            'epoch': round_n + 1,
            'state_dict': model.state_dict(),
            'best_prec': prec_server,
            'variation': 0,
            'optimizer': 0,
        }, True, 'test_result_' + str(count),
            n_clients, filepath=save_path)

        print('The precision of the server at test set is ' + str(prec_server))


    else:

        valid_prec_clients = torch.zeros(n_clients)

        if mod == 'vgg':
            model = vgg(normalization, n_classes, depth=depth)



        elif mod == 'resnet':
            model = resnet(n_classes=n_classes, depth=depth)

        model.cuda()

        f = open(save_path + "/client_model_results.txt", "w")

        for i in range(n_clients):
            checkpoint = load_checkpoint_local('local_model_' + str(i), save_path)
            model.load_state_dict(checkpoint['state_dict'])
            loss, prec = test(model, test_loader)
            f.write(str(prec) + '\n')
            valid_prec_clients[i] = prec

        f.close()

        f = open(save_path + "/incremental_per_round_loss.txt", "w")
        f.write(str(test_r_loss) + '\n')
        f.close()

        f = open(save_path + "/incremental_per_round_prec.txt", "w")
        f.write(str(test_r_prec) + '\n')
        f.close()

        prec_server = torch.mean(valid_prec_clients).item()
        variation = torch.std(valid_prec_clients).item()

        save_checkpoint({
            'epoch': round_n + 1,
            'state_dict': 0,
            'best_prec': prec_server,
            'variation': variation,
            'optimizer': 0,
        }, True, 'test_result_' + str(count),
            n_clients, filepath=save_path)

    model.cpu()



    remove_unwanted(save_path)

    return prec_server


if __name__ == '__main__':

    path_s = '0'

    torch.cuda.set_device(int(path_s))

    torch.cuda.synchronize()

    print_cuda_info()

    argument_dict = {}

    argument_dict['path'] = ''
    argument_dict['wd_factor'] = 1e-4
    argument_dict['normalization'] = 'normal'
    argument_dict['epochs'] = 300
    argument_dict['clients'] = 10
    argument_dict['t_round'] = 10
    argument_dict['lr'] = 0.1
    argument_dict['dataset'] = 'cifar100'
    argument_dict['mod'] = 'resnet'
    argument_dict['depth'] = 56

    count = 0
    iters = 1

    argument_dict['layer'] = 'last'
    argument_dict['mode'] = 'distillation'
    argument_dict['common_dataset_size'] = 0.2
    argument_dict['coef_t'] = 0
    argument_dict['coef_d'] = 1
    argument_dict['external_distillation'] = ''
    argument_dict['average'] = False
    argument_dict['loss'] = 'mse'



    argument_dict['temperature'] = 0.5
    argument_dict['exemplar_percentage'] = 0.2

    argument_dict['incremental_step'] = 20
    argument_dict['incremental_rounds'] = 6
    argument_dict['model_buffer_size'] = 4
    argument_dict['mu'] = 0.5
    argument_dict['beta'] = 0.5

    argument_dict['partition'] = 'random_split'


    argument_dict['multiloss'] = True
    argument_dict['multiloss_type'] = 'b'
    argument_dict['scale'] = 1




    for j in range(iters):
        splits = create_splits(argument_dict)
        save_path = create_save_folder(argument_dict, path_s)

        main(argument_dict['epochs'], False, argument_dict['path'], splits, argument_dict['t_round'],
             argument_dict['wd_factor'], argument_dict['normalization']
             , argument_dict['clients'], j, count, argument_dict['lr'], argument_dict['dataset'],
             argument_dict['partition'], argument_dict['mod'],
             argument_dict['depth'], argument_dict['mode'], argument_dict['coef_t'], argument_dict['coef_d'],
             argument_dict['external_distillation'],
             save_path, argument_dict['layer'], argument_dict['average'],
             argument_dict['common_dataset_size'],  argument_dict['loss'],
             path_s, argument_dict['beta'],
             argument_dict['multiloss'], argument_dict['scale'],
             argument_dict['multiloss_type'],  argument_dict['incremental_step'],
             argument_dict['mu'],
             argument_dict['temperature'], argument_dict['incremental_rounds'],
             argument_dict['model_buffer_size'],
             argument_dict['exemplar_percentage'])
        count += 1




