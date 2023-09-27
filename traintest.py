import numpy as np
import torch
import torchnet as tnt
import torch.nn.functional as F
import torchvision.models
from torch.autograd import Variable
from VGG import vgg
from resnet import *
from utils import *
import torch.optim as optim
import time
from aggregation import *
import torch.nn as nn
import math
from partition import *
from exemplar import *
from torchvision import transforms
import random
import copy

def create_layer_optimizers(model,lr, momentum, weight_decay):

    l3_params = []
    l2_params = []
    l1_params = []

    for name, param in model.named_parameters():

        if name.startswith('linear'):
            pass

        elif name.startswith("layer3"):

            l3_params.append(param)

        elif name.startswith("layer2"):

            l3_params.append(param)
            l2_params.append(param)


        else:

            l3_params.append(param)
            l2_params.append(param)
            l1_params.append(param)

    optimizer_l1 = optim.SGD(l1_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_l2 = optim.SGD(l2_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_l3 = optim.SGD(l3_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    layer_optimizers = [optimizer_l1, optimizer_l2, optimizer_l3]


    return layer_optimizers



def communication_round_train(writer, n_rounds, n_clients, n_classes, normalization, mod, dataset, depth, momentum, weight_decay, lr, t_round, save_path,
                              mode, train_loader_list, valid_loader_list, total_epochs, count,coef_t,coef_d, lr_next, checkpoint_prev, extracted_logits, average,
                              external_distillation, valid_loader_server, common_dataset_size, layer, loss_type,
                              path, multiloss, scale, multiloss_type, incremental_step, train_set, valid_set,
                              batch_size, valid_dataset_server, server_exemplar_size, client_train_exemplar_size, client_valid_exemplar_size,
                              mu, temperature, incremental_rounds, model_buffer_size, testSet, test_loader):


    best_round=0
    best_prec=0
    incremental_round=0
    valid_loader_server_inc = 0

    if mode == 'traditional':
        valid_loader_server = valid_loader_list[-1]



    if incremental_step>0:

        exemplar_ids_tr=[]
        exemplar_ids_val = []
        exemplar_ids_srv=[]

        for c in range(n_clients):

            exemplar_ids_tr.append([])
            exemplar_ids_val.append([])



    if incremental_step>0:
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])



    per_task_soft_activation=[]
    per_round_soft_activation=0


    test_r_loss=[]
    test_r_prec=[]
    for round_n in range(n_rounds):



        per_client_output_avg = []
        per_client_layers_avg = []
        per_client_grads_avg = []
        per_client_target_avg=[]
        per_client_activation_avg = []
        per_client_tot_output_avg = []


        per_client_act_avg=[]

        if round_n == 0:
            list_name = 0
            soft_logits = [0]
            soft_logits_l=[0]





        if incremental_step>0:

            if (round_n % incremental_rounds) == 0 and (round_n>0):
                incremental_round += 1

            seen_ids = [incremental_round * incremental_step, (incremental_round + 1) * incremental_step]
            exemplar_classes = incremental_round * incremental_step
            n_classes = seen_ids[1]






        if (common_dataset_size>0) and (incremental_step>0) and (round_n % incremental_rounds  == 0) :

            feature_extractor = []
            feature_extractor_np=[]




            if mod == 'vgg':
                model_inc = vgg(normalization, dataset, depth=depth)



            elif mod == 'resnet':

                if n_classes-incremental_step>0:
                    model_inc = resnet(n_classes=n_classes-incremental_step, depth=depth)
                else:
                    model_inc = resnet(n_classes=n_classes, depth=depth)


            model_inc.cuda()


            for client in range(n_clients):

                if round_n > 0:
                    checkpoint = load_checkpoint_local('local_model_' + str(client), save_path)
                    model_inc.load_state_dict(checkpoint['state_dict'])

                    feature_extractor.append(
                        incremental_common_dataloader_creation(valid_loader_server, seen_ids, exemplar_classes,
                                                               test_transform, server_exemplar_size, model_inc,
                                                               incremental_step))


                    temp = []
                    for l in range(len(feature_extractor[0])):
                        temp.append([])

                    for c in range(len(feature_extractor)):
                        for l in range(len(feature_extractor[c])):
                            temp[l].append(feature_extractor[c][l])

                    for l in range(len(temp)):
                        feature_extractor_np.append(np.mean(temp[l], axis=0))



            valid_loader_server_temp, exemplar_ids_srv_temp = incremental_dataloader_creation(valid_loader_server,
                                                                                    valid_dataset_server, seen_ids,
                                                                                    exemplar_classes, batch_size,
                                                                                    test_transform,
                                                                                    server_exemplar_size,
                                                                                    feature_extractor_np,
                                                                                    exemplar_ids_srv, incremental_step)







            test_loader_inc, _ = incremental_dataloader_creation(
                test_loader, testSet,
                [0, seen_ids[1]], 0,
                batch_size, test_transform,
                0, model_inc, 0,
                incremental_step)




            if len(exemplar_ids_srv_temp)>0:

                exemplar_ids_srv.extend(exemplar_ids_srv_temp)

                dataloader_sampler = SubsetRandomSampler(exemplar_ids_srv,
                                                         generator=torch.Generator().manual_seed(0))

                valid_loader_server_inc = torch.utils.data.DataLoader(valid_dataset_server, batch_size=batch_size,
                                                                      sampler=dataloader_sampler)

            else:

                valid_loader_server_inc=0





            for client in range(n_clients):


                train_loader_list[client], exemplar_ids_tr[client], valid_loader_list[client], exemplar_ids_val[client], feature_extractor, feature_extractor_np  =\
                    create_all_incremental_dataloaders( model_inc, feature_extractor, client, seen_ids, test_transform,
                   train_loader_list, valid_loader_list, exemplar_classes, incremental_step,train_set, batch_size, client_train_exemplar_size,
                    exemplar_ids_tr, exemplar_ids_val, exemplar_ids_srv, valid_set,  client_valid_exemplar_size, feature_extractor_np,
                    )







            if round_n > 0:
                model_inc.cpu()



            if round_n > 0 and multiloss:

                    for i in range(len(soft_logits_l[-1])):
                        if soft_logits_l[-1][i] in exemplar_ids_srv:
                            pass
                        else:
                            for l in range(3):
                                soft_logits_l[l][i]=np.zeros_like(soft_logits_l[l][i])








        test_c_loss=[]
        test_c_prec=[]
        for client in range(n_clients):





            per_client_output_avg, per_client_act_avg, per_client_layers_avg, per_client_grads_avg, per_client_target_avg, per_client_activation_avg, per_client_tot_output_avg, \
                list_name, epoch,  writer, test_c_loss, test_c_prec= \
                client_train(per_client_output_avg, per_client_act_avg, per_client_layers_avg, per_client_grads_avg, per_client_target_avg, per_client_activation_avg, per_client_tot_output_avg, client, round_n, mod, normalization,
                             dataset, depth,  lr, momentum, weight_decay,
                             writer, save_path, t_round, lr_next, train_loader_list, valid_loader_list, total_epochs,
                             n_classes, mode, coef_t,coef_d, list_name, soft_logits, soft_logits_l, extracted_logits, average, external_distillation, valid_loader_server,
                             common_dataset_size, layer,  loss_type, path,  multiloss, scale, multiloss_type,
                             incremental_step, mu, temperature, incremental_rounds, model_buffer_size, per_round_soft_activation, per_task_soft_activation,
                             test_loader_inc, test_c_loss, test_c_prec, valid_loader_server_inc)











        if mode=='traditional':

            if mod == 'vgg':
                model = vgg(normalization, n_classes, depth=depth)

            elif mod == 'resnet':

                model = resnet(n_classes=n_classes, depth=depth)


            model.cuda()



            model=load_model_federated_traditional(model, list_name, n_clients,path)






        elif mode=='distillation':





            if (not external_distillation) :

                if common_dataset_size==0:

                    soft_logits=federated_distillation_aggregation(per_client_output_avg,per_client_activation_avg,per_client_tot_output_avg, n_classes, n_clients,
                                                                   layer)





                else:


                    per_client_output_avg_np = np.array(per_client_output_avg)

                    if multiloss:

                        if incremental_step>0:

                            for c in range(len(per_client_layers_avg)):

                                #order = np.argsort(per_client_layers_avg[c][-1])
                                order = per_client_layers_avg[c][-1].argsort()[::-1]
                                per_client_layers_avg[c][-1]=np.sort(per_client_layers_avg[c][-1])[::-1]
                                for l in range(3):
                                    per_client_layers_avg[c][l]=per_client_layers_avg[c][l][order]


                        per_client_layers_avg_np=[]

                        for l in range(3):

                            temp=[]

                            for c in range(n_clients):

                                temp.append(per_client_layers_avg[c][l])


                            per_client_layers_avg_np.append(np.array(temp))

                        per_client_layers_avg_np.append(per_client_layers_avg[0][-1])














                    soft_logits = np.mean(per_client_output_avg_np, axis=0)





                    if incremental_step>0:



                        per_client_act_avg_np=np.array(per_client_act_avg)

                        soft_activation=np.mean(per_client_act_avg_np, axis=0)


                        if ((round_n+1) % incremental_rounds ) == 0:
                            per_task_soft_activation.append(soft_activation)
                        else:
                            per_round_soft_activation=soft_activation



                    if multiloss:


                        #     for i in range(len(per_client_layers_avg_np[-1])):
                        #         if np.any(per_client_layers_avg_np[0][:,i,:,:]):
                        #             pos=per_client_layers_avg_np[-1][i]
                        #             index = np.where(soft_logits_l[-1] == pos)
                        #             for l in range(3):
                        #                 soft_logits_l[l][index, :, :] = np.mean(per_client_layers_avg_np[l][:,i,:,:])
                        #
                        # else:



                        soft_logits_l = []

                        for l in range(3):
                            soft_logits_l.append(np.mean(per_client_layers_avg_np[l], axis=0))

                        soft_logits_l.append(per_client_layers_avg_np[-1])



            model = 0












        if mode=='traditional':



            loss_server, prec_server = test(model, valid_loader_server)

            if prec_server > best_prec:
                is_best = True
                best_prec = prec_server
                best_round = round_n + 1
            else:
                is_best = False

            writer.add_scalar("Loss/valid_server", loss_server, round_n)
            writer.add_scalar("Acc/valid_server", prec_server, round_n)

            save_checkpoint_srv({
                'epoch': round_n + 1,
                'state_dict': model.state_dict(),
                'best_prec': prec_server,
                'optimizer': 0,
            }, is_best, str(count)+'_'+str(round_n + 1),  checkpoint_prev, n_clients, filepath=save_path)

            print('The precision of the server is ' + str(prec_server))

            model.cpu()

            writer.flush()


        if incremental_step>0 and ((round_n+1) % incremental_rounds ) == 0:

            test_r_loss=np.mean(test_c_loss)
            test_r_prec=np.mean(test_c_prec)





    return best_prec, best_round, model, lr_next, round_n, test_r_loss, test_r_prec





def client_train(per_client_output_avg, per_client_act_avg, per_client_layers_avg, per_client_grads_avg, per_client_target_avg, per_client_activation_avg, per_client_tot_output_avg, client,round_n, mod,normalization,dataset,depth,lr, momentum, weight_decay,
                 writer, save_path, t_round, lr_next, train_loader_list, valid_loader_list, total_epochs, n_classes, mode, coef_t,coef_d,
                 list_name, soft_logits, soft_logits_l, output_base,average, external_distillation, valid_loader_server, common_dataset_size,
                 layer,  loss_type,path, multiloss, scale, multiloss_type,
                 incremental_step, mu, temperature, incremental_rounds, model_buffer_size, per_round_soft_activation, per_task_soft_activation, test_loader,
                 test_c_loss, test_c_prec,  valid_loader_server_inc):



    print('Training of Client ' + str(client) + ' in Round ' + str(round_n))




    if mod == 'vgg':
        model = vgg(normalization, n_classes, depth=depth)



    elif mod == 'resnet':

        if n_classes - incremental_step > 0:
            if ((round_n % incremental_rounds) == 0) and (incremental_step>0):
                model= resnet(n_classes=n_classes - incremental_step, depth=depth)
            else:
                model = resnet(n_classes=n_classes , depth=depth)
        else:
            model= resnet(n_classes=n_classes, depth=depth)














    if round_n == 0:

        if client == 0:
            list_name = getName(model)


        model.cuda()



        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)



        if multiloss and (multiloss_type=='b' or multiloss_type=='c'):


            layer_optimizers = create_layer_optimizers(model, lr, momentum, weight_decay)


        else:

            layer_optimizers=0









        epoch_init = 0

    else:



        epoch_init = round_n * t_round









        if mode == 'traditional':

            checkpoint = load_checkpoint_srv(save_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=lr_next[client], momentum=momentum,
                                  weight_decay=weight_decay)


        elif mode == 'distillation':

            checkpoint_l = load_checkpoint_local('local_model_' + str(client), save_path)
            model.load_state_dict(checkpoint_l['state_dict'], strict=False)

            if incremental_step>0:
                model = incremental_learning_mod(model, n_classes, 'a')
            model.cuda()




            optimizer = optim.SGD(model.parameters(), lr=lr_next[client], momentum=momentum,
                                  weight_decay=weight_decay)

            if incremental_step==0:
                optimizer.load_state_dict(checkpoint_l['optimizer'])

            if multiloss and (multiloss_type == 'b' or multiloss_type == 'c'):

                layer_optimizers = create_layer_optimizers(model, lr_next[client], momentum, weight_decay)


            else:

                layer_optimizers = 0



            if multiloss and (multiloss_type=='b' or multiloss_type=='c'):
               for l in range(3):
                    layer_optimizers[l].load_state_dict(checkpoint_l['optimizer_layers'][l])


        if coef_d>0:

            ###distill
            distill_loss=model_distillation(valid_loader_server, epoch_init, model, optimizer, layer_optimizers, soft_logits, soft_logits_l, layer, coef_d,
                                            average, loss_type, multiloss, multiloss_type, scale, incremental_step,
                                            model_buffer_size, per_round_soft_activation, per_task_soft_activation, incremental_rounds, round_n, mu, temperature)

            writer.add_scalar("Distill loss/train" + str(client), distill_loss, epoch_init)

    train_loader = train_loader_list[client]
    valid_loader = valid_loader_list[client]




    for epoch in range(epoch_init, epoch_init + t_round):

        if incremental_step==0:
            if epoch in [int(total_epochs * 0.5), int(total_epochs * 0.75)]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

        print('LR')
        print(optimizer.param_groups[0]['lr'])
        lr_next[client] = optimizer.param_groups[0]['lr']

        start_time = time.time()





        if epoch == epoch_init + t_round - 1:
            last_epoch = True
        else:
            last_epoch = False


        ###train
        train_loss, train_prec, output_avg, activation_avg, tot_output_avg = train(train_loader, epoch, model, optimizer, coef_t, soft_logits, last_epoch,
                                    mode, output_base, average, external_distillation, common_dataset_size,layer,loss_type, round_n,
                                    incremental_step, model_buffer_size,per_round_soft_activation, per_task_soft_activation,incremental_rounds, mu, temperature)

        if valid_loader_server_inc!=0:
            train_minimum(valid_loader_server_inc, epoch, model, optimizer)




        valid_loss, valid_prec = test(model, valid_loader)

        writer.add_scalar("Loss/train" + str(client), train_loss, epoch)
        writer.add_scalar("Acc/train" + str(client), train_prec, epoch)
        writer.add_scalar("Loss/valid" + str(client), valid_loss, epoch)
        writer.add_scalar("Acc/valid" + str(client), valid_prec, epoch)

        writer.flush()


        if last_epoch:

            if common_dataset_size>0 and (mode=='distillation') :

                if layer == 'last':
                    output_base = np.zeros((50000, n_classes))



                elif layer == 'prelast':
                    output_base = np.zeros((50000, 64, 8, 8))


                if incremental_step>0:

                    act_base = np.zeros((50000, 64, 8, 8))

                else:

                    act_base = 0









                ###feature extraction
                extracted_logits, extracted_layer_logits,  extracted_act =logits_extraction(model, valid_loader_server, layer, n_classes, average,
                    output_base,act_base, multiloss, scale, incremental_step )



            if mode == 'traditional':
                save_model(model, list_name, client,path)









        elapsed_time = time.time() - start_time
        print('Elapsed time is ' + str(elapsed_time))


    if mode=='distillation':

        if common_dataset_size > 0:

            per_client_output_avg.append(extracted_logits)

            if multiloss:
                per_client_layers_avg.append(extracted_layer_logits)


            if incremental_step>0:

                per_client_act_avg.append(extracted_act)







        else:

            per_client_output_avg.append(output_avg)






    per_client_activation_avg.append(activation_avg)
    per_client_tot_output_avg.append(tot_output_avg)





    if mode=='distillation':

        if multiloss and (multiloss_type=='b' or multiloss_type=='c'):
            optimizer_layers_state_dict=[]
            for l in range(3):
                optimizer_layers_state_dict.append(layer_optimizers[l].state_dict())

            save_checkpoint_local({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_layers': optimizer_layers_state_dict,
            }, 'local_model_' + str(client), filepath=save_path)


        else:

            save_checkpoint_local({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'local_model_' + str(client), filepath=save_path)


    if (incremental_step>0) and ((round_n+1) % incremental_rounds  == 0) :



        test_loss, test_prec = test(model, test_loader)
        test_c_loss.append(test_loss)
        test_c_prec.append(test_prec)

        writer.add_scalar("Loss/test_inc" + str(client), test_loss, epoch)
        writer.add_scalar("Acc/test_inc" + str(client), test_prec, epoch)


    model.cpu()


    if multiloss and (multiloss_type == 'b' or multiloss_type == 'c'):
        for l in range(3):
            optimizer_to_cpu(layer_optimizers[l])

    optimizer_to_cpu(optimizer)
    writer.flush()


    return per_client_output_avg, per_client_act_avg, per_client_layers_avg,  per_client_grads_avg, per_client_target_avg, per_client_activation_avg, \
        per_client_tot_output_avg, list_name,epoch,  writer, test_c_loss, test_c_prec


def evaluate(test_loader, path, mod, normalization,n_classes,depth):

    if mod == 'vgg':
        model = vgg(normalization, n_classes, depth=depth)

    elif mod == 'resnet':
        model = resnet(n_classes=n_classes, depth=depth)

    model.eval()

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    test(model, test_loader)

    return

def model_distillation(train_loader, epoch, model, optimizer, optimizer_layers, soft_logits, soft_logits_l, layer, coef, average, loss_type,
                       multiloss, multiloss_type, scale, incremental_step, model_buffer_size, per_round_soft_activation, per_task_soft_activation,
                       incremental_rounds, round_n, mu, temperature):

    model.train()

    kl_loss=0
    m=0
    n=0

    if loss_type=='kl':
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        n = nn.LogSoftmax(dim=1)


    if (loss_type=='kl') or (loss_type=='ce'):
        m = nn.Softmax(dim=1)





    soft_logits_ts=torch.from_numpy(soft_logits).float().cuda()

    soft_logits_l_ts = []

    if multiloss:
        for l in range(3):
            soft_logits_l_ts.append(torch.from_numpy(soft_logits_l[l]).float().cuda())

        soft_logits_l_ts.append(soft_logits_l[-1])






    for batch_idx, (data, target, index) in enumerate(train_loader):




        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        if multiloss and (multiloss_type=='b' or multiloss_type=='c'):

            for l in range(3):
                optimizer_layers[l].zero_grad()

        output, activation, out_l1, out_l2 = model(data)

        intermediate_output=[out_l1, out_l2, activation]


        loss, loss_i =knowledge_distillation(layer,average,output, soft_logits_ts,soft_logits_l_ts, multiloss, intermediate_output,
                                    index,target,loss_type,0,coef,m,n, kl_loss, activation, scale, incremental_step,  model_buffer_size,
                           per_round_soft_activation, per_task_soft_activation,incremental_rounds, round_n, mu, temperature)











        if multiloss:


            if multiloss_type == 'a':

                for l in range(3):

                    if isinstance(loss_i[l], list):
                        pass

                    else:
                        loss_i[l].backward(retain_graph=True)



                loss.backward()

                optimizer.step()





            elif multiloss_type == 'b':

                loss.backward(retain_graph=True)

                optimizer.step()

                for l in range(3):

                    if l < 2:

                         if isinstance(loss_i[l], list):
                             pass

                         else:
                            loss_i[l].backward(retain_graph=True)

                         optimizer_layers[l].step()




                    else:

                        if isinstance(loss_i[l], list):
                            pass

                        else:

                            loss_i[l].backward()

                        optimizer_layers[l].step()


            elif multiloss_type == 'c':

                loss.backward(retain_graph=True)

                optimizer.step()

                for l in reversed(range(3)):

                    if l > 0:

                        loss_i[l].backward(retain_graph=True)

                        optimizer_layers[l].step()




                    else:

                        loss_i[l].backward()

                        optimizer_layers[l].step()


        else:

            loss.backward()

            optimizer.step()




        del output
        del activation

        # print(float(batch_idx)+1e-12)
        log_interval = 47
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f})\n'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item()))



    return loss.item()


def knowledge_distillation(layer,average,output, soft_logits_ts,soft_logits_l_ts,multiloss, intermediate_output,
                           index,target,loss_type,coef_t,coef_d,m,n, kl_loss, activation, scale, incremental_step, model_buffer_size,
                           per_round_soft_activation, per_task_soft_activation,incremental_rounds, round_n, mu, temperature):

    if layer == 'last':

        if average == False:






            if multiloss:

                soft_labels_i=[]

                for l in range(3):
                    soft_labels_i.append([])

                temp=np.zeros((len(index)), dtype=np.int32)



                for i in range(len(index)):




                    if  index[i].item() in soft_logits_l_ts[-1]:

                        temp[i]=list(np.where(soft_logits_l_ts[-1] == index[i].item())[0])[0]

                    else:

                        temp[i]=-1


                for l in range(3):

                    if incremental_step>0:


                        for r in range(len(temp)):
                            if temp[r]> -1 :
                                soft_labels_i[l].append(soft_logits_l_ts[l][temp[r],:,:])

                    else:

                        soft_labels_i[l].append(soft_logits_l_ts[l][temp, :, :])





            soft_labels = soft_logits_ts[index, :]


            if multiloss:

                soft_labels_index = []
                for r in range(len(temp)):
                    if temp[r]!=-1:
                        soft_labels_index.append(r)


            else:

                soft_labels_index = []
                for r in range(len(index)):

                    if torch.all(soft_logits_ts[index[r]]==0) :

                        pass

                    else:
                        soft_labels_index.append(r)






        else:

            soft_labels = soft_logits_ts[target, :]



        if loss_type == 'mse':

            loss_mse = nn.MSELoss()


            if incremental_step>0:


                if round_n >= incremental_rounds:

                    round_act=per_round_soft_activation[index, :,:,:]



                    task_act=[]
                    for j in range(len(per_task_soft_activation)):
                        task_act.append(per_task_soft_activation[j][index, :,:,:])

                    loss_contrast = contrastive_loss(mu, temperature, output, activation, model_buffer_size,
                            task_act, round_act ).cuda()




                else:
                    loss_contrast=0



                if len(soft_labels_index)==0:

                    loss = coef_t * F.cross_entropy(output, target).cuda() + loss_contrast

                else:

                    a = output[soft_labels_index, :soft_labels.size()[1]]
                    b = soft_labels[soft_labels_index, :]
                    c = coef_d*loss_mse(a, b)

                    if coef_t==0:
                        loss = c + loss_contrast
                    else:
                        loss = coef_t * F.cross_entropy(output, target).cuda() + c + loss_contrast





            else:
                loss = coef_t * F.cross_entropy(output, target).cuda() + coef_d * loss_mse(
                    output, soft_labels).cuda()




            if multiloss:



                    loss_i=[]

                    for l in range(3):

                        out_s=multi_scale_output_calculation(intermediate_output[l], scale).cuda()

                        if incremental_step>0:
                            out_s=out_s[soft_labels_index]


                        if len(soft_labels_i[l])==0:
                            loss_i.append([0])
                            print('ZEROLOSSSS')

                        else:
                            soft_labels_t = torch.stack(soft_labels_i[l])
                            if len(soft_labels_t.size())!=len(out_s.size()):
                                soft_labels_t=soft_labels_t.squeeze(0)

                            loss_i.append(0.1*1/3*loss_mse(out_s, soft_labels_t).cuda())


            else:

                loss_i=0




        elif loss_type == 'ce':

            soft_labels = m(soft_labels)

            loss = coef_t*F.cross_entropy(output, target).cuda()+ coef_d * F.cross_entropy(output, soft_labels).cuda()








        elif loss_type == 'kl':

            soft_labels = m(soft_labels)
            output = n(output)

            loss = coef_t*F.cross_entropy(output, target).cuda()+ coef_d * kl_loss(output, soft_labels).cuda()






    elif layer == 'prelast':

        if average == False:

            soft_labels = soft_logits_ts[index, :,:,:]
        else:

            soft_labels = soft_logits_ts[target, :,:,:]

        if loss_type == 'mse':

            loss = coef_t*F.cross_entropy(output, target).cuda()+ coef_d * F.mse_loss(activation, soft_labels).cuda()

        elif loss_type == 'ce':

            soft_labels = m(soft_labels)

            loss = coef_t*F.cross_entropy(output, target).cuda()+ coef_d * F.cross_entropy(activation, soft_labels).cuda()


        elif loss_type == 'kl':

            soft_labels = m(soft_labels)
            activation = n(activation)

            loss = coef_t*F.cross_entropy(output, target).cuda()+ coef_d * kl_loss(activation, soft_labels).cuda()






    return loss, loss_i

def train(train_loader, epoch, model, optimizer,coef,soft_logits,last_epoch,mode, output_orig,average,external_distillation, common_dataset_size,layer,
          loss_type, round_n, incremental_step, model_buffer_size,per_round_soft_activation, per_task_soft_activation,incremental_rounds, mu, temperature):
    model.train()
    train_acc = 0.
    data_sum = 0



    kl_loss = nn.KLDivLoss(reduction="batchmean")
    n = nn.LogSoftmax(dim=1)
    m = nn.Softmax(dim=1)

    if last_epoch and mode == 'distillation' and (not external_distillation):


        output_avg = []
        tot_output_avg = []
        activation_avg = []


        for i in range(len(train_loader.dataset.dataset.classes)):
            output_avg.append([])
            tot_output_avg.append([])
            activation_avg.append([])

    #common = set(train_loader.batch_sampler.sampler.indices) & set(train_loader.dataset.indices)
    for batch_idx, (data, target, index) in enumerate(train_loader):



        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output, activation, _, _ = model(data)







        if last_epoch and mode=='distillation' and (not external_distillation) and (common_dataset_size==0):






            output=m(output)

            activation_avg, output_avg, tot_output_avg = activations_calculation (output,activation,target,output_avg,activation_avg, tot_output_avg)



        else:

            output_avg=0
            activation_avg=0
            tot_output_avg = 0



        if (external_distillation or round_n>0) and coef>0 :



            if external_distillation:

                soft_logits_ts=torch.from_numpy(output_orig).float().cuda()


            else:

                if isinstance(soft_logits, np.ndarray):
                    soft_logits_ts = torch.from_numpy(soft_logits).float().cuda()
                else:
                    soft_logits_ts = torch.stack(soft_logits)



            soft_logits_l_ts=0
            multiloss=False
            intermediate_output=0
            scale=0



            loss, _ = knowledge_distillation(layer, average, output, soft_logits_ts, soft_logits_l_ts, multiloss, intermediate_output,
                                          index, target, loss_type, 1, coef,  m, n, kl_loss, activation, scale, incremental_step, model_buffer_size,
                           per_round_soft_activation, per_task_soft_activation,incremental_rounds, round_n, mu, temperature)










        else:

           loss = F.cross_entropy(output, target).cuda()


        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        data_sum += data.size()[0]
        del output
        del activation

        # print(float(batch_idx)+1e-12)
        log_interval = 5
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item(), train_acc, data_sum,
                       100. * float(train_acc) / (float(data_sum))))

    if len(train_loader)==0:
        loss=0
        return loss, 0.0, 0,0,0
    else:
        return loss.item(), float(train_acc) / float(len(train_loader.sampler)), output_avg, activation_avg, tot_output_avg



def train_minimum(train_loader, epoch, model, optimizer):
    model.train()
    train_acc = 0.
    data_sum = 0





    for batch_idx, (data, target, index) in enumerate(train_loader):



        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output, _, _, _ = model(data)

        loss = F.cross_entropy(output, target).cuda()


        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        data_sum += data.size()[0]
        del output

        # print(float(batch_idx)+1e-12)
        log_interval = 5
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item(), train_acc, data_sum,
                       100. * float(train_acc) / (float(data_sum))))



    return loss.item(), float(train_acc) / float(len(train_loader.sampler))




def test(model, test_loader):
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    correct = 0
    with torch.no_grad():
        for data, target, index in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output, activation, _, _ = model(data)
            loss = F.cross_entropy(output, target)
            test_loss.add(loss.item())  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            del output



        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss.item(), correct, len(test_loader.sampler),
            100. * float(correct) / len(test_loader.sampler)))
    return loss.item(), float(correct) / float(len(test_loader.sampler))



def logits_extraction(model,train_loader, layer, n_classes, average, output_base,act_base,
                      multiloss, scale, incremental_step):




    model.eval()

    if layer=='last':

        output = np.zeros((len(train_loader.dataset.indices), n_classes))



    elif layer=='prelast':

        output = np.zeros((len(train_loader.dataset.indices),64,8,8))

    if incremental_step>0:

        act = np.zeros((len(train_loader.dataset.indices), 64, 8, 8))


    inds = np.zeros((len(train_loader.dataset.indices),), dtype=int)



    k=0
    soft_label=[]

    if multiloss:
        soft_label_l1=[]
        soft_label_l2=[]
        soft_label_l3=[]

    grad_avg=[]

    for i in range(n_classes):
        soft_label.append([])

        if multiloss:
            soft_label_l1.append([])
            soft_label_l2.append([])
            soft_label_l3.append([])

        grad_avg.append([])






    model.eval()

    with torch.set_grad_enabled(False):
        for batch_idx, (datat, target,index) in enumerate(train_loader):



            datat, target = datat.cuda(), target.cuda()


            datat, target= Variable(datat), Variable(target)


            out, activation, out_l1, out_l2=model(datat)


            if multiloss :

               out_l1_s = multi_scale_output_calculation(out_l1, scale)

               out_l2_s = multi_scale_output_calculation(out_l2, scale)

               out_l3_s = multi_scale_output_calculation(activation, scale)

               if multiloss and batch_idx==0:
                   output_l1 = np.zeros((len(train_loader.dataset.indices), 16, out_l1_s.size()[2]), dtype=np.float32)
                   output_l2 = np.zeros((len(train_loader.dataset.indices), 32, out_l2_s.size()[2]), dtype=np.float32)
                   output_l3 = np.zeros((len(train_loader.dataset.indices), 64, out_l3_s.size()[2]), dtype=np.float32)




            if layer=='last':



                if average:


                    for i in range(out.size()[0]):
                        soft_label[target[i]].append(out[i,:].data.cpu().numpy())

                        if multiloss:
                            soft_label_l1[target[i]].append(out_l1_s[i, :,:].data.cpu().numpy())
                            soft_label_l2[target[i]].append(out_l2_s[i, :, :].data.cpu().numpy())
                            soft_label_l3[target[i]].append(out_l3_s[i, :,:].data.cpu().numpy())

                else:



                    output[k:k + datat.shape[0], :] = out.data.cpu().numpy()

                    if multiloss:
                        output_l1[k:k + datat.shape[0], :] = out_l1_s.data.cpu().numpy()
                        output_l2[k:k + datat.shape[0], :] = out_l2_s.data.cpu().numpy()
                        output_l3[k:k + datat.shape[0], :] = out_l3_s.data.cpu().numpy()

                    inds[k:k + datat.shape[0]] = index.cpu().numpy()




            elif layer=='prelast' :

                if average:



                    for i in range(out.size()[0]):

                        soft_label[target[i]].append(activation[i, :, :, :].data.cpu().numpy())


                else:


                    output[k:k + datat.shape[0], :, :, :] = activation.data.cpu().numpy()
                    inds[k:k + datat.shape[0]] = index.cpu().numpy()
                    k = k + datat.shape[0]


            if incremental_step>0:

                act[k:k + datat.shape[0], :, :, :] = activation.data.cpu().numpy()

            k = k + datat.shape[0]




    if average:




        if layer == 'last':
            soft_labels_class_averaged = np.zeros((100,100))
            soft_labels_l1_class_averaged = np.zeros((100, 100))
            soft_labels_l2_class_averaged = np.zeros((100, 100))
            soft_labels_l3_class_averaged = np.zeros((100, 100))


            for i in range(n_classes):
                soft_label_ar = np.array(soft_label[i])
                soft_labels_class_averaged[i,:]=np.mean(soft_label_ar,axis=0)

                if multiloss:

                    soft_label_ar = np.array(soft_label_l1[i])
                    soft_labels_l1_class_averaged[i, :] = np.mean(soft_label_ar, axis=0)

                    soft_label_ar = np.array(soft_label_l2[i])
                    soft_labels_l2_class_averaged[i, :] = np.mean(soft_label_ar, axis=0)

                    soft_label_ar = np.array(soft_label_l3[i])
                    soft_labels_l3_class_averaged[i, :] = np.mean(soft_label_ar, axis=0)



        elif layer == 'prelast':
            soft_labels_class_averaged=np.zeros((100,64,8,8))

            for i in range(n_classes):
                soft_label_ar = np.array(soft_label[i])
                soft_labels_class_averaged[i]=np.mean(soft_label_ar,axis=0)


        output_base = soft_labels_class_averaged

        if multiloss:
            output_base_layers=[soft_labels_l1_class_averaged, soft_labels_l2_class_averaged, soft_labels_l3_class_averaged, inds]
        else:
            output_base_layers=0


    else:



        for i in range(len(inds)):

            if layer == 'last':
                output_base[inds[i], :] = output[i, :]


                if multiloss:

                    output_base_layers=[output_l1,output_l2, output_l3,inds]


                else:

                    output_base_layers=0





            elif layer == 'prelast':
                output_base[inds[i], :, :, :] = output[i, :, :, :]


            if incremental_step>0:

                act_base[inds[i], :, :, :] = act[i, :, :, :]












    return output_base, output_base_layers, act_base



def multi_scale_output_calculation(out, scale):


    out_f= []



    for s in range(scale):
        a = torch.split(out, out.size()[2] // (s+1), dim=2)
        b = [torch.split(x, out.size()[3] // (s+1), dim=3) for x in a]
        c = [y for x in b for y in x]

        out1= sum(arr.size()[2] for arr in c)
        out2 = sum(arr.size()[3] for arr in c)

        out_s = torch.zeros(out.size()[0], out.size()[1], out1 + out2)
        previous_index=0

        for i in range(len(c)):

            step_w = c[i].size()[2]
            step_h = c[i].size()[3]

            out_s1 = 0

            for j in range(c[i].size()[2]):
                out_s1 += c[i][:, :, j, :]/c[i].size()[2]

            out_s2 = 0

            for j in range(c[i].size()[3]):
                out_s2 += c[i][:, :, :, j]/c[i].size()[3]

            out_s3 = torch.cat((out_s1, out_s2), dim=2)


            out_s[:,:,previous_index:previous_index+(step_w+step_h)]=out_s3
            previous_index=previous_index+step_w+step_h
            out_f.append(out_s)


    out_final=torch.cat(out_f, dim=2)






    return out_final



def external_distillation(mod, normalization, depth, layer, n_classes, average, n_clients, train_loader_list, multiloss, scale ):



    if os.path.isfile(external_distillation):

        if mod == 'vgg':
            model_init = vgg(normalization, n_classes, depth=depth)



        elif mod == 'resnet':
            model_init = resnet(n_classes=n_classes, depth=depth)

        model_init.cuda()

        checkpoint1 = torch.load(external_distillation)
        # best_prec1 = checkpoint1['best_prec1']
        model_init.load_state_dict(checkpoint1['state_dict'])
        print("=> loaded checkpoint '{}'  "
              .format(external_distillation))

    else:
        print("=> no checkpoint found at '{}'".format(external_distillation))

    if layer == 'last':
        output_base = np.zeros((50000, n_classes))

    elif layer == 'prelast':
        output_base = np.zeros((50000, 64, 8, 8))

    grads_base = 0
    target_base = 0
    incremental_step=0
    act_base=0

    if average:
        if layer == 'last':
            per_client_logits = np.zeros((n_clients, n_classes, 100))
        elif layer == 'prelast':
            per_client_logits = np.zeros((n_clients, n_classes, 64, 8, 8))

    for c in range(n_clients):

        output_base, _, _ = logits_extraction(model_init, train_loader_list[c], layer, n_classes, average,
                                output_base, act_base, multiloss, scale, incremental_step)
        if average:
            per_client_logits[c, :, :] = output_base

    if average:
        extracted_logits = np.mean(per_client_logits, axis=0)
    else:
        extracted_logits = output_base



    return extracted_logits
