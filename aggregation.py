import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn



def load_model_federated_traditional(model, list_name, n_clients,path):
    np_path = './SaveModel'+path+'/Model_'


    for var_name in list_name:
        temp_load_numpy = []

        print(var_name)

        for client in range(n_clients):
            temp_load_numpy.append(np.load(np_path + str(client) + "/%s.ndim.npy" % (var_name)))


        temp = np.asarray(temp_load_numpy)
        temp_load_numpy_server = np.mean(temp, 0)




        tensor_load = torch.tensor(temp_load_numpy_server)

        model.state_dict()[var_name].copy_(tensor_load)

    return model







def federated_distillation_aggregation(per_client_output_avg,per_client_activation_avg, per_client_tot_output_avg, n_classes, n_clients,layer):

    output_per_class=[]
    tot_output_per_class=[]
    activation_per_class=[]


    for j in range(n_classes):
        output_per_class.append([])
        tot_output_per_class.append([])
        activation_per_class.append([])






    for i in range(n_classes):
        for c in range(n_clients):

            for j in range(len(per_client_activation_avg[c][i])):

                output_per_class[i].append(per_client_output_avg[c][i][j])
                tot_output_per_class[i].append(per_client_tot_output_avg[c][i][j])
                activation_per_class[i].append(per_client_activation_avg[c][i][j])




    ###weighted average
    weighted_average=[]

    for i in range(n_classes):
        weighted_average.append([])

    for i in range(n_classes):



        activation_ts = torch.unsqueeze(activation_per_class[i][0], 0)
        output_ts = torch.unsqueeze(output_per_class[i][0], 0)
        tot_output_ts = torch.unsqueeze(tot_output_per_class[i][0], 0)



        for j in range(1,len(activation_per_class[i])):

            activation_ts=torch.cat((activation_ts, torch.unsqueeze(activation_per_class[i][j],0)),0)
            output_ts = torch.cat((output_ts, torch.unsqueeze(output_per_class[i][j], 0)), 0)
            tot_output_ts = torch.cat((tot_output_ts, torch.unsqueeze(tot_output_per_class[i][j], 0)), 0)



        activation_ts=torch.permute(activation_ts, (1,2,3,0))
        tot_output_ts=torch.permute(tot_output_ts, (1,0))






        if layer=='last':

            weighted_average[i]=torch.mean(tot_output_ts, dim=1)


        elif layer=='prelast':

            weighted_average[i]=torch.mean(activation_ts, dim=3)






    return weighted_average




