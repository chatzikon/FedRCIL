import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.nn import functional as F
from PIL import Image
import random
import torch.nn as nn
from utils import *

def incremental_dataloader_creation(loader_init, dataset_init, seen_ids, exemplar_classes, batch_size, transform, exemplar_size, model,
                                    exemplar_ids_prev, incremental_step):



    if exemplar_classes>0:

        class_size = exemplar_size // exemplar_classes

        classes_list= [class_size for _ in range(exemplar_classes)]

        if class_size * exemplar_classes < exemplar_size:

            extra_imgs=random.sample(range(0, exemplar_classes), k= exemplar_size-(class_size*exemplar_classes))

            for j in range(len(extra_imgs)):

                classes_list[j]+=1


        exemplar_ids=[]


        for i in range(incremental_step):




            incremental_step_images, incremental_step_images_ids = get_class_images(loader_init, seen_ids[0]- incremental_step+i)


            if incremental_step_images.shape[0]>0:
                class_exemplar_ids = construct_exemplar_set(incremental_step_images, incremental_step_images_ids, classes_list[i], transform, model, i)
                exemplar_ids.extend(class_exemplar_ids)







    loader_ids = get_classes_images(loader_init, seen_ids)

    if exemplar_classes>0:

        loader_ids.extend(exemplar_ids+exemplar_ids_prev)

    else:

        exemplar_ids = exemplar_ids_prev

    dataloader_sampler = SubsetRandomSampler(loader_ids,
                                                      generator=torch.Generator().manual_seed(0))

    loader_incremental = torch.utils.data.DataLoader(dataset_init, batch_size=batch_size,
                                                      sampler=dataloader_sampler)


    return loader_incremental, exemplar_ids




def incremental_dataloader_creation_alt(loader_init, dataset_init, seen_ids, exemplar_classes, batch_size, transform, exemplar_size, model,
                                    exemplar_ids_prev, incremental_step):








    return loader_incremental, exemplar_ids



def incremental_common_dataloader_creation(loader_init,  seen_ids, exemplar_classes, transform, exemplar_size, model, incremental_step):



    class_size = exemplar_size // exemplar_classes

    classes_list= [class_size for _ in range(exemplar_classes)]

    if class_size * exemplar_classes < exemplar_size:

        extra_imgs=random.sample(range(0, exemplar_classes), k= exemplar_size-(class_size*exemplar_classes))

        for i in range(len(extra_imgs)):

            classes_list[i]+=1




    feature_extractor_output = []

    for i in range(incremental_step):




        incremental_step_images, _ = get_class_images(loader_init, seen_ids[0]-incremental_step+i)

        _, class_feature_extractor_output = compute_class_mean(incremental_step_images, transform, model, i)



        feature_extractor_output.append(class_feature_extractor_output)

    #feature_extractor_output_np = np.concatenate(feature_extractor_output)



    return  feature_extractor_output




def get_classes_images(loader, seen_ids):

    loader_ids = []

    if hasattr(loader.dataset, 'indices'):

        for ind in range(len(loader.dataset.indices)):
            if loader.dataset.dataset.targets[loader.dataset.indices[ind]] >= seen_ids[0] and loader.dataset.dataset.targets[loader.dataset.indices[ind]] < seen_ids[1]:
                loader_ids.append(ind)


    else:

        for ind in range(len(loader.dataset.targets)):
            if loader.dataset.targets[ind] >= seen_ids[0] and loader.dataset.targets[ind] < seen_ids[1]:
                loader_ids.append(ind)


    return loader_ids


def get_class_images(loader, seen_id):

    incremental_step_images_ids = []
    for ind in range(len(loader.dataset.indices)):
        if loader.dataset.dataset.targets[loader.dataset.indices[ind]] == seen_id:
            incremental_step_images_ids.append(ind)


    incremental_step_images=loader.dataset.dataset.data[incremental_step_images_ids]

    return incremental_step_images, incremental_step_images_ids

def construct_exemplar_set(images, ids, m, transform, model, c):

    class_mean, feature_extractor_output = compute_class_mean(images, transform, model, c)
    exemplar_ids = []
    now_class_mean = np.zeros_like(class_mean)

    for i in range(m):
        x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
        x = np.linalg.norm(x, axis=(2,3))
        x  = np.linalg.norm(x, axis=1)
        index = np.argmin(x)
        now_class_mean += feature_extractor_output[index]
        exemplar_ids.append(ids[index])


    return exemplar_ids

def compute_class_mean(images, transform, model, c):


    if type(model) == list:

        class_mean=np.mean(model[c], axis=0)
        feature_extractor_output= model[c]

    else:
        x = Image_transform(images, transform).cuda()
        _, activation, _, _ = model(x)
        feature_extractor_output = F.normalize(activation.detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)

    return class_mean, feature_extractor_output



def Image_transform(images, transform):
    data = transform(Image.fromarray(images[0])).unsqueeze(0)
    for index in range(1, len(images)):
        data = torch.cat((data, transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
    return data



def create_all_incremental_dataloaders(model_inc, feature_extractor, client,  seen_ids, test_transform,
                                       train_loader_list, valid_loader_list, exemplar_classes, incremental_step, train_set, batch_size,
                                       client_train_exemplar_size, exemplar_ids_tr, exemplar_ids_val, exemplar_ids_srv, valid_set, client_valid_exemplar_size, feature_extractor_np,
                                       ):









    train_loader_list[client], exemplar_ids_tr[client] = incremental_dataloader_creation(train_loader_list[client],
                                                                                         train_set[client],
                                                                                         seen_ids,
                                                                                         0,
                                                                                         batch_size, test_transform,
                                                                                         client_train_exemplar_size,
                                                                                         model_inc,
                                                                                         exemplar_ids_srv,
                                                                                         incremental_step)



    valid_loader_list[client], exemplar_ids_val[client] = incremental_dataloader_creation(valid_loader_list[client], valid_set[client],
                                                               [0, seen_ids[1]], 0,
                                                               batch_size, test_transform,
                                                               client_valid_exemplar_size, model_inc, 0,
                                                               incremental_step)














    return train_loader_list[client], exemplar_ids_tr[client], valid_loader_list[client], exemplar_ids_val[client], feature_extractor, feature_extractor_np



















