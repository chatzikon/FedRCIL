# FedRCIL

The source code of the paper "FedRCIL: Federated Knowledge Distillation for Representation-based Contrastive Incremental Learning" that will be presented in the ICCV workshop "The First Workshop on Visual Continual Learning"

**Installation**

Run pip install -r equirements.txt to install the required packages

**Proposed method**

![model architecture image](https://github.com/chatzikon/FedRCIL/blob/main/images/FL_IL_scheme_iccv_generic_figure.png)


**Train**

Use federated_train.py for training a new model. The default parameters are about an experiment with IID data, mu=0.5, buffer_size=4
The majority of the functions required for training and testing are at the script traintest.py. 

The arguments of the main function are explained below:

**path:** The path to evaluate a pretrained model. 

**wd_factor:** The weight decay factor for training

**normalization:** Type of initial normalization for model parameters.

**epochs:** number of the experiment total epochs

**clients:** number of the experiment clients

**t_round:** number of the local epoch of each client or each round

**lr:** the learning rate of the experiment

**dataset:** the dataset to use (right now it supports only cifar10 and cifar100)

**mod:** which model to use for training (right now only vgg and resnet models are supported)

**depth:** the depth of the model (i.e. 56 for resnet56)

**layer:** the layer to use during distillation (last and prelast layer are supported)

**mode:** "distillation" for federated distillation and traditional for the federated learning scheme

**common dataset size:** the size of the dataset that it is common to all clients (as percentage of the total training dataset)

**coef_t, coef_d:** weights of the losses during training or distillation

**external distillation:** the path to a pretrained teacher model employed for distillation

**average:** whether or not to average local anchors before transmit them to central node

**loss:** which loss to use for distillation (mse, crossentropy or kl loss)

**temperature:** denotes the weighting of the contrastive loss terms

**exemplar_percentage:** the size of the dataset that it is employed as exemplar set (as percentage of the total training dataset)

**incremental_step:** number of new classes each new task introduces

**incremental_rounds:** number of federated rounds of each incremental task

**model_buffer_size:** the number of models of previous incremental rounds employed to the contrastive loss

**mu:** weight factor for the contrastive loss

**beta:** the concentration parameter of the Dirichlet distriution. It controls the non-IIDnes of data (applicable only if partition=noniid)

**multiloss:** whether or not to use intermediate multi-scale knowledge distillation losses

**multiloss type:** the multiloss type as described at section 4.3.1 of the paper

**scale:** the scale of the multi-scale knowledge distillation loss



