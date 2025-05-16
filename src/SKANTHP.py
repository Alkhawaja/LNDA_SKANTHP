# Standard Libraries

from tqdm import tqdm

# Data Science and Machine Learning Libraries
import numpy as np
# PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Custom Modules
import src.Constants as Constants
import Utils
from src.Dataset import get_dataloader
from src.Models import Transformer

def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()
    Final_lambdas=[]
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for bi, batch in enumerate(training_data):
        """ prepare data """
        event_time, _, event_type, vertex = map(lambda x: x.to(opt), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, vertex, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll, all_lambda = Utils.log_likelihood(model, enc_out, event_time, event_type,bi)
        event_loss = -torch.sum(event_ll - non_event_ll)
        Final_lambdas.append(all_lambda)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, Final_lambdas

def train(model, training_data, optimizer, scheduler, pred_loss_func, device,epoch):
    """ Start training. """

    # valid_event_losses = []  # validation log-likelihood
    # valid_pred_losses = []  # validation event type prediction accuracy
    # valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(epoch):
        epoch = epoch_i + 1
        train_event, train_type, train_time,Final_lambdas = train_epoch(model, training_data, optimizer, pred_loss_func, device)

        scheduler.step()
        torch.set_default_device('cuda')

    return Final_lambdas


def SKANTHP(mue,A_emb,W_emb,num_vertices,num_types,Sequences,model_type=None,device='cuda',epoch=50):
    if model_type==None:
        pass
    
    else:
        batch_size=1
        n_head = 4
        if model_type=='mlp':
            n_layers = 4
            d_model = 512
            d_rnn = 64
            d_inner = 1024
            d_k = 512
            d_v = 512
            dropout = 0.1
            lr = 1e-4
            smooth = 0.1
            epoch = 10

        else:
            n_layers = 1
            torch.set_default_device('cpu')
            d_model = 4#16
            d_rnn =2# 4
            d_inner = 8#32
            d_k = 4#16
            d_v = 4#16
            dropout = 0.1
            lr = 1e-2
            smooth = 0.1
            epoch = 10
  

        """ prepare model """
        model = Transformer(
            num_vertices=num_vertices,
            num_types=num_types,
            d_model=d_model,
            d_rnn=d_rnn,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            A=A_emb,
            W=W_emb,
            model_type=model_type,
            use_kan_bias=False,
            beta=mue
        )
        model.to(device)
        trainloader = get_dataloader(Sequences, batch_size, shuffle=False)

        """ optimizer and scheduler """
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                lr, betas=(0.9, 0.999), eps=1e-05)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

        """ prediction loss function, either cross entropy or label smoothing """
        if smooth > 0:
            pred_loss_func = Utils.LabelSmoothingLoss(smooth, num_types, ignore_index=-1)
        else:
            pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        """ train the model """
        return train(model, trainloader, optimizer, scheduler, pred_loss_func, device,epoch)

