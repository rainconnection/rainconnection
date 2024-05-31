import numpy as np
import torch
from torch import nn

from sklearn.metrics import f1_score

import time

def get_loss(infered, label, loss_f, opt = None):
    loss_func = loss_f
        
    loss = loss_func(infered, label)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), len(label) 
    
def train(model, optimizer, train_loader, valid_loader, scheduler, device, epochs = 50):
    
    model.to(device)
    best_score = 0
    epoch = 0
    early_count = 0
    start = time.time()
    while(epoch < epochs and early_count < 10):
        epoch += 1
        model.train()
        losses = []
        nums = []
        for x in train_loader:
            x = x[0]
            x = x.float().to(device)

            _x = model(x)
            loss, num = get_loss(_x, x, nn.L1Loss(), optimizer)
            
            losses.append(loss)
            nums.append(num)

        train_loss = np.sum(np.multiply(losses, nums))/np.sum(nums)

        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        model.eval()
        pred = []
        true = []
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.float().to(device)
    
                _x = model(x)
                diff = cos(x, _x).to(device).tolist()
                batch_pred = np.where(np.array(diff)<0.95, 1,0).tolist()
                pred += batch_pred
                true += y.tolist()
    
        valid_score = f1_score(true, pred, average='macro')
        print('train loss : ', train_loss, ' & valid f1-score : ', valid_score, ' & epoch : ', epoch, ' & time', time.time()-start)
    
        if scheduler is not None:
            scheduler.step(valid_score)

        if best_score < valid_score:
            best_score = valid_score
            torch.save(model, '../credit_card_fds_dacon/encoder_model.pt')
            print('best model save at encoder_model.pt')
        else:
            early_count += 1

def prediction(model, test_loader, device):
    model.to(device)
    model.eval()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pred = []
    with torch.no_grad():
        for x in iter(test_loader):
            x = x[0]
            x = x.float().to(device)
            
            _x = model(x)
            
            diff = cos(x, _x).cpu().tolist()
            pred += diff
    return pred

def prediction2(model, test_loader, device):
    model.to(device)
    model.eval()
    pdist = nn.PairwiseDistance(p=2)
    pred = []
    with torch.no_grad():
        for x in iter(test_loader):
            x = x[0]
            x = x.float().to(device)
            
            _x = model(x)
            
            diff = pdist(x, _x).cpu().tolist()
            pred += diff#batch_pred
    return pred