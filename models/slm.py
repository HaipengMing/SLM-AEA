import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from apex import amp


def slm_one_epoch(epoch, data_loader, net, optimizer, scheduler, dataset, hyperparameter):
    running_loss = 0.
    correct = 0.
    iter_cnt = 0
    total = 0.
    num_steps = len(data_loader)
    alpha, beta, lam = hyperparameter
    net.train()
    for i, data in enumerate(tqdm(data_loader, ncols=80)):
        inputs, labels, labels_mid, idx = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels_mid = labels_mid.cuda()
        labels_d = torch.softmax(labels_mid, dim=1)
        iter_cnt += 1
        total += len(labels)
        outputs = net(inputs)
        optimizer.zero_grad()
        loss1 = cross_entropy(outputs, labels_d)

        with amp.scale_loss(loss1, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        num_iter = num_steps * epoch + i
        optimizer.param_groups[0]['lr'] = scheduler(num_iter)

        _, predicted = torch.max(outputs, 1)
        correct += torch.eq(predicted, labels.long()).sum().item()
        running_loss += loss1.item()

        labels_mid.requires_grad = True
        if labels_mid.grad is not None:
            labels_mid.grad.zero_()
        outputs_dt = torch.softmax(outputs, dim=1).detach()
        loss_o = nn.CrossEntropyLoss()(labels_mid, labels.long())
        loss_m = cross_entropy(labels_mid, outputs_dt)
        loss2 = ramp_u(num_iter, num_steps, alpha, beta) * loss_m + ramp_d(num_iter, num_steps, alpha, beta) * loss_o
        loss2.backward()
        labels_mid_grad = labels_mid.grad.detach()
        labels_mid_dt = labels_mid.detach()
        labels_mid_dt -= lam * labels_mid_grad
        dataset.labels_mid[idx] = labels_mid_dt.data.cpu()

    acc = correct / total
    running_loss /= iter_cnt
    return acc, running_loss


def valid_one_epoch(data_loader, net):
    with torch.no_grad():
        correct = 0.
        iter_cnt = 0
        running_loss = 0.
        total = 0.
        net.eval()
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            iter_cnt += 1
            total += len(labels)
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels.long())
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += torch.eq(predicted, labels).sum().item()
    acc = correct / total
    running_loss /= iter_cnt
    return acc, running_loss


def ramp_u(e, ns, alpha, beta):
    if e <= beta * ns:
        return np.exp(-(alpha - (e/ns)/beta) ** 2)
    else:
        return 1


def ramp_d(e, ns, alpha, beta):
    if e <= beta * ns:
        return 1
    else:
        return np.exp(-(alpha - beta/(e/ns)) ** 2)


def cross_entropy(predict_label, true_label):
    return torch.mean(torch.sum(-true_label * torch.log_softmax(predict_label, 1), dim=1))


def baseline_one_epoch(epoch, data_loader, net, optimizer, scheduler):
    running_loss = 0.
    correct = 0.
    iter_cnt = 0
    total = 0.
    num_steps = len(data_loader)
    net.train()
    for i, data in enumerate(tqdm(data_loader, ncols=80)):
        inputs, labels, _, _ = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        iter_cnt += 1
        total += len(labels)
        outputs = net(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels.long())
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.param_groups[0]['lr'] = scheduler(num_steps * epoch + i)
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += torch.eq(predicted, labels.long()).sum().item()
    acc = correct / total
    running_loss /= iter_cnt
    return acc, running_loss


