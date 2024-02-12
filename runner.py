import copy
import numpy as np
import torch
import wandb

from torch.optim import SGD

from utils import accuracy_and_loss

N_train = 50000


def run_proxyprox(net, device, trainloader, testloader, subsetloader, n_epoch=5, optimizer=None,
                  optimizer_out=None, checkpoint=150, noisy_train_stat=True, scheduler=None,
                  run_name=None, use_wandb=True, clip=None, n_epoch_in=1, wandb_project=''):
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    net.train() # Enable things such as dropout
    criterion = torch.nn.CrossEntropyLoss()
   
    prev_net = copy.deepcopy(net)
    prev_net.to(device)
    prev_net.train()
    prev_optimizer = SGD(prev_net.parameters(), lr=1.)
       
    running_loss = 0.0
    i_checkpoint = 0
    if use_wandb:
        run = wandb.init(project=wandb_project, name=run_name, reinit=True)
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
           
            prev_net.load_state_dict(net.state_dict())

            running_loss += loss.item()
            i_checkpoint += 1
            if noisy_train_stat and (i % 20) == 0:
                losses.append(loss.cpu().item())
                it_train.append(epoch + i * trainloader.batch_size / N_train)

            if i % checkpoint == 0 or i == 0:
                ave_loss = running_loss / i_checkpoint
                if ave_loss < 0.01:
                    print(f'[{epoch + 1}, {i + 1:5}] loss: {ave_loss:.4f}')
                else:
                    print(f'[{epoch + 1}, {i + 1:5}] loss: {ave_loss:.3f}')
                ep = epoch + i * trainloader.batch_size / N_train
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                net.train()
                it_test.append(ep)
                running_loss = 0.0
                i_checkpoint = 0
                if use_wandb:
                    wandb.log({'lr': optimizer.param_groups[0]['lr']}, commit=False)
                    wandb.log({'Test accuracy': test_a * 100, 'Epoch': ep}, commit=False)
                    wandb.log({'Running loss': ave_loss, 'Epoch': ep}, commit=False)
                    wandb.log({'Test loss': test_l, 'Epoch': ep}, commit=False)
           
            prev_optimizer.zero_grad()
            inner_loss = 0
            for j, data_subset in enumerate(subsetloader, 0):
                inputs, labels = data_subset
                inputs, labels = inputs.to(device), labels.to(device)
               
                prev_outputs = prev_net(inputs)
                prev_loss = criterion(prev_outputs, labels) / len(subsetloader.dataset)
                prev_loss.backward()
                inner_loss += prev_loss.item()
            if use_wandb:
                wandb.log({'Inner loss': inner_loss, 'Epoch': ep}, commit=False)
           
            for epoch_in in range(n_epoch_in):
                for j, data_subset in enumerate(subsetloader, 0):
                    inputs, labels = data_subset
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    optimizer.inner_step(prev_optimizer=prev_optimizer)
            inner_loss = 0
            net.eval()
            inner_grad_norm = 0
            with torch.no_grad():
                for j, data_subset in enumerate(subsetloader, 0):
                    inputs, labels = data_subset
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    inner_loss += criterion(outputs, labels) / len(subsetloader.dataset)
               
                for group, prev_group in zip(optimizer.param_groups, prev_optimizer.param_groups):
                    for p, prev_p in zip(group['params'], prev_group['params']):
                        if p.grad is None:
                            continue
                        state = optimizer.state[p]
                        prev_d_p = prev_p.grad.data
                        inner_grad_norm += torch.sum(prev_d_p**2)
                        inner_loss += torch.sum((p.data - prev_p.data) * (state['out_grad'] - prev_d_p))
                        inner_loss += 1/2 * group['reg'] * torch.linalg.norm(p.data - prev_p.data)**2
            inner_grad_norm = np.sqrt(inner_grad_norm.item())
            wandb.log({'Inner grad norm': inner_grad_norm, 'Epoch': ep}, commit=False)
            wandb.log({'Inner loss 2': inner_loss, 'Epoch': ep})
            net.train()

        if not noisy_train_stat:
            it_train.append(epoch)
            train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
            print(f'Train accuracy, loss:   {train_a*100:.2f}, {train_l:.5f}')
            print(f'Test  accuracy, loss:   {test_a*100:.2f}, {test_l:.5f}')
            train_acc.append(train_a)
            losses.append(train_l)
            net.train()
            if use_wandb:
                wandb.log({'Train accuracy': train_a * 100, 'Train epoch': it_train[-1]}, commit=False)
                wandb.log({'Train loss': train_l, 'Train epoch': it_train[-1]})
       
        if scheduler is not None:
            scheduler.step()

    print('Finished Training')
    if use_wandb:
        run.finish()
    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test))


def run(net, device, trainloader, testloader, n_epoch=5, optimizer=None,
        checkpoint=150, noisy_train_stat=True, scheduler=None,
        run_name=None, use_wandb=True, clip=None, wandb_project=''):
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    net.train() # Enable things such as dropout
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    i_checkpoint = 0
    if use_wandb:
        run = wandb.init(project=wandb_project, name=run_name, reinit=True)
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            running_loss += loss.item()
            i_checkpoint += 1
            if noisy_train_stat and (i % 20) == 0:
                losses.append(loss.cpu().item())
                it_train.append(epoch + i * trainloader.batch_size / N_train)

            if i % checkpoint == 0 or i == 0:
                ave_loss = running_loss / i_checkpoint
                if ave_loss < 0.01:
                    print(f'[{epoch + 1}, {i + 1:5}] loss: {ave_loss:.4f}')
                else:
                    print(f'[{epoch + 1}, {i + 1:5}] loss: {ave_loss:.3f}')
                ep = epoch + i * trainloader.batch_size / N_train
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                net.train()
                it_test.append(ep)
                running_loss = 0.0
                i_checkpoint = 0
                if use_wandb:
                    wandb.log({'lr': optimizer.param_groups[0]['lr']}, commit=False)
                    wandb.log({'Test accuracy': test_a * 100, 'Epoch': ep}, commit=False)
                    wandb.log({'Running loss': ave_loss, 'Epoch': ep}, commit=False)
                    wandb.log({'Test loss': test_l, 'Epoch': ep}, commit=False)
           
            net.eval()

        if not noisy_train_stat:
            it_train.append(epoch)
            train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
            print(f'Train accuracy, loss:   {train_a*100:.2f}, {train_l:.5f}')
            print(f'Test  accuracy, loss:   {test_a*100:.2f}, {test_l:.5f}')
            train_acc.append(train_a)
            losses.append(train_l)
            net.train()
            if use_wandb:
                wandb.log({'Train accuracy': train_a * 100, 'Train epoch': it_train[-1]}, commit=False)
                wandb.log({'Train loss': train_l, 'Train epoch': it_train[-1]})
       
        if scheduler is not None:
            scheduler.step()

    print('Finished Training')
    if use_wandb:
        run.finish()
    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test))
