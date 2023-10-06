'''
train程序
'''

from dataset import get_dataloader
import torch
import config
import numpy as np
import plotresults as pr
from attenmodel import Attenmodel


model = Attenmodel().to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

def train(model, epochs, patience = 10, ):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    for epoch in range(epochs):

        model.train()

        train_dataloader = get_dataloader(dataset1='train')
        for index, (_,input,flowbase,area, target) in enumerate(train_dataloader):
            input = input.to(config.device)
            flowbase = flowbase.to(config.device)
            area = area.to(config.device)
            target = target.to(config.device)
            optimizer.zero_grad()
            decoder_outputs= model(input, flowbase, area)
            loss = criterion(decoder_outputs, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_dataloader = get_dataloader(dataset1='val')
        for index, (_,input,flowbase,area, target) in enumerate(val_dataloader):
            input = input.to(config.device)
            flowbase = flowbase.to(config.device)
            area = area.to(config.device)
            output = model(input, flowbase, area).cpu()
            loss = criterion(output, target)
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.3f} ' +
                     f'valid_loss: {valid_loss:.3f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        if epoch == 0:
            min_loss = valid_loss
            es = 0
        if valid_loss < min_loss:
            min_loss = valid_loss
            es = 0
            torch.save(model.state_dict(), config.model_save_path)
            torch.save(optimizer.state_dict(), config.optimizer_save_path)
            print("saving, model")
        else:
            es += 1
            print("Counter {} of 50".format(es))

            if es > patience:
                print("Early stopping with min_loss: ", min_loss)
                break

    return model, avg_train_losses, avg_valid_losses

if __name__ == '__main__':
    model, train_loss, val_loss = train(model, 1000, config.patience)
    pr.plotloss(train_loss, val_loss)



