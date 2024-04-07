import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

recreation_criteria = nn.MSELoss()

def train_recon(model, device, train_loader, optimizer, criteria):
    model.train()
    train_loss = 0
    rec_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, recreation = model(data, mask=True)
        loss1 = criteria(output.view(-1), target.view(-1))
        #loss2 = recreation_criteria(recreation, data)
        loss = recreation_criteria(recreation, data) #[:, :, :116] +data[:, :, 116:])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #rec_loss += loss2.item()

    return train_loss / len(train_loader)


def test_recon(model, device, test_loader, criteria):
    model.eval()
    test_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, recreation = model(data)
            #test_loss += criteria(output.view(-1), target.view(-1)).item()
            test_loss += recreation_criteria(recreation, data) #[:, :, :116] +data[:, :, 116:])
            y_pred.extend(output.view(-1).cpu().numpy())
            y_true.extend(target.view(-1).cpu().numpy())
    correlation = np.corrcoef(y_pred, y_true)[0, 1]
    return test_loss / len(test_loader), correlation

# PREDICTION!
def train_predict(model, device, train_loader, optimizer, criteria):
    model.train()
    train_loss = 0
    rec_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, recreation = model(data, mask=True)
        loss = criteria(output.view(-1), target.view(-1))  #recreation_criteria(recreation, data) #loss1 + loss2
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # for _ in range(5):
        #     data_mixup, target_mixup = mixup(data, target)
        #     optimizer.zero_grad()
        #     output, _ = model(data_mixup, mask=True)
        #     loss = criteria(output.view(-1), target_mixup.view(-1))
        #     loss.backward()
        #     optimizer.step()
        #     train_loss += loss.item()

    #print(f"Batch: {batch_idx}, Train Loss: {train_loss}, Rec Loss: {rec_loss/len(train_loader)}")
    return train_loss / len(train_loader)


def test_predict(model, device, test_loader, criteria):
    model.eval()
    test_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, recreation = model(data)
            test_loss += criteria(output.view(-1), target.view(-1)).item()
            y_pred.extend(output.view(-1).cpu().numpy())
            y_true.extend(target.view(-1).cpu().numpy())
    avg_loss = test_loss / len(test_loader.dataset)
    # rmse = mean_squared_error(y_true, y_pred, squared=False)
    # mae = mean_absolute_error(y_true, y_pred)
    # r2 = r2_score(y_true, y_pred)
    correlation = np.corrcoef(y_pred, y_true)[0, 1]
    return avg_loss, correlation

def test_predict_and_save(model, device, test_loader, criteria, y_scaler):
    model.eval()
    test_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, recreation = model(data)
            test_loss += criteria(output.view(-1), target.view(-1)).item()
            y_pred.extend(output.view(-1).cpu().numpy())
            y_true.extend(target.view(-1).cpu().numpy())

    y_pred_rescaled = y_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    y_true_rescaled = y_scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()

    
    avg_loss = test_loss / len(test_loader.dataset)
    rmse = mean_squared_error(y_true_rescaled, y_pred_rescaled, squared=False)
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    r2 = r2_score(y_true_rescaled, y_pred_rescaled)
    correlation = np.corrcoef(y_pred_rescaled, y_true_rescaled)[0, 1]
    return avg_loss, correlation, rmse, mae, r2, y_true_rescaled, y_pred_rescaled

