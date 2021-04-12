from dataset import stockDataset
from stock_feature import feature_add
from model import Net
from train import train_test
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd 
import csv
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = pd.read_csv(f'data/{args.training}', header= None, names= ['open', 'high', 'low', 'close'])

    seq_len=10
    drop_prob=0.5
    lr=1e-3
    num_epochs=15
    batch_size=16
    step_size=10
    hidden_dim = 64
    num_layers = 2

    data, [max_price, min_price] = feature_add(train_data)
    dataset = stockDataset(data, seq_len=seq_len, label_idx=0)

    in_features = len(data[0])

    model = Net(in_features=in_features, hidden_dim=hidden_dim, n_classes=1, num_layers=num_layers, drop_prob=drop_prob).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    criterion = nn.MSELoss(reduction='mean')

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = train_test(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                       train_dataloader=dataloader, num_epochs=num_epochs, device=device)

    # global test_input
    test_data = pd.read_csv(f'data/{args.testing}', header= None, names= ['open', 'high', 'low', 'close'])
    test_input = ((train_data.iloc[-seq_len:].to_numpy() -min_price)/(max_price-min_price)).tolist()
    # print(len(test_input))

    # initial position 
    model.eval()
    with torch.no_grad():
        pred= model(torch.tensor(test_input, dtype= torch.float).to(device).unsqueeze(dim= 0))

    if ((pred.item() - test_input[-5][0])/test_input[-5][0]) >= 0.014:
        hold_position = 0
    elif ((pred.item() - test_input[-5][0])/test_input[-5][0]) <= -0.0115:
        hold_position = 1 
    else:
        hold_position = 0


    preds = []
    # hold_position = 0

    def predict_action(row):
        
        global preds
        global hold_position
        global test_input

        new_input = (row.values - min_price)/(max_price-min_price)
        test_input.pop(0)
        test_input.append(new_input)
        
        tmp_test_input = torch.tensor(test_input, dtype= torch.float).to(device)
        
        model.eval()
        with torch.no_grad():
            pred= model(tmp_test_input.unsqueeze(dim= 0))
        
        if ((pred.item() - test_input[-5][0])/test_input[-5][0]) >= 0.014:
            if hold_position == 1:
                hold_position = 0
                return '-1' 
            else:
                return '0' 
            
        elif ((pred.item() - test_input[-5][0])/test_input[-5][0]) <= -0.0115:
            if hold_position == 0:
                hold_position = 1
                return '1'
            else:
                return '0' 
        else: 
            return '0'

    with open(f'{args.output}', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, row in test_data.iterrows():
            if i < 19:
                writer.writerow([f'{predict_action(row)}'])
        print('output already done!')