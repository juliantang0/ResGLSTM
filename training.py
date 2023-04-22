import argparse
import pandas as pd
import torch.nn as nn

from models.gin import ResGINNet
from utils import *


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, hidden, cell):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()  # 训练模式 - GCNNet.dropout = True
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, hidden, cell)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader, hidden, cell):
    model.eval()  # 评估模式 - GCNNet.dropout = False
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data, hidden, cell)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


loss_fn = nn.MSELoss()
LOG_INTERVAL = 20


def main(args):
    dataset = args.dataset
    modeling = [ResGINNet]
    model_index = args.model
    model_st = modeling[model_index].__name__

    cuda_name = "cuda:0"
    print('cuda_name:', cuda_name)

    TRAIN_BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.batch_size
    LR = args.lr

    NUM_EPOCHS = args.epoch

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    # Main program: iterate over different datasets
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling[model_index](k1=1, k2=2, k3=3, embed_dim=128, num_layer=1, device=device).to(device)

        # load pretrained state dict of model
        if args.save_file:
            model_file_name = args.save_file + ".model"
            if os.path.isfile(model_file_name):
                print("Pretrained model found: " + model_file_name)
                param_dict = torch.load(model_file_name)
                model.load_state_dict(param_dict)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        # model_file_name = 'model' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result' + model_st + '_' + dataset + '.csv'

        # load the result of pretrained model
        if args.save_file and os.path.isfile(result_file_name):
            result = pd.read_csv(result_file_name).loc[0, :]
            best_mse = result[1]
            best_ci = result[-2]
            print('rmse of previous model: best_mse, best_ci:', best_mse, best_ci, model_st, dataset)

        for epoch in range(NUM_EPOCHS):
            hidden, cell = model.init_hidden(batch_size=TRAIN_BATCH_SIZE)
            train(model, device, train_loader, optimizer, epoch + 1, hidden, cell)
            G, P = predicting(model, device, test_loader, hidden, cell)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P),
                   get_rm2(G.reshape(G.shape[0], -1), P.reshape(P.shape[0], -1))]
            if ret[1] < best_mse:
                if args.save_file:
                    model_file_name = args.save_file + '.model'
                    torch.save(model.state_dict(), model_file_name)

                with open(result_file_name, 'w') as f:
                    f.write('rmse,mse,pearson,spearman,ci,rm2\n')
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_ci = ret[-2]
                print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st,
                      dataset)
            else:
                print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,
                      model_st, dataset)

            # send to database
            save2db(best_mse, best_ci, ret[1], ret[-2], dataset, model_st)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepGLSTM")

    parser.add_argument("--dataset", type=str, default='davis',
                        help="Dataset Name (davis,kiba,DTC,Metz,ToxCast,Stitch)")

    parser.add_argument("--epoch",
                        type=int,
                        default=1000,
                        help="Number of training epochs. Default is 1000."
                        )

    parser.add_argument("--lr",
                        type=float,
                        default=0.0005,
                        help="learning rate",
                        )

    parser.add_argument("--batch_size", type=int,
                        default=128,
                        help="Number of drug-target per batch. Default is 128 for davis while 512 for others.")  # batch 128 for Davis

    parser.add_argument("--save_file", type=str,
                        default=None,
                        help="Where to save the trained model. For example davis.model")

    parser.add_argument("--model", type=int,
                        default=0,
                        help="Which model to train. 0 - ResGINNet. Default is 0.")

    args = parser.parse_args()
    print(args)
    main(args)
