import argparse

from torch_geometric.loader import DataLoader

from models.resgin import ResGINNet
from utils import *
from training import predicting


def main(args):
    dataset = args.dataset
    modelings = [ResGINNet]
    cuda_name = "cuda:0"
    print('cuda_name:', cuda_name)

    TEST_BATCH_SIZE = args.batch_size

    result = []

    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if not os.path.isfile(processed_data_file_test):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)
        for modeling in modelings:
            model_st = modeling.__name__
            print('\npredicting for ', dataset, ' using ', model_st)
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling(embed_dim=128, num_layer=1, device=device).to(device)
            model_file_name = args.load_model
            if os.path.isfile(model_file_name):
                param_dict = torch.load(model_file_name)
                model.load_state_dict(param_dict)
                hidden, cell = model.init_hidden(batch_size=TEST_BATCH_SIZE)
                G, P = predicting(model, device, test_loader, hidden, cell)
                plot(G, P)
                ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P),
                       get_rm2(G.reshape(G.shape[0], -1), P.reshape(P.shape[0], -1))]
                ret = [dataset, model_st] + [round(e, 3) for e in ret]
                result += [ret]
            else:
                print('model is not available!')
    file_name = "Prediction_result_" + dataset + ".csv"
    with open(file_name, 'w') as f:
        f.write('dataset,model,rmse,mse,pearson,spearman,ci,rm2\n')
        for ret in result:
            f.write(','.join(map(str, ret)) + '\n')

    print("Prediction Done and result is Saved in the file!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ResGLSTM with pretrained models")

    parser.add_argument(
        "--dataset", type=str,
        default='davis', help='Dataset Name (davis, kiba, metz)'
    )

    parser.add_argument(
        "--batch_size", type=int,
        default=128, help='Test Batch size. For davis and metz is 128 while 512 for kiba.'
    )

    parser.add_argument(
        "--load_model", type=str,
        default="Pretrained_model/davis.model", help="Load a pretrained model"
    )

    args = parser.parse_args()
    print(args)
    main(args)
