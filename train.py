from function import *
from util import *
import pandas as pd
import os


if __name__ == '__main__':
    BASE_PATH = os.path.split(os.path.abspath(__file__))[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH = 'data'
    MODELS = ('random', 'RAA','RIA','EPSGF')
    METRICS = ('rouge_l', 'rouge_s', 'rouge_2')
    DATASETS = ('meetup_ca', 'meetup_sg')
    TRAIN_TIMES = 1
    TRAIN_PR = 0.7
    VAL_PR = 0.1
    result = np.zeros((len(DATASETS), len(MODELS), len(METRICS), TRAIN_TIMES))
    for k,dataset in enumerate(DATASETS):
        for seed in range(TRAIN_TIMES):
            for i, model in enumerate(MODELS):
                print("training on {}, model: {},times: {}".format(dataset,model,seed))
                values = sovle_EPSG(model, device, os.path.join(BASE_PATH, SAVE_PATH, dataset, 'data.pt'), os.path.join(BASE_PATH, SAVE_PATH), TRAIN_PR, VAL_PR, seed, METRICS)
                for j, value in enumerate(values):
                    result[k,i,j,seed] = value
                print(values)
    result = result.mean(axis=-1)
    df = pd.DataFrame(np.zeros((len(MODELS), len(METRICS) * len(DATASETS))), index=MODELS,
                      columns=pd.MultiIndex.from_product([DATASETS, METRICS]))
    for i,dataset in enumerate(DATASETS):
        df[dataset] = result[i, :, :]
    print(df)
    df.to_csv(os.path.join(BASE_PATH, 'result.csv'))

