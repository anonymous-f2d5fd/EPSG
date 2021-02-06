from model import *
from method import *
from util import *
import os



def train_dl(type, device, path, save_path, train_pr, val_pr, random_seed, max_epochs=200, early_stop_epochs=5, batch_size=2000, lr=0.001):
    dataset,train_data,val_data,test_data = load_and_split_data(path,device,train_pr,val_pr,random_seed)
    if type== 'EPSGF':
        model = EPSGF(dataset['num_node_dic'], dataset['num_feature_dic'], 128, 64,64,'cuda')
    elif type == 'EPSGF_MLP':
        model = EPSGF_MLP(dataset['num_node_dic'], dataset['num_feature_dic'], 128, 64, 64, 'cuda')
    elif type == 'EPSGF_RAND':
        model = EPSGF_RAND(dataset['num_node_dic'], dataset['num_feature_dic'], 128, 64, 64, 'cuda')
    elif type == 'EPSGF_SOFTMAX':
        model = EPSGF_SOFTMAX(dataset['num_node_dic'], dataset['num_feature_dic'], 128, 64, 64, 'cuda')
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = SequenceBatchNLLLoss()
    val_accs = [99999999]
    early_stop_remain = early_stop_epochs
    train_losses = []
    for epoch in range(max_epochs):
        train_loss= train_model(model,opt,loss_func,dataset,train_data,batch_size,device)
        train_losses.append(train_loss)
        current_val_acc = eval_model(model,loss_func,dataset,val_data,batch_size,device)
        if current_val_acc < min(val_accs):
            early_stop_remain = early_stop_epochs
            val_accs.append(current_val_acc)
            state = {'net':model.state_dict()}
            torch.save(state, os.path.join(save_path, 'paras.pt'))
        early_stop_remain -= 1
        if early_stop_remain <= 0:
            break
        print('epoch:{}, val_loss:{}'.format(epoch,current_val_acc))
    checkpoint = torch.load(os.path.join(save_path, 'paras.pt'))
    model.load_state_dict(checkpoint['net'])
    if type == 'EPSPF':
        test_acc = test_model(model,dataset,test_data,batch_size,device,save_path=save_path,save_data=True)
    else:
        test_acc = test_model(model, dataset, test_data, batch_size, device)
    return test_acc


def sovle_EPSG(model_type, device, path, save_path, train_pr, val_pr, random_seed, eval_metrics):
    assert model_type in ('random', 'RIA','RAA','EPSGF','EPSGF_MLP','EPSGF_RAND','EPSGF_SOFTMAX')
    if model_type in ['RAA', 'RIA','random']:
        dataset, train_data, val_data, test_data = load_and_split_data(path, 'cpu', train_pr, val_pr, random_seed)
        num_user= dataset['num_node_dic']['user']
        train_seqs = [item.numpy() for item in train_data.values()]
        test_seqs = [item.numpy() for item in test_data.values()]
        if model_type == 'random':
            res = random_solve(test_seqs,eval_metrics)
        elif model_type == 'RAA':
            res = rank_on_activeness(num_user,train_seqs,test_seqs,eval_metrics)
        elif model_type == 'RIA':
            res = rank_on_influence(num_user,train_seqs,test_seqs,eval_metrics)
    else:
        res = train_dl(model_type, device, path, save_path, train_pr, val_pr, random_seed)

    return res