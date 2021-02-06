import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class HeterogeneousAggregationLayer(MessagePassing):
    def __init__(self, user_dim, event_dim, user_out_dim, event_out_dim, num_user, num_event, device):
        super(HeterogeneousAggregationLayer, self).__init__(aggr='add', )
        self.user_linear = nn.Linear(user_dim, user_out_dim)
        self.event_linear = nn.Linear(event_dim, event_out_dim)
        self.num_user = num_user
        self.user_norm = nn.LayerNorm(user_out_dim)
        self.event_norm = nn.LayerNorm(user_out_dim)
        self.num_event = num_event
        self.device = device
        self.to(device)

    def forward(self, x, edge_index):
        x['event'] = self.event_linear(x['event'])
        x['user'] = self.user_linear(x['user'])

        x['user'], x['event'] = self.propagate(edge_index=edge_index, x=(x['event'], x['user']), size=(self.num_event, self.num_user)) + x[
            'user'], self.propagate(edge_index=edge_index[[1, 0]], x=(x['user'], x['event']), size=(self.num_user, self.num_event)) + x['event']
        x['event'], x['user'] = x['event'] / (degree(edge_index[0], num_nodes=self.num_event) + 1).unsqueeze(1),\
                                x['user'] / (degree(edge_index[1], num_nodes=self.num_user) + 1).unsqueeze(1)

        return x



class HeterogeneousAggregationLayers(nn.Module):
    def __init__(self, in_dim_dic, mid_dim_dic, out_dim_dic, num_user, num_event, device):
        super(HeterogeneousAggregationLayers, self).__init__()
        self.hal0 = HeterogeneousAggregationLayer(in_dim_dic['user'], in_dim_dic['event'], mid_dim_dic['user'],
                                                  mid_dim_dic['event'], num_user, num_event, device)
        self.hal1 = HeterogeneousAggregationLayer(mid_dim_dic['user'], mid_dim_dic['event'], out_dim_dic['user'],
                                                  out_dim_dic['event'], num_user, num_event, device)

    def forward(self, x, edge_index):

        x = self.hal0(x,edge_index)

        x = self.hal1(x, edge_index)
        return x

class SequenceMaskedSoftmax(nn.Module):
    def __init__(self,device):
        super(SequenceMaskedSoftmax, self).__init__()
        self.eps = 1e-45
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, inputs, seq):
        for i in range(seq.shape[1] -1):
            mask = torch.zeros((inputs.shape[0], inputs.shape[-1]), dtype=torch.float).to(self.device).scatter_(-1, seq[:, i:].sort()[0], 1)
            inputs[:, i, :] = inputs[:, i, :] + (mask + self.eps).log()
        inputs = self.softmax(inputs)
        return inputs


class SequencePredictionLayer(nn.Module):
    def __init__(self, num_user, num_event, input_size, hidden_size, num_layers,device,use_softmax=False):
        super(SequencePredictionLayer, self).__init__()
        self.use_softmax = use_softmax
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, num_user)
        if use_softmax:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = SequenceMaskedSoftmax(device)
        self.num_layers = num_layers
        self.start = nn.Parameter(torch.randn((1,input_size), dtype=torch.float))
        self.to(device)

    def forward(self, x, ids, sequence):
        hx = nn.functional.embedding(ids,x['event']).unsqueeze(0).repeat(self.num_layers,1,1)
        start = self.start.unsqueeze(0).repeat(1,sequence.shape[0],1)
        x = nn.functional.embedding(sequence, x['user']).transpose(0,1)
        x = torch.cat([start,x], dim=0)
        x, _ = self.gru(x,hx)
        x = self.linear(x).transpose(0,1)
        if self.use_softmax:
            x = self.softmax(x)
        else:
            x = self.softmax(x,sequence)


        return x

    def test(self, feature_dic, ids, sequence):

        hx = nn.functional.embedding(ids, feature_dic['event']).unsqueeze(0).repeat(self.num_layers, 1, 1)
        x = self.start.unsqueeze(0).repeat(1, sequence.shape[0], 1)
        x, hx = self.gru(x, hx)
        x = self.linear(x).squeeze(0)
        x = x.gather(dim=-1, index=sequence).sort(descending=True)[1]
        sequence = sequence.gather(dim=-1, index=x)
        for i in range(0, sequence.shape[1]-1):
            x = nn.functional.embedding(sequence[:,i:i+1], feature_dic['user']).transpose(0, 1)
            x, hx = self.gru(x, hx)
            x = self.linear(x).squeeze(0)
            x = x.gather(dim=-1, index=sequence[:,i+1:]).sort(descending=True)[1]
            sequence[:,i+1:] = sequence[:,i+1:].gather(dim=-1,index=x)
        return sequence


class EPSGF(nn.Module):
    def __init__(self, num_node_dic,num_feature_dic, mid_size, out_size,input_size,device):

        super(EPSGF, self).__init__()

        self.in_dim_dic = num_feature_dic
        self.mid_dim_dic = {'event': mid_size, 'user': mid_size}
        self.out_dim_dic = {'event': out_size, 'user': out_size}

        self.hals = HeterogeneousAggregationLayers(self.in_dim_dic, self.mid_dim_dic, self.out_dim_dic, num_node_dic['user'],
                                                       num_node_dic['event'], device)
        self.sequence_prediction_layer = SequencePredictionLayer(num_node_dic['user'], num_node_dic['event'], out_size,
                                                                 out_size, 1, device)

        self.user_embedding =  nn.Parameter(torch.randn((num_node_dic['user'], input_size), dtype=torch.float).to(device))
        self.device =device
        self.to(device)

    def forward(self, raw_feature,batch_size,edge_index, ids, sequence):
        raw_feature = self.hals(raw_feature, edge_index)
        raw_feature['user'] = self.user_embedding

        raw_feature = self.sequence_prediction_layer(raw_feature, ids, sequence)
        return raw_feature

    @torch.no_grad()
    def test(self, raw_feature,batch_size,edge_index, ids, sequence):
        raw_feature = self.hals(raw_feature, edge_index)
        raw_feature['user'] = self.user_embedding
        return self.sequence_prediction_layer.test(raw_feature,ids,sequence)

    @torch.no_grad()
    def get_embedding(self, raw_feature, edge_index):

        raw_feature = self.hals(raw_feature, edge_index)
        return raw_feature['event']


class EPSGF_MLP(EPSGF):
    def __init__(self,num_node_dic,num_feature_dic, mid_size, out_size,input_size,device):

        super(EPSGF_MLP, self).__init__(num_node_dic,num_feature_dic, mid_size, out_size,input_size,device)
        self.linear = nn.Linear(num_feature_dic['event'], out_size)
        self.activation = nn.ReLU()
        self.hals = None
        self.to(device)

    def forward(self, raw_feature,batch_size,edge_index, ids, sequence):
        raw_feature = {'user': self.user_embedding, 'event': self.activation(self.linear(raw_feature['event']))}
        raw_feature = self.sequence_prediction_layer(raw_feature, ids, sequence)
        return raw_feature

    @torch.no_grad()
    def test(self, raw_feature, batch_size, edge_index, ids, sequence):
        raw_feature = {'user': self.user_embedding, 'event': self.activation(self.linear(raw_feature['event']))}
        return self.sequence_prediction_layer.test(raw_feature, ids, sequence)

class EPSGF_RAND(EPSGF):
    def __init__(self, num_node_dic, num_feature_dic, mid_size, out_size, input_size, device):
        super(EPSGF_RAND, self).__init__(num_node_dic, num_feature_dic, mid_size, out_size, input_size, device)
        self.linear = nn.Linear(num_feature_dic['event'], out_size)
        self.event_embedding = torch.randn((num_node_dic['event'], out_size), dtype=torch.float).to(device)
        self.hals = None
        self.to(device)

    def forward(self, raw_feature, batch_size, edge_index, ids, sequence):
        raw_feature = {'user':self.user_embedding,'event':self.event_embedding}
        raw_feature = self.sequence_prediction_layer(raw_feature, ids, sequence)
        return raw_feature

    @torch.no_grad()
    def test(self, raw_feature, batch_size, edge_index, ids, sequence):
        raw_feature = {'user': self.user_embedding, 'event': self.event_embedding}
        return self.sequence_prediction_layer.test(raw_feature, ids, sequence)

class EPSGF_SOFTMAX(EPSGF):
    def __init__(self, num_node_dic, num_feature_dic, mid_size, out_size, input_size, device):
        super(EPSGF_SOFTMAX, self).__init__(num_node_dic, num_feature_dic, mid_size, out_size, input_size, device)
        self.sequence_prediction_layer = SequencePredictionLayer(num_node_dic['user'], num_node_dic['event'],
                                                                 out_size, out_size, 1, device, True)
        self.to(device)