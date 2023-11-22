import torch as th
from torch import nn
import dgl
import dgl.ops as F
import dgl.function as fn

import deep_neural_decision_forests



class EOPA(nn.Module):
    def __init__(
        self, input_dim, output_dim, batch_norm=True, feat_drop=0.0, activation=None
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def reducer(self, nodes):
        m = nodes.mailbox['m']  # (num_nodes, deg, d)
        # m[i]: the messages passed to the i-th node with in-degree equal to 'deg'
        # the order of messages follows the order of incoming edges
        # since the edges are sorted by occurrence time when the EOP multigraph is built
        # the messages are in the order required by EOPA
        _, hn = self.gru(m)  # hn: (1, num_nodes, d)
        return {'neigh': hn.squeeze(0)}

    def forward(self, mg, feat):
        with mg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            mg.ndata['ft'] = self.feat_drop(feat)
            if mg.number_of_edges() > 0:
                mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
                neigh = mg.ndata['neigh']
                rst = self.fc_self(feat) + self.fc_neigh(neigh)
            else:
                rst = self.fc_self(feat)
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class SGAT(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = activation

    def forward(self, sg, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        q = self.fc_q(feat)
        k = self.fc_k(feat)
        v = self.fc_v(feat)
        e = F.u_add_v(sg, q, k)
        e = self.fc_e(th.sigmoid(e))
        a = F.edge_softmax(sg, e)
        rst = F.u_mul_e_sum(sg, v, a)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class AttnReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.activation = activation

    def forward(self, g, feat, last_nodes):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        feat_u = self.fc_u(feat)
        feat_v = self.fc_v(feat[last_nodes])
        feat_v = dgl.broadcast_nodes(g, feat_v)
        e = self.fc_e(th.sigmoid(feat_u + feat_v))
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        feat_norm = feat * alpha
        rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst
    

# Note: add the semantic information in the node embedding, that is all
class LESSR(nn.Module):
    def __init__(self, num_items, gf_embedding_dim, num_layers, batch_norm=True, feat_drop=0.0, sem_embedding_dim=128):
        # num_items = 38579
        ori_sem_dim = 768
        print("now in __init__ lessr")
        super().__init__()
        self.gf_embedding = nn.Embedding(num_items, gf_embedding_dim, max_norm=1)
        self.sem_dense = nn.Linear(ori_sem_dim, sem_embedding_dim, bias=False)
        # self.sem_embedding = nn.Embedding(num_items, sem_embedding_dim, max_norm=1) we do not need generate the semantic embedding during training
        self.indices = nn.Parameter(th.arange(num_items, dtype=th.long), requires_grad=False)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        input_dim = gf_embedding_dim

        for i in range(num_layers):
            if i % 2 == 0:
                layer = EOPA(
                    input_dim,
                    gf_embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(gf_embedding_dim),
                )
            else:
                layer = SGAT(
                    input_dim,
                    gf_embedding_dim,
                    gf_embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(gf_embedding_dim),
                )
            input_dim += gf_embedding_dim      
            self.layers.append(layer)
        #input dim here is 128

        sem_gf_input_dim = input_dim + sem_embedding_dim
        # embedding_dim = input_dim + sem_embedding_dim
        self.readout = AttnReadout(
            sem_gf_input_dim,
            gf_embedding_dim,
            gf_embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.PReLU(gf_embedding_dim),
        )
        input_dim += gf_embedding_dim # now input dim is 160
        input_dim += sem_embedding_dim # now input dim is 288
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None #Note this 
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(input_dim, gf_embedding_dim, bias=False)

        # self.RF_num_input = gf_embedding_dim
        self.RF_num_input = gf_embedding_dim*(num_layers + 2) + sem_embedding_dim # 32*(3+2)

        num_trees = 16
        depth = 11
        print("Num trees:", num_trees)
        print("Depth:", depth)

        self.RF = deep_neural_decision_forests.NeuralDecisionForest(
            num_trees=num_trees,
            depth=5,
            num_features=self.RF_num_input,
            used_features_rate=0.6,
            num_classes=num_items
        )

    def forward(self, mg, sg=None, sem = None): # sg: shortcut graph, mg: multi graph, sem: semantic embedding for each item
        print("now in forward of LESSR")
        iid = mg.ndata['iid'] # item id, the size of this iid is the same as the # nodes in the graph 3490, 3637 ... 
        feat = self.gf_embedding(iid) # embedding matrix for the nodes shape:([3490, 32]), ([3637, 32]) add the sematic feature here
        '''
        print(mg)
        Graph(num_nodes=3490, num_edges=3037,
                    ndata_schemes={'iid': Scheme(shape=(), dtype=torch.int64), 'last': Scheme(shape=(), dtype=torch.int32)}
                    edata_schemes={}),
        print(sg)
        Graph(num_nodes=3490, num_edges=12067,
                    ndata_schemes={}
                    edata_schemes={})
        '''
        
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                out = layer(mg, feat)
            else:
                out = layer(sg, feat)
            feat = th.cat([out, feat], dim=1)
        # at this step feat.shape torch.Size([3544, 128]) at this step, the feat only contian graph info
        sem = th.zeros(feat.shape[0], 768) #Note remove this loc when semantic embedding is know this is just for testing
        sem = self.sem_dense(sem) # sem: [3544, 768] after the sem_dense it should be [3544, 128]
        feat = th.cat([feat, sem], dim=1) # at this step feat.shape torch.Size([3544, 256]) at this step, the feat contian graph info and sem info
        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        sr_g = self.readout(mg, feat, last_nodes) # sr_g.shape  torch.Size([1024, 32]) -> sr_g.shape  torch.Size([1024, 32])
        sr_l = feat[last_nodes] # sr_l.shape  torch.Size([1024, 128]) ->  sr_l.shape  torch.Size([1024, 256])
        sr = th.cat([sr_l, sr_g], dim=1) # sr.shape  torch.Size([1024, 160]) -> sr.shape  torch.Size([1024, 288])
        RF_result = self.RF(sr) # RF_result.shape  torch.Size([1024, 38579]) -> RF_result.shape  torch.Size([1024, 38579])
        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        sr = self.fc_sr(self.feat_drop(sr)) # torch.Size([1024, 32])



        Embedding_result = sr @ self.gf_embedding(self.indices).t() # Embedding_result torch.Size([1024, 38579]) indices torch.Size([38579])
        
        logits = (Embedding_result + RF_result)/2
        return logits


class LESSR_Dec(nn.Module):
    def __init__(
            self,
            num_items,
            embedding_dim,
            num_layers,
            used_embedding_dim,
            num_leaves,
            batch_norm=True,
            feat_drop=0.0
    ):
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        input_dim = used_embedding_dim
        for i in range(num_layers):
            if i % 2 == 0:
                layer = EOPA(
                    input_dim,
                    used_embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(used_embedding_dim),
                )
            else:
                layer = SGAT(
                    input_dim,
                    used_embedding_dim,
                    used_embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(used_embedding_dim),
                )
            input_dim += used_embedding_dim
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            used_embedding_dim,
            used_embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.PReLU(used_embedding_dim),
        )
        input_dim += used_embedding_dim

        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None

        self.feat_drop = nn.Dropout(feat_drop)

        # The linear function for embedding result
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)
        self.fc_RF = nn.Linear(input_dim, num_leaves)

    def forward(self, mg, sg=None):
        iid = mg.ndata['iid']
        feat = self.embedding(iid)
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                out = layer(mg, feat)
            else:
                out = layer(sg, feat)
            feat = th.cat([out, feat], dim=1)
        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        sr_g = self.readout(mg, feat, last_nodes)
        sr_l = feat[last_nodes]
        sr = th.cat([sr_l, sr_g], dim=1)

        dec = self.fc_RF(self.feat_drop(sr))
        logits = self.fc_sr(self.feat_drop(sr))


        return dec, logits

