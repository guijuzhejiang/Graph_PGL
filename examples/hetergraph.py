import numpy as np
import paddle
import paddle.nn as nn
import pgl


#MessagePassing on Heterogeneous Graph
class HeterMessagePassingLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, etypes):
        super(HeterMessagePassingLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.etypes = etypes

        self.weight = []
        for i in range(len(self.etypes)):
            self.weight.append(
                self.create_parameter(shape=[self.in_dim, self.out_dim]))

    def forward(self, graph, feat):
        def send_func(src_feat, dst_feat, edge_feat):
            return src_feat

        def recv_func(msg):
            return msg.reduce_mean(msg["h"])

        feat_list = []
        for idx, etype in enumerate(self.etypes):
            h = paddle.matmul(feat, self.weight[idx])
            msg = graph[etype].send(send_func, src_feat={"h": h})
            h = graph[etype].recv(recv_func, msg)
            feat_list.append(h)

        h = paddle.stack(feat_list, axis=0)
        h = paddle.sum(h, axis=0)

        return h
    

#Create a simple GNN by stacking two HeterMessagePassingLayer.
class HeterGNN(nn.Layer):
    def __init__(self, in_dim, hidden_size, etypes, num_class):
        super(HeterGNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.etypes = etypes
        self.num_class = num_class

        self.layers = nn.LayerList()
        self.layers.append(
            HeterMessagePassingLayer(self.in_dim, self.hidden_size, self.etypes))
        self.layers.append(
            HeterMessagePassingLayer(self.hidden_size, self.hidden_size, self.etypes))

        self.linear = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, graph, feat):
        h = feat
        for i in range(len(self.layers)):
            h = self.layers[i](graph, h)

        logits = self.linear(h)

        return logits
    
seed = 0
np.random.seed(0)
paddle.seed(0)

node_types = [(0, 'user'), (1, 'user'), (2, 'user'), (3, 'user'), (4, 'item'), 
             (5, 'item'),(6, 'item'), (7, 'item')]
num_nodes = len(node_types)
node_features = {'features': np.random.randn(num_nodes, 8).astype("float32")}
labels = np.array([0, 1, 0, 1, 0, 1, 1, 0])

edges = {
        'click': [(0, 4), (0, 7), (1, 6), (2, 5), (3, 6)],
        'buy': [(0, 5), (1, 4), (1, 6), (2, 7), (3, 5)],
    }
clicked = [(j, i) for i, j in edges['click']]
bought = [(j, i) for i, j in edges['buy']]
edges['clicked'] = clicked
edges['bought'] = bought

#build a heterogenous graph by using PGL
g = pgl.HeterGraph(edges=edges, 
                   node_types=node_types,
                   node_feat=node_features)

#Training
model = HeterGNN(in_dim=8, hidden_size=16, etypes=g.edge_types, num_class=2)

criterion = paddle.nn.loss.CrossEntropyLoss()

optim = paddle.optimizer.Adam(learning_rate=0.05,
                              parameters=model.parameters())

g.tensor()
labels = paddle.to_tensor(labels)
for epoch in range(10):
    # print(g.node_feat["features"])
    logits = model(g, g.node_feat["features"])
    loss = criterion(logits, labels)
    loss.backward()
    optim.step()
    optim.clear_grad()

    print("epoch: %s | loss: %.4f" % (epoch, loss.numpy()[0]))