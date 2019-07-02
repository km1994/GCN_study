# DGL 入门

## 引言


## DGL 核心 —— 消息传递 

DGL 的核心为消息传递（message passing），主要分为消息函数 （message function）和累加函数（reduce function）。如下图所示：

![](img/message_passing.png)

消息函数（message function）利用边获得该边出发节点的表示（e.src.data）、该边目标节点的表示（e.dst.data）、该边自身表示（e.data）。

累加函数（reduce function）通过汇聚目标节点表示和消息函数（message function）所传递过来的表示，以获得一个新的表示。

## GCN 的数学表达

GCN 的逐层传播公式如下所示：

$H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$

该公式的形象解释：每个节点拿到邻居节点信息然后汇聚到自身 embedding 上。具体 GCN 内容介绍可参考【[GNN 教程：GCN](https://archwalker.github.io/blog/2019/06/01/GNN-Triplets-GCN.html)】。

## 从消息传递的角度分析 GCN

本章，我们将从消息传递的角度对 GCN 进行分析，其分析过程可以被概括为以下步骤：

1. 在 GCN 中每个节点都有属于自己的表示 $h_i$;
2. 根据消息传递（Message passing）的范式，每个节点将会收到来自邻居节点发送的 message（表示）；
3. 每个节点将会对来自邻居节点的 message（表示）进行汇聚以得到中间表示 $\hat{h}_{i}$ ；
4. 对中间节点表示 $\hat{h}_{i}$ 进行线性变换，然后在利用非线性函数$f$进行计算：$h^{new}_{u}=f\left(W_{u} \hat{h}_{u}\right)$;
5. 利用新的节点表示 $h^{new}_{u}$ 对该节点的表示 $h_{u}$进行更新。

## 具体实现

在 GCN 的具体实现过程中，可以分为消息函数 （message function）和累加函数（reduce function）定义模块和非线性函数$f$定义模块。



```python
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
net = Net()
print(net)

from dgl.data import citation_graph as citegrh
def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    mask = th.ByteTensor(data.train_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(g.selfloop_edges())
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask

import time
import numpy as np
g, features, labels, mask = load_cora_data()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(30):
    if epoch >=3:
        t0 = time.time()

    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))
```






## 参考资料

1. [DGL Basics](https://docs.dgl.ai/tutorials/basics/2_basics.html)
2. [DGL 作者答疑！关于 DGL 你想知道的都在这里](https://mp.weixin.qq.com/s?__biz=MzI2MDE5MTQxNg==&mid=2649695390&idx=1&sn=ad628f54c97968d6fff55907c47cb77e)