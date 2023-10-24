## GCN代码框架
PyG依然使用的Torch框架，因此需要对torch有一定的使用经验。在理论部分首先介绍的是GCN（图卷积模型），这个模型本质上就是嵌入向量和邻接混淆矩阵的不断左乘，因此在框架中的写法非常简单：

```python
在框架搭建部分：
self.conv1 = GCNConv(输入节点嵌入维度, hidden_channel1)
self.conv2 = GCNConv(hidden_channel1, hidden_channel2)
...
self.convn = GCNConv(hidden_channen, 最终分类数量)
在前向传播部分：
x = self.conv1(x, 节点与节点之间的连接关系)
```
这里节点与节点的连接关系输入格式如下：

```python
tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,
         33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33],
        [ 1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 17, 19, 21, 31,  0,  2,
         18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32]])
```

## GraphSAGE代码框架

```python
class Net(torch.nn.Module): #针对图进行分类任务
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        ...
        self.lin = torch.nn.Linear(128, output_dim)
        
    def forward(self, data):
    	x, edge_index, batch = data.x, data.edge_index, data.batch
    	...
```
