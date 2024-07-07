在torch_geometric中封装了构造数据集的方法。需要指定节点信息，类别信息和边信息。

使用如下代码模板即可实现pyg数据格式构建。
```python
import torch
from torch_geometric.data import Data

# 点特征
x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
y = torch.tensor([0, 1, 0 ,1], dtype=torch.float)

# 边
edge_index = torch.tensor([[0,1,2,0,3], #Start
                           [1,0,1,3,2]], dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)
```

## 雅虎电商数据构建


yoochoose-clicks.dat保存了用户浏览行为，session_id表示一次登录，item_id表示该次登录浏览的商品，yoochoose-buys描述最终是否购买(labels)

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.read_csv('yoochoose-clicks.dat', header=None)
df.columns = ['session_id', 'timestamp', 'item_id', 'category']

buy_df = pd.read_csv('yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price','category']

item_encoder = LabelEncoder()
df['item_id'] = item_encoder.fit_transform(df.item_id)

sampled_session_id = np.random.choice(df.session_id.unique(), 100000, replace=False)
df = df.loc[df.session_id.isin(sampled_session_id)]
df['label'] = df.session_id.isin(sampled_session_id)
```

    构建数据集

    1. 遍历每一组session_id，制作成torch_geometric.data格式
    2. 由于data格式需要从0开始，因此对每一组session_id中所有item_id进行编码
    3. 构建边
    4. 保存数据集

```python
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
 
class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform) # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])
 
    @property
    def raw_file_names(self): #检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件 
                              #如有文件不存在，则调用download()方法执行原始文件下载
        return []
    
    @property
    def processed_file_names(self): #检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['yoochoose_click_binary_1M_sess.dataset']
 
    def download(self):
        pass
    
    def process(self):
        
        data_list = []
 
        # process by session_id
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id==session_id,['sess_item_id','item_id']].sort_values('sess_item_id').item_id.drop_duplicates().values
 
            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]
 
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features
 
            y = torch.FloatTensor([group.label.values[0]])
 
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
dataset = YooChooseBinaryDataset(root='data/')
```

> GPU加速

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
d = []
for data in dataset:
    d.append(data.to(device))
```

> 模型搭建

```python
embed_dim = 128
from torch_geometric.nn import TopKPooling,SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
class Net(torch.nn.Module): #针对图进行分类任务
    def __init__(self):
        super(Net, self).__init__()
 
        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() +10, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x) # n*1*128 特征编码后的结果
        x = x.squeeze(1) # n*128   
        x = F.relu(self.conv1(x, edge_index))# n*128
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)# pool之后得到 n*0.8个点
        x1 = gap(x, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = gap(x, batch)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = gap(x, batch)
        x = x1 + x2 + x3 # 获取不同尺度的全局特征
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin3(x)).squeeze(1)# batch个结果
        return x
```

> 训练

```python
from torch_geometric.loader import DataLoader

def train():
    model.train()
 
    loss_all = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)
    
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()
train_loader = DataLoader(dataset, batch_size=256)
for epoch in range(10):
    print('epoch:',epoch)
    loss = train()
    print(loss)
```


