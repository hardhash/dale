> 配套代码[DeepWalk](paper_read/gnn/DeepWalk/code/)、[node2vec](paper_read/gnn/node2vec/code/)

这一部分接上文，应该也是属于传统图机器学习部分，但是由于其思想非常重要，因为无论是图机器学习还是图神经网络，都需要把图数据映射成一个低维连续稠密向量，也就是node embedding过程，可以说模型的好坏一切皆基于它。因此单独作为一个章节重点梳理。

Node embedding的出现，表示人工的特征工程逐渐退出历史舞台，一种图表示，随机游走的方法逐渐进入人们视野。在同一个随机游走序列中共同出现的节点，视为相似节点，从而构建类似Word2Vec的自监督学习场景。衍生出DeepWalk、Node2Vec等基于随机游走的图嵌入方法。从数学上，随机游走方法和矩阵分解是等价的。进而讨论嵌入整张图的方法，可以通过所有节点嵌入向量聚合、引入虚拟节点、匿名随机游走等方法实现。

随机游走的方法是DeepWalk一文率先提出，后node2vec继续改进，建议先精读[DeepWalk](paper_read/gnn/DeepWalk/paper/)、[Node2Vec](paper_read/gnn/node2vec/paper/)的论文。而Node2Vec在一定程度上和Word2Vec有相似之处，如果只想学习图网络，可以直接看我[NLP编码](nlp/n2l/)浅做了解。

具体逐字逐句的论文精读和代码实战可以在前面配套连接跳转阅读。