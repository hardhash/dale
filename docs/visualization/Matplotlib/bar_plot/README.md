## 利用pandas读取数据
利用pandas从csv文件导入数据：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.xkcd()
data = pd.read_csv('data.csv')
data.head()
```
数据结构如图所示：
![img.png](img.png)
![img_1.png](img_1.png)
把LanguagesWorkedWith列具体的语言数量统计一下：
```python
from collections import Counter
language_responses=data['LanguagesWorkedWith']
cnt = Counter()
for l in language_responses:
    cnt.update(l.split(';'))
```
![img_2.png](img_2.png)
取前15个：`cnt.most_common(15)`
```python
lang=[]
popularity=[]
for c in cnt.most_common(15):
    lang.append(c[0])
    popularity.append(c[1])
```
## 提取后的数据绘制柱状图
绘制柱状图：`plt.bar(x,y)`

```python
plt.bar(lang,popularity)
plt.title('Top 15 Languages')
plt.xlabel('Language')
plt.ylabel('Popularity')
```
![img_3.png](img_3.png)
发现x轴数据无法完全展示，这里有三种解决方案：

方案1:放大图表`plt.figure(figsize=(10,8))`
![img_4.png](img_4.png)
方案2:x轴文字倾斜60度`plt.xticks(rotation=60)`
![img_5.png](img_5.png)
方案3：翻转x，y轴`plt.barh(lang,popularity)`
![img_6.png](img_6.png)
希望可以是从大到小而不是从小到大排列，则需要对数据倒置。
```python
lang.reverse()
popularity.reverse()
```
![img_7.png](img_7.png)