## 键盘输入
首先新建一个工程。
![img.png](img.png)
点击数据资料。
![img_1.png](img_1.png)
选中节点和添加节点，即可添加对应节点。并且在概览中可以看到添加的节点。
![img_2.png](img_2.png)
![img_3.png](img_3.png)
![img_4.png](img_4.png)
![img_5.png](img_5.png)
![img_6.png](img_6.png)
![img_7.png](img_7.png)

## 导入CSV文件
手动输入只能应付较少节点和边的情况，通常数据都是放在excel表格中，此时使用csv文件导入会更加便捷。但是需要注意，必须把csv文件编辑成gephi能识别的形式，可以使用python的pandas库进行数据处理工作。

这里手动编辑了一个描述边的csv文件：
![img_8.png](img_8.png)
在数据资料部分，点击导入电子表格，找到创建好的.csv文件，选择导入即可。
![img_9.png](img_9.png)
![img_10.png](img_10.png)
此时，在概览可以看到导入的图：
![img_11.png](img_11.png)
## 直接在概览界面鼠标点击创建
创建新工程文件后，点击创建节点，在画布区双击即可创建一个节点。
![img_12.png](img_12.png)
![img_13.png](img_13.png)
接下来选择画点下面的功能，选择画边
![img_14.png](img_14.png)
先选中一个源节点，再选中一个目标节点，即可在两点见创立一个连线。
![img_15.png](img_15.png)
## 自己创建一个红楼梦关系网络图
![img_16.png](img_16.png)
创建这样一个人物关系.csv文件。然后倒入gephi。
![img_17.png](img_17.png)
![img_18.png](img_18.png)
然后在布局，选择第一个搅拌一下。
![img_19.png](img_19.png)
然后再选择第三个，把他展开。
![img_20.png](img_20.png)
然后可以在左侧外观部分，进行外观调整。得到下图，重要的节点变大，重要的边颜色更深。
![img_21.png](img_21.png)
调整完布局之后，进入预览，调整变和节点的属性，优化细节，可以得到下图：
![img_22.png](img_22.png)
导出为png格式图片：
![img_23.png](img_23.png)
## 用一个Web of Science上的数据创建一个有向关系图
这里有一份Web of science上2019-2022年3年来有关混凝工艺的一些研究报道数据。该数据包括研究人员，研究类型，国籍，研究内容等多个关联内容。
![img_24.png](img_24.png)
导入Gephi后，即可绘制关系网络图。这个图表示了不同的研究人员之间研究的共性。（外观后面再介绍如何调整）
![img_25.png](img_25.png)