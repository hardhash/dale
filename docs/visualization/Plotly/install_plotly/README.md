## Plotly的安装

方法1:终端输入：`pip3 install plotly`

方法2:命令行输入：`pip install plotly`

方法3:anaconda环境中安装：`conda install plotly`

如果下载速度慢可以使用清华源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple plotly`

## 检查是否安装成功

```python
import plotly
from plotly import __version__
print(__version__)
```
这里显示版本即表示安装成功，我用的是4.14.3版本。