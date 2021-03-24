## 任务目标

通过机器学习或者深度学习模型实现“非对称文本语义匹配任务”（指两个表述和篇幅具有较大差异但语义相似的文本之间的匹配任务）

## 模型选择

[BiMPM](https://github.com/zhiguowang/BiMPM)是一个常用于文本匹配的模型，任务中给出的数据集与WikiQA高度相似，根据论文描述，BiMPM在WikiQA数据集上的分类准确度可以达到0.731，因此考虑使用这个较为成熟的模型，有完整的方案可供参考。

>  ![模型图示](./log/figures/pic1.jpg)
>
> \* 图片来自原论文



## 训练过程

任务只给出了训练集而未给出验证集，因此对训练集做了预处理，将训练集以8:2的比例划分为训练集和验证集。

使用GloVE作为词向量。

对于二分类问题，损失函数设定为二元交叉熵。

我的笔记本配置不高，BiMPM模型非常复杂，在我的笔记本上训练速度非常缓慢。在设置batch_size=16的情况下（大于这个数值我的机器就显存不足了），训练50个epoch需要耗时7个小时。在这个速度下，我基本没有办法对模型的参数进行调整，因此大部分参数保持默认，并没有调节到最优状态。最终训练得到的模型在测试集上的分类准确度约为0.6702。

![acc](./log/figures/ep_acc.png)

![acc](./log/figures/ep_loss.png)

```shell
$ python3 eval.py
0.670206121174266
```



## 环境配置

* tensorflow==1.15.0
* nltk==3.2.5
* numpy==1.16.0
* Keras==2.2.4

使用了NTLK对句子作预处理，需要下载语料库。在命令行下输入`python`，然后：

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```



## 使用方法

* 预处理：将训练集划分为训练集和验证集，对Question和Answer句子作预处理——去除停用词和词形还原

  ```python
  python main.py --pre
  ```

* 训练

  ```python
  python main.py --train
  ```

* 生成pred.csv（文件夹中已包含训练好的模型，可以直接生成）

  ```python
  python main.py --gen
  ```

* 评估准确率

  ```python
  python eval.py
  ```

  

## 参考

### 模型源码来自

* [BiMPM_keras](https://github.com/ijinmao/BiMPM_keras)

### 文献

* [BiMPM: Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf)
