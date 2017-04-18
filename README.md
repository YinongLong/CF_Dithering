# CF & Dithering

根据Item-CF以及User-CF的推荐结果实现Dithering。

- utility.py

这个脚本主要是对于数据文件的读取，包括需要指定的用来构建CF模型的用户-物品评分记录，以及用来测试模型推荐结果的测试数据集。
还有包含电影ID与电影名称对应的信息。

- collaborative.py

这个文件里面实现了ItemCF和UserCF两种类型的协同过滤算法，其中关于用户相似度以及物品相似度的计算都是采用简单的余弦相似
度的计算方式，但是后续的其他相似度的计算还可以添加进来。其次，在这个文件中也实现了Dithering，其中Dithering的实现是
根据模型已经推荐出的结果，然后根据一个formulation来计算加入一定noise的dithering结果，其中formulation如下：

$$S_i = \alpha * \log(Rank(i)) + (1 - \alpha) * N(0, sqrt(\log(\beta)))$$

然后对于计算出的 $S_i$ 进行升序排序，即可得到以参数 $\alpha$ 和参数 $\beta$ 抖动后的推荐结果。


其中关于 $\alpha$ 参数和 $\beta$ 参数的组合有：

- $\alpha = 0.1$，$\beta = 1.1$
- $\alpha = 0.1$，$\beta = 2.0$
- $\alpha = 0.1$，$\beta = 5.0$
- $\alpha = 0.5$，$\beta = 1.1$
- $\alpha = 0.5$，$\beta = 2.0$
- $\alpha = 0.5$，$\beta = 5.0$
- $\alpha = 0.9$，$\beta = 1.1$
- $\alpha = 0.9$，$\beta = 20.0$
- $\alpha = 0.9$，$\beta = 5.0$