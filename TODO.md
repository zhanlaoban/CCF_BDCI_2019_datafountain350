# TODO

1. 实验xlnet-large

2. 调研其他比赛的做法

3. 其他融合方法:如blending,bagging

4. 尝试pseudo-labelling.当前效果不佳,需要进行改进:仅仅提取一千条数据

5. 尝试bert后面接一个全连接层

6. 如果是同一模型,不同参数之间进行融合投票,效果会如何呢?效果会很差.

7. 开始使用stacking

   > 写两份代码:有的模型直接加载预测dev.csv就可以了,有的模型需要重新训练再加载预测dev.csv. 主要是将dev.csv的预测结果保存下来
   >
   > 将各个模型的结果,输入LightGBM或xgb进行第二层的训练

   

8. 开始尝试10-fold的效果:有的模型取得了一定的效果,如bert_base

9. 尝试给warmup加上300步进行训练:效果不行

10. 学习达观杯十强分享汇总PPT:已学习完

11. 加权融合可以试一试,如在投票中,三个模型里面设置4,3,2的权重:效果不错!!!



# 1.stacking设计

和投票不同,stacking时应该不需要考虑单模型的效果如何,只要不太差就可以,并不是要如top3这样的.

当前voting融合中,效果最好提高了0.0067.而stacking应该可以提高0.01

### 具体设计

- 基于5折的stacking,共两层

- 第一阶段基模型: 5个, 'backup1_ERNIE_base', 'backup1_roberta_large', 'backup2_roberta_wwm_ext',

  加上backup1_bert_base,backup2_roberta_l12

- 第二阶段模型: LightGBM.或者LR.或者全连接层

  - 第一阶段的数据为:7340*5=36700
  - 问题是这里如何对第一阶段传来的数据集进行特征处理?
  - 是否需要将第一阶段传来的数据进行CV操作?要不直接拿来训练?答:做CV来训练吧



# 2.类别不平衡问题

```python
train_df['label'].value_counts():
1    3646
2    2931
0     763
Name: label, dtype: int64
```

## 方法一:TTA

> 将0号标签数量提升至->763*5=3815
>
> 将2号标签数量提升至->2931+763=3694
>
> 1号标签数不变,仍为3646
>
> 这样总共为11155条数据

中文->英文->中文

中文->德文->中文

中文->法语->中文

### 参考

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038
https://github.com/PavelOstyakov/toxic/tree/master/tools



# 3.TTA技术

将训练数据和测试数据分别进行TTA转换,这样数据出现了更多的多样性.







# 问题集合

1. **重新做一下EDA分析；如何清洗数据？**

2. 一个问题:存在标签数据类别不均衡的问题

   ```
   train_df['label'].value_counts():
   1    3646
   2    2931
   0     763
   Name: label, dtype: int64
   ```

3. 特征工程

4. 折数是否改变？需要增加吗？

   > 若改变，则所有模型都需要重新训练。
   >
   > 试一下10折robert-large

5. 交叉验证的每个子模型结果的结合是否可以采用投票的方法？

6. 尝试数据增强方法

   > 在群里因为很多人说效果不好, 所以暂不尝试

7. 模型融合：
   3. Voting。等权重和非等权重。
   4. Averaging
   5. stacking
   6. blending

8. 代码阅读：
   1. 那个简单的代码
   2. 本baseline代码

9. KAGGLE ENSEMBLING GUIDE 文章阅读

   > 已阅读完

10. 其他集成学习文章阅读（知乎上的）

11. 因为模型之间相关性越低越好，那么是否有必要训练textcnn,textrnn之类的模型呢？

12. 要不要试试树模型：XGBoost、LightGBM等等，主要是拿来做模型融合的。

13. 如何加入传统特征？如何加入呢？

    > 如TF-IDF。通过ensemble来取长补短。
    >
    > 在融合的时候加入？？
    >
    > 将文本通过传统机器学习方法转换为向量,然后输入到神经网络之中

14. 如果将后面直接接一个全连接层(即bert官方的做法),效果如何?

    > 在这个项目中[GIthub](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch),作者测试接一个BERT后面接一个全连接层的效果比较好.

15. 伪标签 pseudo labeling

16. 回译对BERT没有用吗?
