用户兴趣建模大赛

特征v1: 
用户相关特征：User_ID, 点击率，点赞率，关注率，视频推荐数，点击数，点赞数，关注数，平均播放时长，平均播放时长比，
视频相关特征：Photo_ID,作品时长，人脸特征（占比，性别，年龄，相貌），视频时长

首先对于以前的结果是对每一个用户建模还是整体建一个模，在看了一些博客之后，发现还是建一个模最为可靠。为此，为了比较二者差异，我以MLPC模型，作为例子进行提交。分别建模线上分数为0.68156340， 建一个模线上分数为0.68349758。可见二者的差异并不大，另外注意到我们的先验结果为0.67932901，我们第一次特征工程完的baseline结果和先验差距不大，这说明了我们特征工程的不足。

为了实现添加足够多的变量特征，我学习到了谷歌的一种GBDT+LR( https://cloud.tencent.com/developer/article/1005416 ),这里我只实现了第一种方法。
利用GBDT的叶子结点作为新的特征，然后再利用LR基于这些新的特征进行重新训练预测。由于数据的庞大性，必须进行数据集划分，为了测试特征的重要性以及模型的表现能力，我将处理好的训练集随机划分成了10份，每份有500W sample，用一份进行训练，随机取一份进行验证。由于树的数目决定了特征数量，所以我比较了不同的数目大小对结果的影响
* GBDT+LR (without ID, tree_num=10, 500W samples)
0.713389428527014
* GBDT+LR (without ID, tree_num=20, 500W samples)
0.715630629174749
* GBDT+LR (without ID, tree_num=30, 500W samples)
0.7161264841096022

后来内存原因比较不了tree_num>=50的情况，因此基于上述数据集再次进行了划分5份，这一次每一份有100W samples
* GBDT+LR (without ID, tree_num=30, 100W samples)
0.7183528481948638
* GBDT+LR (without ID, tree_num=50, 100W samples)
0.7159582224648336
* GBDT+LR (without ID, tree_num=100, 100W samples)
0.7159911134238772

基于上述结果，我们大致决定30棵树最佳。
 
在CTR问题上，常常把ID也作为一类特征，这里仅为user_id。处理id类特征有许多方法，特征哈希( http://breezedeus.github.io/2014/11/20/breezedeus-feature-hashing.html ), 等。这里我们进行比较ID对GDBT+LR的影响。
* GBDT+LR (with ID, tree_num=10, 100W samples)
0.7133788310088042
* GBDT+LR (with ID, tree_num=20, 100W samples)
0.7149672754945591
* GBDT+LR (with ID, tree_num=50, 100W samples)
0.7159259932471839
* GBDT+LR (with ID, tree_num=100, 100W samples)
0.7163773840332825
* GBDT+LR (with ID, tree_num=150, 100W samples)
0.7164534200723435

基于上述结果，我们发现了ID的特征加入对结果影响并不大，可能是我ID类特征的处理方式不当。

另外，CTR问题中，最常用的还是FM，FFM，DeepFFM，这里我们借鉴了kaggle上Display Advertising Challenge 的第一名方案( https://github.com/yy1252450987/kaggle-2014-criteo ) ,主要思想还是GBDT对数值型特征进行融合，然后再利用FFM进行预测，这里他们利用C++进行recode以及并行所以速度为比较快。我在子数据集上进行了验证测试。
* GBDT+FFM(100W samples)
0.7167461353582496

其实，GBDT+FFM的模型结果并不尽如人意，我们认为最可能的结果就是我们的分类型特征太少了，仅user_id，这个模型对于高纬稀疏性的数据表现最好，因此，我需要进一步进行特征工程。

接下来的一些想法
（1）特征工程部分： photo_id的被展示次数（可以代表是否为热门推荐视频）, <br>
（2）划分数据集：要注意到一个问题就是时间问题，训练集为0-36h，测试集为36-48h，所以在进行划分时应该考虑到时序关系。 <br>
（3）mini batch: 由于数据集的大小问题，无法一下子全部读入内存的，因此我们需要进行一部分的读入，训练，再读一部分，再训练，scikit-learn 的一些分类器提供了partial-fit，可实现mini batch思想。 <br>
（4）对于text, visual信息，由于已经是整理好的信息了，最方便的方式就是直接作为特征进行加入，考虑到维度问题，上述mini batch可以解决，或者直接降维即可。 <br>

由于现在我还在理论分析上，大部分结果并未提交，所以接下来大佬讲讲如何text信息的处理和利用。
