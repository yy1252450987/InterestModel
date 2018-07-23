v1 特征利用LR对每个变量进行单独的回归分析，由于python 不提供p-value值，即以validation的auc结果作为评判变量重要性的标准。<br>
* ('time', 0.5025972114461132)<br>
* ('duration_time', 0.5064001319726044)<br>
* ('view_count', 0.614818514873556)<br>
* ('click_count', 0.5516637992594825)<br>
* ('click_rate', 0.7090988482012806)</td><br>
* ('like_count', 0.5157279999104014)<br>
* ('like_rate', 0.534279809166853) <br>
* ('follow_count', 0.503836169387149)<br>
* ('follow_rate', 0.513719607778748)<br>
* ('playing_time_mean', 0.6845685441680154)<br>
* ('playing_time_ratio_mean', 0.6866974724395316)<br>
* ('face_num', 0.5050447162606572)<br>
* ('face_ratio', 0.518149796684362)<br>
* ('age_weight', 0.5002675032103372)<br>
* ('beauty_weight', 0.5134135759040895)<br>
* ('sex_0.0', 0.5128795943126225)<br>
* ('sex_1.0', 0.5013272491928207)<br>
* ('sex_2.0', 0.5039264848610463)<br>
* ('has_face', 0.5076258602587556)<br>

LR importance feature: view_count, click_rate, playing_time_mean, playing_time_ratio_mean

GBDT importance feature: feature importances: 

click_rate=636, duration_time=306,face_ratio=269, age_weight=122, playing_time_ratio_mean=42, click_count=32, view_count=26, playing_time_mean=15, beauty_weight=2

visual 信息是否需要处理？为此先试试全部加入对结果的影响，同样以100Wsample 的数据进行训练和验证
特征v1:每次读入10000,读入5次，训练mlpc(8,8,4,4)， 预测10000，('AUC: ', 0.7112641140019018)<br>
特征v1+visual(1024维): 每次读入10000,读入五次，mlpc(512，128，32，4)，预测10000, ('AUC: ', 0.6300197353789786)<br> 大概耗时5-6分钟
visual(1024维)：每次读入10000,读入五次，mlpc(512，128，32，4)，预测10000,<br>


特征v2:<br>

face部分特征离散化，增强模型的非线性拟合能力：<br>
* face_num -> 0(非人物图片), 1（一人）, 2（两人）, >=3（多人）
* age_weight -> 0-3(婴幼儿)，3-7(学龄前儿童),7-13（小学生）,13-18（初中生）, 18-25（大学生）,25-30（研究生）, 30-（青年）
* beauty_weight -> 非常漂亮(80-100)，漂亮(60-80)， 一般(40-60)，丑陋(<40)

user部分异常值的处理:<br>
用户3434的playing_time 出现异常，‘3434	5893764	1	0	0	761119287418	1912441	20’ 将1912441--> 19<br>
对长尾分布进行处理：<br>
  ##like_count : if like_count>5 return 5<br>
  ##follow_count: if follow_count>5 return 5<br>


用户每一时间戳上的点击率，序列化；
视频被展示次数，率。
