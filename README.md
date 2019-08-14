﻿# JDATA_2019-Cate_Shop_predict
 
京东杯 2019 第六届泰达创新创业挑战赛-用户对品类下店铺购买预测_季军方案

详细文档：https://weavingwong.github.io/2019/06/JDATA-2019-Cate_Shop_predict/

 ## “京东杯”2019第六届泰达创新创业挑战赛比赛攻略（详情见以上文档地址）
 - 比赛链接：https://jdata.jd.com/html/detail.html?id=8
队伍名称： 优生801
### 赛题解读
- 1、赛题背景
目前，京东零售集团第三方平台签约商家超过21万个，实现了全品类覆盖，为维持商家生态繁荣、多样和有序，全面满足消费者一站式购物需求，需要对用户购买行为进行更精准地分析和预测。基于此，本赛题提供来自用户、商家、商品等多方面数据信息，包括商家和商品自身的内容信息、评论信息以及用户与之丰富的互动行为。参赛队伍需要通过数据挖掘技术和机器学习算法，构建用户购买商家中相关品类的预测模型，输出用户和店铺、品类的匹配结果，为精准营销提供高质量的目标群体。
- 2、数据理解
用户对品类下店铺的购买预测的数据包括用户行为、商品评论、商品信息、店铺信息、用户信息5个数据表。由于需要预测的是用户未来7天内对某个目标品类下某个店铺的购买意向，因此时间的相关信息显得尤为重要。
- 3、目标解读
本次大赛通过给出2018-02-01到2018-04-15两个半月内用户U的浏览、购买、收藏、评价、加购等数据信息，需要参赛者预测两个半月后的一周时间（4.16-4.22）可能购买的用户U以及对应购买的品类C和店铺S。
大赛可分为两个部分，一个是需要预测在考察周内用户的购买品类，一个是对应品类下的店铺预测。对每个用户的预测包括用户-品类和相应品类下店铺两个方面，评分采用加权的方式。此处可以分成两个任务去进行分别预测，但是出于对简化问题复杂度的考虑，这里直接合并作为一个二分类任务进行预测。
- 4、赛题评分
参赛者提交的结果文件中包含对所有用户购买意向的预测结果。对每一个用户的预测结果包括两方面：
（1）该用户2018-04-16到2018-04-22是否对品类有购买，提交的结果文件中仅包含预测为下单的用户和品类（预测为未下单的用户和品类无须在结果中出现）。评测时将对提交结果中重复的“用户-品类”做排重处理，若预测正确，则评测算法中置label=1，不正确label=0。
（2）如果用户对品类有购买，还需要预测对该品类下哪个店铺有购买，若店铺预测正确，则评测算法中置pred=1，不正确pred=0。

### 赛题难点
本次比赛分为A，B榜，但是两个榜都是采用同一套数据集。通过EDA分析可知，数据集存在很多噪声，例如加购数据存在大量缺失，浏览数据也存在两天的缺失，2月份数据受春节影响流量异常。如何建模尽可能达到最大的预测准确性。我们将本次比赛的难点归纳为如下几点。
（1）本次比赛的label需要自己构建, 如何建模使我们能在给定的数据集上达到尽可能大的预测准确性，是本次比赛考虑的关键点之一。
（2）对于训练集不同时间段的选取对最终结果都很造成一定的影响，如何选用时间段，让训练集分布和测试集分布类似，也是本次比赛的关键之一。
（3）如何刻画每个时间段的时序特点，使其能够捕捉数据集的趋势性，周期性，循环性。
（4）给来的数据集存在太多影响因素，比如加购数据缺失，浏览数据部分缺失，春节流量异常，节后效应等，所以该如何选取训练集&保证模型稳定的情况。
（5）模型预测出来是概率文件，如何确定划分正负样本的概率阈值，如何确定最优的提交结果数，也是本次比赛不可忽略的关键点之一。

....


### 比赛经验总结
本次大赛没有直接提供训练集对应标签，需要参赛者根据业务数据的理解进行训练集、验证集以及测试集的构建，赛题具有相当的灵活性，但也增加了赛题的难度。针对需要解决的问题和数据特征，我们队伍主要从四个方面进行处理：数据预处理，标签数据集划分，特征工程，模型训练和融合。
对于这次比赛，我们团队有很多收获，也有很多遗憾，首先，在模型选择方面考虑的不多，由于计算资源有限以及平时比较忙的原因（团队只有两个人参与算法编码），没有进行更多的模型尝试和调参，比赛初期有尝试过其他模型，但是由于效果不好，缺乏时间去调参，就放弃了使用其他模型的想法，比赛后期，基本只用XGBoost单模型进行训练。同样，正是由于模型这一块使用的单模，所以我们团队的时间大部分放在了挖掘更有效的特征上面，因此在特征构造这一块我们考虑相对是比较全面的，单模型跑出来的结果也不错。
一些比赛技巧上我们也有待加强，例如比赛前期我们团队一直使用的是有加购的数据，但是有加购的数据存在缺失，无法进行时间划窗采样更多的数据，导致线下验证函数不准确，无法获得比较优的阈值。
    总的来说特征为王，例如比赛前期我们团队一直使用基础特征+计数型特征+排序型特征+比值型特征的方式进行模型训练，时间类特征挖掘的比较少，虽然也取得不错的成绩，但是与前排大佬们还是有一段距离。在比赛后期，我们团队侧重于时间类特征的构造，挖掘了很多强特，在B榜结束前终于冲进了前20。同时提交的结果数也是一个影响线上成绩的一个关键因素，由于每次评测次数有限，所以一个可靠的线下验证方案尤为重要。
限于时间、精力和硬件资源的限制，本次比赛并未尝试使用深度学习相关技术，从赛题题意上来说是可以进行尝试的。
