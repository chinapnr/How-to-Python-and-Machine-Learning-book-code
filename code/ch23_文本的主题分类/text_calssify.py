# 本例实现一个对文本内容二分类的例子

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import re
import jieba
from sklearn.metrics import f1_score

# 设置变量

# 读取文件路径
file_path = r'sample.txt'

# 是否分词/不分词 （True=分词）
cut_word = True

# 去除特殊字符类型  all=所有
stop_character_type = 'all'

# 停用词词典
stopword = False  # 是否调用停用词词典
stopwords_path = 'stopword.txt'  # 停用词词典路径

# 分词方法选择（目前仅支持jieba）
cut_word_method = 'jieba'

# 是否调用行业自定义词典
load_userdict = False

# 行业自定义词典路径
userdict_path = r'userdict.txt'

# 训练集切割比例
train_ratio = 0.7  # 训练集的比例，取值[0-1]

# 模型参数
model = 'NB'  # 选择分类算法  'NB'=朴素贝叶斯算法, 'SVM'=SVM算法

analyzer = 'char'   # 文本特征组成方式: string, {‘word’, ‘char’}

ngram_range = (1, 2)   # n_gram的上限和下限:  tuple (min_n, max_n)

min_df = 1  # 忽略出现次数小于该值的词项: float in range [0.0, 1.0] or int, default=1

k = 2000  # 保留最有效的k个特征

kernel = 'linear'  # SVM算法的kernal: string, optional,{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}

svm_c = 1.0  # SVM算法的C值（误差项的惩罚系数）

alpha = 0.1  # 朴素贝叶斯算法的alpha值

# 1. 读取文本数据
# 读取文件存入Pandas dataframe
data = pd.read_table(file_path)

# 2. 删除含缺失值的样本
data = data.dropna()

# 3. 分割训练集和测试集
# 随机选取一定比例的数据作为训练样本
print('总数据条数',len(data))
train = data.sample(n=int(len(data)*train_ratio))  # 生成训练集

# 剩余部分作为测试样本
index = data.index.tolist()
train_index = train.index.tolist()  # 生成训练集的index
test_index = [i for i in index if i not in train_index]  # 生成测试集的index
test = data.loc[test_index, ]  # 生成测试集

# 将训练集和测试集进一步拆分成文本和标签，并转成list
text = data.columns[0]  # 获取第一列列名，即存放文本内容的列
label = data.columns[1]  # 获取第二列列名，即存放标签的列（0，1两个类别）
x_train = train[text].tolist()  # 训练集的文本内容
y_train = train[label].tolist()  # 训练集的标签
x_test = test[text].tolist()  # 测试集的文本内容
y_test = test[label].tolist()  # 测试集的标签


# 3. 判断使用分词/不分词方法
if cut_word is True:
    print('使用分词方法')

    # 4. 选择要去除的特殊字符
    if stop_character_type == 'all':  # 去除所有
        stop_character = u'[’（）〈〉：．!"#$%&\'()*+,-./:;～<=>?@，。?↓★、�…【】《》？©“”▪►‘’！•[\\]^_`{|}~]+|[a-zA-Z0-9]'
    elif stop_character_type == 'chararter':  # 去除符号
        stop_character = u'[’（）〈〉：．!"#$%&\'()*+,-./:;～<=>?@，。?↓★、�…【】《》？©“”▪►‘’！•[\\]^_`{|}~]+'
    elif stop_character_type == 'english_word':  # 去除英文
        stop_character = u'[a-zA-Z]'
    elif stop_character_type == 'number':  # 去除数字
        stop_character = u'[0-9]'
    elif stop_character_type == 'url':  # 去除url
        stop_character_type = u'http://[a-zA-Z0-9.?/&=:]*'

    # 去除停用词
    if stopword is True:
        with open(stopwords_path, 'r') as f:
            stopwords_set = {line.strip() for line in f}  # 从停用词字典中获取所有停用词
        print('成功从以下路径获取停用词词典', stopwords_path)
    else:
        stopwords_set = {}
        print('停用词词典为空')

    #  构建jieba分词方法
    if cut_word_method == 'jieba':
        print('使用jieba分词')
        # 是否调用行业自定义词典, True=调用
        if load_userdict is True:
            jieba.load_userdict(f=userdict_path)  # 调用自定义词典，帮助实现更精确的分词（如“汇付天下”等）
            print('使用jieba自定义词典')
        else:
            print('不使用jieba自定义词典')

        def jieba_cutword(paper, stopwords_set):
            r = re.compile(stop_character)
            paper = re.sub(r, '', paper)
            seg_words = jieba.cut(paper, cut_all=False)
            words = [word for word in seg_words if word not in stopwords_set and word != ' ']
            return " ".join(words)

        def my_sentence(paper_list):
            words = []
            for paper in paper_list:
                words.append(jieba_cutword(paper, stopwords_set))
            return words

        x_train = my_sentence(x_train)  # 对数据执行jieba分词
        x_test = my_sentence(x_test)


if cut_word is False:
    print('使用不分词方法')

    stop_character = ''

    # 4. 选择要去除的特殊字符
    if stop_character_type == 'all':
        stop_character = u'[’（）〈〉：．!"#$%&\'()*+,-./:;～<=>?@，。?↓★、�…【】《》？©“”▪►‘’！•[\\]^_`{|}~]+|[a-zA-Z0-9]'
    elif stop_character_type == 'chararter':
        stop_character = u'[’（）〈〉：．!"#$%&\'()*+,-./:;～<=>?@，。?↓★、�…【】《》？©“”▪►‘’！•[\\]^_`{|}~]+'
    elif stop_character_type == 'english_word':
        stop_character = u'[a-zA-Z]'
    elif stop_character_type == 'number':
        stop_character = u'[0-9]'
    elif stop_character_type == 'url':
        stop_character = u'http://[a-zA-Z0-9.?/&=:]*'

    r = re.compile(stop_character)
    x_train = [re.sub(r, '', text) for text in x_train]
    x_test = [re.sub(r, '', text) for text in x_test]

# 5. 文本特征提取
# 对训练集的文本提取特征
vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df)   # 构建TF-IDF特征提取器
fea_train = vectorizer.fit_transform(x_train)  # 对训练集文本提取特征
chi_vectorizer = SelectKBest(chi2, k=k)  # 构建特征筛选器，只保留最有效的k个特征
trainvec = chi_vectorizer.fit_transform(fea_train, y_train)  # 对训练集文本提取最有效的k个特征


# 6. 模型建立

clf = None

if model == 'SVM':  # 模型一：SVM算法
    clf = SVC(kernel=kernel, C=svm_c)  # 可调参数
    clf.fit(trainvec, y_train)

if model == 'NB':   # 模型二：朴素贝叶斯算法
    clf = MultinomialNB(alpha=alpha)  # 可调参数
    clf.fit(trainvec, y_train)


# 7. 利用模型进行预测，并输出准确率表现
# 对预测集进行特征转化
testvec = chi_vectorizer.transform(vectorizer.transform(x_test))

# 利用训练好的模型对测试样本预测
y_pred = clf.predict(testvec)

# 输出混淆矩阵和准确率报告
print('模型预测结果：')
print('混淆矩阵')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('f1分数', f1_score(y_test, y_pred))
print('Test Accuracy:%.2f' % clf.score(testvec, y_test))
