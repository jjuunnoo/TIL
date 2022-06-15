import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 실습과제
# iris 데이터 사용
# iris['data'][:,0] feature 첫번째 1개만 사용
# iris['target']이 클래스
# x의 최적 분기 조건을 찾아라  최초 분기점만 찾기
# gini index를 사용해서 

iris = load_iris()
x_data = iris['data'][:,0]
y_data = iris['target']
xy_data = np.c_[x_data, y_data]

sort_x = sorted(list(set(x_data)))
condition = [(sort_x[i]+sort_x[i+1])/2 for i in range(len(sort_x)-1)]


def Gini_min(condition):    
    G_cond = []
    for cond in condition:
        dic_yes = {0 : 0, 1 : 0, 2 : 0}
        dic_no = {0 : 0, 1 : 0, 2 : 0}
        for x, y in xy_data:
            if x >= cond:
                dic_yes[int(y)] +=1
            else:
                dic_no[int(y)] +=1
                        
        G_yes = (1 - (dic_yes[0]/sum(dic_yes.values()))**2 - (dic_yes[1]/sum(dic_yes.values()))**2 - (dic_yes[2]/sum(dic_yes.values()))**2 )
        G_no = (1 - (dic_no[0] / sum(dic_no.values()))**2 - (dic_no[1] / sum(dic_no.values()))**2 - (dic_no[2]/sum(dic_no.values()))**2 )
        G = G_yes*(sum(dic_yes.values())/len(xy_data)) + G_no*(sum(dic_no.values())/len(xy_data))
        G_cond.append(G)
    return condition[np.argmin(G_cond)]

Gini_min(condition)



########################### 라이브러리 사용 ########################

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.DataFrame(data= np.c_[iris['data'][:, 0], iris['target']], columns= ['x', 'y'])

x_train = np.array(df['x']).reshape(-1,1)
y_train = df['y']
labels = set(df['y'])

clf = DecisionTreeClassifier(max_depth=1)
clf.fit(x_train, y_train)

plt.figure(figsize = (12,4))
tree.plot_tree(clf, feature_names = ['x'], fontsize=12)
plt.show()



# 실습과제
# iris 데이터 사용
# 8:2 로 스플릿
# D.T 로 x_train y_train 로 학습
# x_test로 y_pred 추정 후 y_test와 비교 정확도 측정
# max_depth 1~20까지 변화하면서 정확도 확인


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2)

test_score = []
train_score = []
d_max = 30
for i in range(1,d_max+1):
    dt_clf = DecisionTreeClassifier(max_depth=i)
    dt_clf.fit(x_train, y_train)
    test_score.append(dt_clf.score(x_test, y_test))
    train_score.append(dt_clf.score(x_train, y_train))

df_test = pd.DataFrame(data=test_score, index=np.arange(1,31), columns=['max_depth'])
df_train = pd.DataFrame(data=train_score, index=np.arange(1,31), columns=['max_depth']) 

plt.figure(figsize = (12,7))
plt.plot(df_test, marker='o', label = 'test score')
plt.plot(df_train, marker='o',label = 'train score')
plt.legend()
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(np.arange(d_max+1))
plt.yticks(np.arange(min(df_test.min()[0], df_train.min()[0]), 1.05, 0.05))
plt.show()

opt_depth = np.argmax(test_score)
opt_acc = test_score[opt_depth]
print(f'max depth: {opt_depth+1}, accuray: {opt_acc:0.3f}')

# 활용단계
x_new = np.array([2.3, 1.8, 3.2, 2.5]).reshape(-1,4)
dt_clf.predict(x_new)