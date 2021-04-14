# %%
from numpy import poly
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from preamble import *
%matplotlib inline

# %%
X, y = mglearn.datasets.make_forge()
print("data count:{}".format(len(y)))
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))

# %%
mglearn.plots.plot_knn_classification(n_neighbors=1)

# %%
mglearn.plots.plot_knn_classification(n_neighbors=3)

# %%
# 分离测试数据与样本数据
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 构件KNN分类器
clf = KNeighborsClassifier(n_neighbors=3)
# 训练样本
clf.fit(X_train, y_train)
# 预测测试数据
print("Test set predictions:{}".format(clf.predict(X_test)))
# 评估预测准确率
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
# %%
# 绘图
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# 可视化1，3，9 个邻居
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(
        clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")

# %%
# 下面将使用KNN算法来绘制计算精准度
# 导入数据
cancer = load_breast_cancer()
print("cancer.keys():\n", cancer.keys())
print("Shape of cancer data:", cancer.data.shape)
print("Sample counts per class:\n",
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
print("Feature names:\n", cancer.feature_names)

# %%
# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
# neighbors_settings form 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

# %%
###
# KNN 回归
###
# %%
# wave 数据集
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")


# %%
mglearn.plots.plot_knn_regression(n_neighbors=1)
mglearn.plots.plot_knn_regression(n_neighbors=3)


# %%
X, y = mglearn.datasets.make_wave(n_samples=40)
# 拆分训练样本与测试样本
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 使用KNN回归
reg = KNeighborsRegressor(n_neighbors=3)
# 训练目标
reg.fit(X_train, y_train)
# 预测
print("Test set predictions:\n{}".format(reg.predict(X_test)))
# R^2决定系数 = score
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))


# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    axes[0].legend(["Model predictions", "Training data/target",
                    "Test data/target"], loc="best")
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")
# %%
