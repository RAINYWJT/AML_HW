import numpy as np

# 原始数据
X = np.array([[3, 4, 4, 6, 3],
              [2, 3, 2, 3, 0]]).T

# 1. 数据中心化
mean = np.mean(X, axis = 0)
X_centered = X - mean

# 2. 计算协方差矩阵
cov_matrix = np.cov(X_centered.T)

# 3. 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 4. 构建投影矩阵（这里本身二维，直接使用特征向量）
projection_matrix = eigenvectors

# 5. 投影数据
result = np.dot(X_centered, projection_matrix)

print("原始数据中心化后：", X_centered)
print("协方差矩阵：", cov_matrix)
print("特征值：", eigenvalues)
print("特征向量：", eigenvectors)
print("投影后的数据：", result)