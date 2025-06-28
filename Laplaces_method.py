#import
import numpy as np
import math_function as m_func
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

#変数
#ベイズ推論のパラメータ(任意)
beta = [-2.0, 1.5]



#人口データの生成
#学習時間を表すxの値をランダム生成
x = random_integers = np.random.randint(0,5, size = 100)
a = beta[0] + beta[1] * x

#学習者が合格する確率
p = m_func.sigmoid(a)

y = np.random.binomial(n = 1, p = p)

#ベルヌーイ分布の可視化とxとyの関係性の可視化
"""
m_func.v_bel(y)
m_func.x_value_plot(x, y)
"""

#2.事後分布の対数をとった関数 (計算の簡単化のため)
L_data = m_func.log_posterior(beta, x, y)

#3.MAP推定値の発見
objective_func = lambda beta: -m_func.log_posterior(beta, x, y)
#最適化の初期位置
initial_beta = np.array([0.0, 0.0])

result = minimize(objective_func, initial_beta, method='BFGS')

# 最適化されたパラメータ（MAP推定値）は .x 属性に入っている
beta_map = result.x

print("最適化成功:", result.success)
print("MAP推定値 (beta_0, beta_1):", beta_map)

#4.ヘッセ行列の計算
# 最適化結果から、ヘッセ行列の「逆行列」の近似値を取得
# 近似された共分散行列になる
covariance_matrix = result.hess_inv

print("近似された共分散行列:")
print(covariance_matrix)

# 2次元正規分布オブジェクトを作成
m_func.visu_con(covariance_matrix, beta_map)
