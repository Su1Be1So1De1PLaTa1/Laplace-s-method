#ラプラス近似実装のための関数

#import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import multivariate_normal

#シグモイド関数
def sigmoid(value):
    return 1/(1 + np.exp(-value))

#ベルヌーイ分布の可視化
def v_bel(value):
    v = [0,0]
    x = [0,1]
    labels = ["0","1"]
    for i in value:
        if i == 0:
            v[0] += 1
        else:
            v[1] += 1
    
    fig, ax = plt.subplots()

    ax.bar(x, v)
    plt.xticks(x,labels)
    plt.show()


#学習時間と合格率の関係性
def x_value_plot(x, y):
    # xとyの散布図をプロット
    # xとyからデータフレームを作成
    df = pd.DataFrame({'x': x, 'y': y})

    # xの値でグループ化し、yの平均値を計算
    mean_y_by_x = df.groupby('x')['y'].mean()

    # 結果をプロット
    plt.plot(mean_y_by_x.index, mean_y_by_x.to_numpy(), marker='o')
    plt.xlabel("Study Time (x)")
    plt.ylabel("Pass Rate (Mean of y)")
    plt.title("Pass Rate by Study Time")
    plt.grid(True)
    plt.show()


#事後分布の対数をとった関数 (計算の簡単化のため)
def log_posterior(beta, x, y):
    beta_0 = beta[0]
    beta_1 = beta[1]
    #「事後分布の対数をとった関数」の計算で使う定数　0でもOK
    C = 0

    p = sigmoid(beta_0 + beta_1 * x) 
    #尤度の計算式(尤度 = p^y * (1-p)^(1-y))
    """
    p_u = p**y * (1 - p)**(1 - y)
    p_u_sum = np.sum(p_u)
    """
    #上記の計算を安定化させたもの
    #scipy.stats.bernoulli.logpmf(y, p): ベルヌーイ分布の対数確率質量関数（Log PMF）
    log_likelihood = np.sum(stats.bernoulli.logpmf(y, p))

    #(stats.norm.logpdf(beta_0)) = ln(p(beta_0))  正規分布音確率密度関数を計算
    L_data = log_likelihood + (stats.norm.logpdf(beta_0)) + (stats.norm.logpdf(beta_1)) + C

    return L_data

def visu_con(covariance_matrix, beta_map):
    # ステップ1: 描画範囲の設定
    # 平均値(beta_map)と共分散行列から、各パラメータの標準偏差を計算
    b0_std = np.sqrt(covariance_matrix[0, 0])
    b1_std = np.sqrt(covariance_matrix[1, 1])

    # 平均値から標準偏差の約3倍の範囲を描画範囲とする
    x_range = [beta_map[0] - 3 * b0_std, beta_map[0] + 3 * b0_std]
    y_range = [beta_map[1] - 3 * b1_std, beta_map[1] + 3 * b1_std]

    # ステップ2: グリッド（計算用の格子）の作成
    # 描画範囲を100x100の格子に分割
    b0_grid = np.linspace(x_range[0], x_range[1], 100)
    b1_grid = np.linspace(y_range[0], y_range[1], 100)
    B0, B1 = np.meshgrid(b0_grid, b1_grid)

    # ステップ3: 各グリッドポイントでの確率密度の計算
    # 2次元正規分布オブジェクトを作成
    approx_posterior = multivariate_normal(mean=beta_map, cov=covariance_matrix)

    # 各グリッドポイントの座標をまとめる
    pos = np.dstack((B0, B1))
    # 各グリッドポイントでの確率密度(高さ)を計算
    Z = approx_posterior.pdf(pos)

    # ステップ4 & 5: 等高線プロットの描画と中心点のプロット
    plt.figure(figsize=(8, 7))

    # 背景を塗りつぶした等高線プロットを作成
    plt.contourf(B0, B1, Z, levels=10, cmap='Blues')

    # 線のみの等高線プロットを重ねる
    C = plt.contour(B0, B1, Z, levels=10, colors='black', linewidths=0.5)

    # MAP推定値（山の頂上）を赤い点でプロット
    plt.scatter(beta_map[0], beta_map[1], c='red', s=50, zorder=10, label='MAP Estimate')

    # グラフの装飾
    plt.xlabel('beta_0')
    plt.ylabel('beta_1')
    plt.title('Approximated Posterior Distribution (Contour Plot)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.colorbar(C, label='Probability Density') # カラーバーを追加

    # グラフを表示
    plt.show()

