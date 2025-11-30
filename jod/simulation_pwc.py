
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import itertools
import random
import jod_normal_distribution as jodlib
import asap_cpu
import tqdm

def boostrap_resample_Cmatrix(C_matrix_samples, n_samples=None):
    """_summary_

    该函数用于对给定的协方差矩阵样本进行自助法重采样，并返回重采样后的协方差矩阵。
    Args:
        C_matrix_bootstrap_samples (_type_): shape NxN
        n_samples (_type_): _description_
    """
    length=len(C_matrix_samples)
    N=np.sqrt(length).astype(int)
    if n_samples is None:
        n_samples=length
    C_matrix_bootstrap_samples=np.random.choice(
            length, 
            size=n_samples, 
            replace=True,
            p=C_matrix_samples/np.sum(C_matrix_samples))
    # C_matrix_bootstrap_cnt=np.zeros_like(C_matrix_bootstrap_samples)
    # for idx in C_matrix_bootstrap_samples:
    #     C_matrix_bootstrap_cnt[idx] +=1
    counts = np.bincount(C_matrix_bootstrap_samples, minlength=length)  # minlength保证索引全覆盖
    # 2. 按原索引映射回计数数组（与原samples同形状）
    C_matrix_bootstrap = counts.reshape((N,N))
    # print(C_matrix_bootstrap,C_matrix_bootstrap.sum())
    return C_matrix_bootstrap

def Cmatrix_to_Quality(C_matrix):
    # Example of running ASAP on CPU
    N = np.shape(C_matrix)[0]
    asap = asap_cpu.ASAP(N, selective_eig = True)
    pairs_to_compare = asap.run_asap(C_matrix, mst_mode=True)

    # Get current score estimation
    scores_mean, scores_std = asap.get_scores()
    scale=1.5127867480084503
    scores_mean = scores_mean * scale
    # print("Scale to make the lowest score 1: ", scale)
    # print("Indeces from pwc_mat to compare:")
    # print(pairs_to_compare)
    # print("Scores means \n",scores_mean)
    # print("Scores standard deviaion \n", scores_std)
    return scores_mean, scores_std


def simulation_one_dataset(n_observer=10,Isshow=False):
    n_boostrap=10
    n_repetition=3
    n_condition=2
    qualitys=np.arange(n_condition)
    # qualitys=np.array([0,1,3,4,6])
    sigma_ij=1 / norm.ppf(0.75, 0, 1) # 标准差设为1.4826，以匹配JOD尺度的标准差 xscale
    n_experiment=np.math.comb(n_condition, 2)
    
    N=n_observer * n_repetition 
    n_sample = n_experiment * N
    # 模拟每个观察者对每个条件的评分
    condition_pairs=list(itertools.combinations(qualitys, 2))
    C_matrix=np.zeros((n_condition, n_condition)) # 记录每对条件的比较结果,count of i preferred over j
    # print("Simulating pairwise comparisons...")        
    for i in range(n_condition):
        for j in range(i+1,n_condition):
            # 遍历配对比较 n_experiment
            quality_i=qualitys[i]
            quality_j=qualitys[j]
            # 计算两个条件的质量差异
            delta_quality=quality_i - quality_j # y_ij, d_ij
            tau=-delta_quality/sigma_ij
            # 基于标准正态分布模拟观察者的选择
            c_ij_samples=np.random.normal(loc=0, scale=1, size=N)
            c_ij_samples=c_ij_samples> tau  # 观察者选择i优于j的情况
            c_ij_count=np.sum(c_ij_samples)
            C_matrix[i, j]=c_ij_count
            C_matrix[j, i]=N - c_ij_count
    # print(C_matrix,C_matrix.sum())

    # Bootstrap 重采样估计不确定度
    #     we generate a new sample of the same size by randomly replicating data 
    # for some participants and  removing data for others. 
    # The procedure is know as random sampling with replacement. 
    
    C_matrix_samples=C_matrix.ravel()
    iter_qualitys_mean_list=[]
    for b in (range(n_boostrap)):

        C_matrix_bootstrap = boostrap_resample_Cmatrix(C_matrix_samples,n_samples=n_sample)
        iter_qualitys_mean, iter_scores_std = Cmatrix_to_Quality(C_matrix_bootstrap)
        iter_qualitys_mean-=np.min(iter_qualitys_mean)
        iter_qualitys_mean_list.append(iter_qualitys_mean)
    iter_qualitys_mean_array=np.array(iter_qualitys_mean_list)
    qualitys_mean_std=np.std(iter_qualitys_mean_array, axis=0)
    qualitys_75=np.percentile(iter_qualitys_mean_array,75,axis=0)
    qualitys_25=np.percentile(iter_qualitys_mean_array,25,axis=0)
    qualitys_975=np.percentile(iter_qualitys_mean_array,97.5,axis=0)
    qualitys_025=np.percentile(iter_qualitys_mean_array,2.5,axis=0)
    qualitys_mean=np.mean(iter_qualitys_mean_array,axis=0)
    # print("qualitys_25",qualitys_25)
    # print("qualitys_75",qualitys_75)
    # print("qualitys_mean",qualitys_mean)
    # print("qualitys_mean_std",qualitys_mean_std)
    


    # 生成模拟数据（正态分布）
    np.random.seed(42)  # 固定随机种子，保证结果可复现
    data = np.random.normal(loc=100, scale=20, size=200)  # 均值100，标准差20，200个数据点
    
    if not Isshow:
        return qualitys_mean, qualitys_mean_std,qualitys_25,qualitys_75,qualitys_025,qualitys_975

    # 创建画布
    plt.figure(figsize=(8, 6))

    # 绘制箱线图
    box_plot = plt.boxplot(
        iter_qualitys_mean_array,
        patch_artist=True,  # 填充箱体颜色
        whis=[2.5, 97.5],            # 须覆盖2.5% ~ 97.5%分位数（核心参数）
        positions=qualitys,  # 强制指定每个箱体的位置
        boxprops=dict(facecolor='lightblue', color='blue'),  # 箱体样式
        medianprops=dict(color='red', linewidth=2),  # 中位数样式
        whiskerprops=dict(color='green'),  # 须样式
        capprops=dict(color='black'),  # 须端横线样式
        flierprops=dict(marker='o', color='orange', markersize=8)  # 异常值样式
    )
    plt.plot(qualitys, qualitys, 'g--', label='Mean')
    # 添加标题和标签
    plt.title(f'Boxplot of Simulated Data, real qualitys={qualitys}\n , \
              t={n_repetition}, n={n_condition}, k={n_observer}, boost={n_boostrap} ')
    plt.xlabel('real quality scores')
    plt.ylabel('sampled quality scores')
    plt.ylim(-0.2, 8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加横向网格线
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # 添加纵向网格线

    # 显示图表
    plt.show()
    return qualitys_mean, qualitys_mean_std,qualitys_25,qualitys_75
    

if __name__ == "__main__":
    
    
    motocalo_num=100
    obser_CIs=[]
    num_observer_list=np.arange(10, 101, 30)
    for n_observer in num_observer_list:
        CI_s=[]
        for seti in range(motocalo_num):
            print("n_observer",n_observer,"seti",seti)
            qualitys_mean, qualitys_mean_std,qualitys_25,qualitys_75,qualitys_025,qualitys_975= simulation_one_dataset(n_observer=n_observer,Isshow=False)
            CI_s.append(qualitys_975- qualitys_025)
        # get CI mean std , 025 25 75 975
        CI_s=np.array(CI_s)
        CI_mean=np.mean(CI_s,axis=0)
        CI_std=np.std(CI_s,axis=0)
        CI_025=np.percentile(CI_s,2.5,axis=0)
        CI_25=np.percentile(CI_s,25,axis=0)
        CI_75=np.percentile(CI_s,75,axis=0)
        CI_975=np.percentile(CI_s,97.5,axis=0)
        obser_CIs.append(CI_mean)
        # print("n_observer",n_observer,"CI_mean",CI_mean,"CI_std",CI_std,"CI_025",CI_025,"CI_25",CI_25,"CI_75",CI_75,"CI_975",CI_975)
    # 绘制箱线图
    plt.figure(figsize=(8, 6))
    box_plot = plt.boxplot(
        obser_CIs,
        patch_artist=True,  # 填充箱体颜色
        whis=[2.5, 97.5],            # 须覆盖2.5% ~ 97.5%分位数（核心参数）
        positions=num_observer_list,  # 强制指定每个箱体的位置
        boxprops=dict(facecolor='lightblue', color='blue'),  # 箱体样式
        medianprops=dict(color='red', linewidth=2),  # 中位数样式
        whiskerprops=dict(color='green'),  # 须样式
        capprops=dict(color='black'),  # 须端横线样式
        flierprops=dict(marker='o', color='orange', markersize=8)  # 异常值样式
    )
    plt.boxplot(obser_CIs, patch_artist=True, whis=[2.5, 97.5], positions=num_observer_list)
    plt.title('Boxplot of CI for different number of observers')
    plt.xlabel('Number of Observers')
    plt.ylabel('CI')
    plt.show()

        
        
    
    
    
        
        
        
        
    
        
    # transform C_matrix to  P_matrix , probability of i preferred over j
    # P_matrix=np.zeros((n_condition, n_condition))
    # for i in range(n_condition):
    #     for j in range(i+1,n_condition):
    #         P_matrix[i, j]=C_matrix[i, j]/(C_matrix[i, j]+C_matrix[j, i])
    #         P_matrix[j, i]=1 - P_matrix[i, j]
    # # transform P_matrix to D_matrix , JOD distance between i and j
    # D_matrix=np.zeros((n_condition, n_condition))
            
            
    
    