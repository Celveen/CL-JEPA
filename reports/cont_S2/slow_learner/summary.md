# 离线慢学习器：从过去的 episode 里能学到多少可迁移的东西

训练数据：S2 前两段（I×20、square×20）执行过的转移；每个学习器 300 步、batch 8。
数值 = 5 步开环预测误差（规划器的用法）相对 θ0 的比值，几何平均；<1 更准，>1 变差。
I_future / square_future = S2 第 4、5 段（之后的同领域 episode）；T_block、bank_T/L/Z/+ = 干扰；bank_small_tee = 未出现形状。
latent_var_ratio = 隐表示方差 / θ0 的方差（明显 <1 说明 encoder 在塌缩，误差变小可能是假象）。

| 学习器 | 方式 | I_future | square_future | T_block | bank_T | bank_L | bank_Z | bank_+ | bank_small_tee | latent_var_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delta-mean a=1 | per-domain (future) / shared (interference) | 0.928 | 0.932 | 0.986 | 1.087 | 1.012 | 1.013 | 0.883 | 0.968 | nan |
| delta-mean a=2 | per-domain (future) / shared (interference) | 0.894 | 0.889 | 1.030 | 1.217 | 1.015 | 1.026 | 0.889 | 0.977 | nan |
| pred_last pred_last=0.0001 loss=one | shared | 0.920 | 0.866 | 1.368 | 1.196 | 1.215 | 1.496 | 1.321 | 1.271 | 1.000 |
| pred_last pred_last=0.0001 loss=one | per-domain | 0.811 | 0.883 | 1.587 | 1.400 | 1.314 | 1.597 | 1.622 | 1.275 | 1.000 |
| pred_last pred_last=0.0003 loss=one | shared | 0.914 | 0.894 | 1.659 | 1.421 | 1.904 | 2.206 | 2.298 | 1.516 | 1.000 |
| pred_last pred_last=0.0003 loss=one | per-domain | 0.823 | 0.952 | 2.215 | 1.822 | 1.754 | 2.351 | 2.720 | 1.381 | 1.000 |
| pred_last pred_last=0.001 loss=one | shared | 1.040 | 1.046 | 2.180 | 1.781 | 2.514 | 3.263 | 4.025 | 1.553 | 1.000 |
| pred_last pred_last=0.001 loss=one | per-domain | 0.961 | 1.106 | 3.222 | 2.621 | 2.191 | 3.487 | 4.109 | 1.579 | 1.000 |
| pred_all pred_all=3e-05 loss=one | shared | 0.771 | 0.820 | 1.831 | 1.523 | 1.351 | 2.005 | 2.312 | 1.305 | 1.000 |
| pred_all pred_all=3e-05 loss=one | per-domain | 0.687 | 0.761 | 1.640 | 1.583 | 1.388 | 1.822 | 1.904 | 1.243 | 1.000 |
| pred_all pred_all=0.0001 loss=one | shared | 0.696 | 0.749 | 2.085 | 2.053 | 2.163 | 2.657 | 3.574 | 1.321 | 1.000 |
| pred_all pred_all=0.0001 loss=one | per-domain | 0.650 | 0.750 | 2.026 | 1.883 | 1.911 | 2.481 | 2.377 | 1.270 | 1.000 |
| pred_all pred_all=0.0003 loss=one | shared | 0.699 | 0.760 | 2.743 | 2.557 | 3.129 | 3.918 | 5.148 | 1.387 | 1.000 |
| pred_all pred_all=0.0003 loss=one | per-domain | 0.752 | 0.868 | 2.798 | 2.396 | 2.683 | 3.891 | 3.579 | 1.392 | 1.000 |
| enc_head enc_head=1e-05 loss=one | shared | 8.267 | 13.648 | 48.373 | 31.400 | 26.411 | 43.319 | 49.575 | 12.825 | 0.508 |
| enc_head enc_head=1e-05 loss=one | per-domain | 10.978 | 9.893 | 43.708 | 32.250 | 26.457 | 38.348 | 44.131 | 10.490 | 0.485 |
| enc_head enc_head=0.0001 loss=one | shared | 12.724 | 18.681 | 64.619 | 44.179 | 34.958 | 55.204 | 55.380 | 14.651 | 0.023 |
| enc_head enc_head=0.0001 loss=one | per-domain | 14.738 | 20.160 | 72.228 | 47.880 | 36.829 | 58.167 | 60.019 | 16.462 | 0.025 |
| enc_head enc_head=1e-05 loss=one anchor=10 | shared | 0.964 | 0.977 | 0.952 | 0.971 | 1.025 | 1.049 | 0.963 | 1.025 | 1.000 |
| enc_head enc_head=1e-05 loss=one anchor=10 | per-domain | 0.954 | 0.986 | 0.969 | 1.043 | 1.006 | 1.066 | 1.014 | 1.034 | 1.001 |
| enc_head enc_head=0.0001 loss=one anchor=10 | shared | 0.933 | 0.982 | 1.034 | 1.077 | 1.192 | 1.359 | 1.200 | 1.117 | 1.002 |
| enc_head enc_head=0.0001 loss=one anchor=10 | per-domain | 0.944 | 0.973 | 0.955 | 1.106 | 1.022 | 1.064 | 1.029 | 1.041 | 1.000 |
| enc_all enc_all=1e-05 loss=one | shared | 17.044 | 24.308 | 90.358 | 57.945 | 47.210 | 73.135 | 75.502 | 20.799 | 0.087 |
| enc_all enc_all=1e-05 loss=one | per-domain | 18.368 | 16.291 | 68.823 | 47.371 | 40.040 | 60.708 | 60.955 | 15.960 | 0.033 |
| enc_all enc_all=3e-05 loss=one | shared | 15.717 | 23.531 | 78.108 | 54.521 | 45.510 | 71.512 | 67.894 | 17.145 | 0.069 |
| enc_all enc_all=3e-05 loss=one | per-domain | 15.717 | 24.003 | 78.892 | 55.455 | 44.794 | 68.622 | 68.862 | 18.296 | 0.063 |
| enc_all enc_all=1e-05 loss=one anchor=10 | shared | 0.957 | 0.972 | 0.953 | 1.030 | 0.995 | 1.090 | 1.012 | 1.027 | 1.000 |
| enc_all enc_all=1e-05 loss=one anchor=10 | per-domain | 0.941 | 0.960 | 0.966 | 1.080 | 1.064 | 1.068 | 1.003 | 1.015 | 1.001 |
| enc_all enc_all=3e-05 loss=one anchor=10 | shared | 0.958 | 0.974 | 0.987 | 1.008 | 1.075 | 1.167 | 1.058 | 1.072 | 1.000 |
| enc_all enc_all=3e-05 loss=one anchor=10 | per-domain | 0.929 | 0.984 | 0.968 | 1.078 | 1.106 | 1.098 | 1.009 | 1.025 | 0.999 |
| pred_last+enc_head pred_last=0.0003 enc_head=0.0001 loss=one | shared | 1.823 | 2.429 | 10.129 | 10.692 | 8.017 | 11.772 | 12.832 | 2.938 | 0.576 |
| pred_last+enc_head pred_last=0.0003 enc_head=0.0001 loss=one | per-domain | 1.860 | 2.413 | 11.695 | 10.654 | 8.638 | 11.032 | 14.471 | 2.990 | 0.495 |
| pred_last+enc_head pred_last=0.0003 enc_head=0.0001 loss=one anchor=10 | shared | 0.830 | 0.807 | 1.661 | 1.452 | 1.773 | 2.376 | 2.165 | 1.394 | 1.002 |
| pred_last+enc_head pred_last=0.0003 enc_head=0.0001 loss=one anchor=10 | per-domain | 0.793 | 0.881 | 1.958 | 1.870 | 1.649 | 2.242 | 2.546 | 1.356 | 0.999 |
| pred_last pred_last=0.0001 loss=roll | shared | 0.950 | 0.962 | 1.500 | 1.393 | 1.280 | 1.584 | 1.801 | 1.198 | 1.000 |
| pred_last pred_last=0.0001 loss=roll | per-domain | 0.847 | 1.032 | 1.596 | 1.508 | 1.435 | 1.873 | 1.784 | 1.264 | 1.000 |
| pred_last pred_last=0.0003 loss=roll | shared | 0.894 | 0.957 | 1.687 | 1.877 | 1.550 | 1.829 | 2.726 | 1.367 | 1.000 |
| pred_last pred_last=0.0003 loss=roll | per-domain | 0.859 | 1.072 | 2.094 | 1.954 | 1.815 | 2.607 | 3.093 | 1.481 | 1.000 |
| pred_all pred_all=0.0001 loss=roll | shared | 0.808 | 0.844 | 2.122 | 1.671 | 1.837 | 2.126 | 2.689 | 1.188 | 1.000 |
| pred_all pred_all=0.0001 loss=roll | per-domain | 0.832 | 0.869 | 1.993 | 1.941 | 1.949 | 2.464 | 2.697 | 1.387 | 1.000 |
| pred_all pred_all=0.0003 loss=roll | shared | 1.009 | 1.166 | 3.152 | 3.412 | 3.439 | 4.923 | 6.999 | 1.456 | 1.000 |
| pred_all pred_all=0.0003 loss=roll | per-domain | 1.091 | 1.421 | 3.591 | 3.813 | 3.391 | 4.233 | 4.864 | 1.862 | 1.000 |
| pred_last+enc_head pred_last=0.0003 enc_head=0.0001 loss=roll anchor=10 | shared | 0.855 | 0.907 | 1.359 | 1.522 | 1.777 | 1.462 | 2.518 | 1.260 | 0.997 |
| pred_last+enc_head pred_last=0.0003 enc_head=0.0001 loss=roll anchor=10 | per-domain | 0.874 | 1.020 | 1.882 | 2.029 | 1.899 | 2.143 | 2.718 | 1.501 | 0.998 |
