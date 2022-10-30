import pandas as pd
import os

reg_results = pd.read_csv('reg_results.csv')
mean_fixed = reg_results.groupby(['fixed_image']).mean()

registration_mi = {}
registration_mse = {}

for reg_method in ['rigid', 'affine', 'non_rigid']:
    sorted_pd_mi = mean_fixed.sort_values([reg_method + '_mi'], ascending=False)
    registration_mi[reg_method + '_' + sorted_pd_mi.index[0]] = sorted_pd_mi.iloc[0][reg_method + '_mi']

    sorted_pd_mse = mean_fixed.sort_values([reg_method + '_mse'])
    registration_mse[reg_method + '_' + sorted_pd_mse.index[0]] = sorted_pd_mse.iloc[0][reg_method + '_mse']

print(registration_mse)
print('min based on mse', min(registration_mse, key=registration_mse.get))

print(registration_mi)
print('max based on mi', max(registration_mi, key=registration_mi.get))

print(1)
