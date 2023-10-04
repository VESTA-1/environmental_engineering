from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process.kernels import RationalQuadratic
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_validate
import numpy as np
import warnings
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

start = time.time()
def warn(*args, **kwargs):
    pass
warnings.warn = warn

'''data載入'''
data_name = 'train1'
data_path = f'./data/{data_name}.csv'
ori_data = pd.read_csv(data_path, index_col=0)
# x_data = ori_data.iloc[:,1:].to_numpy().astype('float32')
# y_data = ori_data.iloc[:,:1].to_numpy().ravel()
# scaler = MinMaxScaler()
# scaler.fit(x_data)
# x_data = scaler.transform(x_data)

'''全域變數'''
values_list = list(i * 5 /1000 for i in range(2,201,1))
recorder = list(i for i in range(1, len(ori_data)+1))
random_seed = 42


'''分割資料集'''
def split_train_test(data, size):
    x_data = data.iloc[:,1:].to_numpy().astype('float32')
    y_data = data.iloc[:,:1].to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)
    y_data = y_data.ravel()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=size, random_state=random_seed)
    return (x_data, y_data), (x_train, y_train), (x_test, y_test)


'''使用Gridcv來尋找最好的參數'''
def find_gpr_model(model, x, y):
    print('尋找最佳參數')
    alpha_list = list(10.0 ** i for i in range(-10,11))
    kernel_list = list(RationalQuadratic(alpha = i/10, length_scale = i/10, length_scale_bounds = (1e-150, 1e150), alpha_bounds = (1e-150, 1e150)) for i in range(1,11))
    param_grid = {'alpha': alpha_list, 'kernel': kernel_list}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x, y)
    best_params = grid_search.best_params_
    print(f'最佳的參數 {best_params}')
    return best_params


'''交叉驗證'''
def cross_evaluate(model, x, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    scores = cross_validate(model, x, y, cv=kf, scoring=['neg_mean_squared_error', 'r2'])
    mse_scores = -scores['test_neg_mean_squared_error']  # 將負的MSE轉換為正數
    r2_scores = scores['test_r2']
    # print(f'mse : {mse_scores}')
    # print(f'r2 : {r2_scores}')
    mean_mse = np.mean(mse_scores)
    mean_r2 = np.mean(r2_scores)
    return mean_mse, mean_r2


'''計算mse, r2'''
def model_evaluate(model, in_data, split_size):
    all, train, test = split_train_test(in_data, split_size)
    model.fit(train[0], train[1])
    y_all_pred = model.predict(all[0])
    all_r2 = r2_score(all[1], y_all_pred)
    all_mse = mean_squared_error(all[1], y_all_pred)
    y_train_pred = model.predict(train[0])
    train_r2 = r2_score(train[1], y_train_pred)
    train_mse = mean_squared_error(train[1], y_train_pred)
    y_test_pred = model.predict(test[0])
    test_r2 = r2_score(test[1], y_test_pred)
    test_mse = mean_squared_error(test[1], y_test_pred)
    return {'train': (train_mse, train_r2), 'test': (test_mse, test_r2), 'all': (all_mse, all_r2)}


'''計算特徵跟預測目標的關係'''
def cal_corr(in_data):
    corr_dict = in_data.corr().apply(lambda x:abs(x)).iloc[0].to_dict()
    corr_list = sorted(corr_dict.items(), key=lambda x:x[1])
    corr_df = pd.DataFrame(columns = ['ASV', 'corr'], data=corr_list)
    corr_df.drop(corr_df.tail(1).index, inplace=True)
    corr_df = corr_df.sort_values(by=['corr'], ascending=False)
    corr_sum = corr_df['corr'].sum()
    corr_df['corr_div'] = corr_df['corr'].cumsum()/corr_sum
    # corr_df.to_csv('./result/train1/feature_select.csv', encoding='utf-8')
    return corr_df


'''計算所有特徵出現0的次數,把次數大於平均值的特徵刪除'''
def data_cleaning_zero_count(data):
    # print(data.columns.shape)
    zero_count  = data[data == 0].count()
    mean = np.mean(zero_count.to_numpy())
    drop_list = zero_count[zero_count > len(data)*0.8]
    new_data = data.drop(drop_list.index.tolist(), axis=1)
    # print(new_data.columns.shape)
    return new_data


'''方差過濾'''
def data_cleaning_variance(data):
    # print(data.columns.shape)
    x = data.iloc[:,1:].to_numpy().astype('float32')
    y = ori_data.iloc[:,:1].to_numpy()
    new_data = pd.DataFrame(y, columns=['TCE_change'])
    variances = np.var(x, axis=0)
    threshold = np.mean(variances)
    # print(variances)
    # print(threshold)
    # exit()
    df = pd.DataFrame(x, columns=data.columns.to_list()[1:])
    selected_features = df.iloc[:, variances > threshold]
    new_data = pd.concat([new_data, selected_features], axis=1)
    # print(new_data.columns.shape)
    return new_data


'''模型建立'''
def GPR(in_data, **kwargs):
    kernel = RationalQuadratic()
    model = GaussianProcessRegressor(kernel=kernel, normalize_y = True, random_state=random_seed)
    train_mse = 0
    train_r2 = 0
    test_mse = 0
    test_r2 = 0
    if len(kwargs) == 2:
        temp = kwargs['params']
        s = kwargs['split_size']
        model.set_params(**temp)
        result = model_evaluate(model, in_data, s)
        test_mse = result['test'][0]
        test_r2 = result['test'][1]
        train_mse = result['train'][0]
        train_r2 = result['train'][1]
    elif len(kwargs) == 1:
        temp = kwargs['params']
        model.set_params(**temp)
    return model, (train_mse, train_r2), (test_mse, test_r2)


'''根據比例篩選特徵'''
def feature_selece_by_pareto(in_data, value):
    result = ['TCE_change']
    result += in_data[in_data['corr_div'] < value]['ASV'].to_list()
    # exit()
    return result


'''計算所有模型的結果，找出最好的'''
def feature_select(params, in_data, feature_percent):
    result = {}
    progress1 = tqdm(total=2)
    progress2 = tqdm(total=len(values_list))
    for s in [0.3, 0.2]:
        all_model = {}
        all_scores = []
        train_scores = []
        for v in values_list:
            progress2.update(1)
            temp_features = feature_selece_by_pareto(feature_percent, v)
            if len(temp_features) == 1:
                print('a')
                all_scores.append((999,-1))
                continue
            temp_data = in_data[temp_features]
            model, train_eval, test_eval = GPR(temp_data, params=params, split_size=s)
            all_model[v] = model
            all_scores.append((test_eval[0], test_eval[1]))
            train_scores.append((train_eval[0], train_eval[1]))
        progress1.update(1)
        progress2.reset()
        mse = [i[0] for i in all_scores]
        r2 = [i[1] for i in all_scores]
        max_value = max(r2)
        min_value = min(mse)
        train_mse_list = [i[0] for i in train_scores]
        train_r2_list = [i[1] for i in train_scores]
        train_r2 = max(train_r2_list)
        train_mse = min(train_mse_list)
        max_index = r2.index(max_value)
        min_index = mse.index(min_value)
        result[s] = (train_mse, train_r2, min_value, max_value, values_list[min_index], values_list[max_index], r2, mse, all_model[values_list[max_index]])
    return result


'''畫出根據比例算出的R2和MSE'''
def draw_plot(corr, r2, measure, title):
    plt.title(title)
    plt.plot(corr, r2)
    plt.xlabel("Pareto percent")
    plt.ylabel(measure)
    plt.tight_layout()
    # plt.savefig(f'./result/{data_name}/GPR/{title}_{measure}_plot.png')
    plt.show()


'''劃出實際值和預測值的位置'''
def draw_scatter(pred, true, title):  
    plt.scatter(recorder, pred, label="Pred samples", c="#8A2BE2")
    plt.scatter(recorder, true, label="True samples", c="#d95f02")
    plt.title(f'{title}  Scatter')
    plt.xlabel("number")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f'./result/{data_name}/GPR/{title}_scatter.png')
    plt.show()


'''資料清洗'''
# ori_data = data_cleaning_variance(ori_data)
# ori_data = data_cleaning_zero_count(ori_data)
# print(ori_data)


'''原始GPR model建立'''
model, _, _ = GPR(ori_data)

'''切分'''
all, train, test = split_train_test(ori_data, 0.2)

# '''使用kfold來判斷方法好不好'''
# mse, r2 = cross_evaluate(model, all[0], all[1])
# print('-----------------------kfold結果-----------------------')
# print(f'cross_val : {mse}, {r2}')

'''rough'''
for e in [0.2, 0.3]:
    model, _, _ = GPR(ori_data)
    result = model_evaluate(model, ori_data, e)
    print(f'{e} {result}')
# exit()    

'''超參數尋找'''
params = find_gpr_model(model, all[0], all[1])
new_model, _, _  = GPR(ori_data, params = params)

'''評估結果'''
result = model_evaluate(new_model, ori_data, 0.3)
print(result)

# exit()
'''特徵選擇'''
correlation = cal_corr(ori_data)
# exit()
# print(correlation)
result = feature_select(params, ori_data, correlation)
result_3 = result[0.3]
result_2 = result[0.2]
model3 = result_3[-1]
model2 = result_2[-1]
model2_data = ori_data[feature_selece_by_pareto(correlation, result_2[4])]
model3_data = ori_data[feature_selece_by_pareto(correlation, result_3[4])]
all2, _, _ = split_train_test(model2_data, 0.2)
all3, _, _ = split_train_test(model3_data, 0.3)
print(model2_data)
print(model3_data)

'''印出結果'''
print(f'8:2的結果 : (mse {result_2[0]} {result_2[2]}) (r2 {result_2[1]} {result_2[3]}) (percent {result_2[4]} {result_2[5]})')
print(f'7:3的結果 : (mse {result_3[0]} {result_3[2]}) (r2 {result_3[1]} {result_3[3]}) (percent {result_3[4]} {result_3[5]})')


'''畫圖'''
draw_plot(values_list, result_3[6], 'R2', '73')
draw_plot(values_list, result_3[7], 'mse', '73')
draw_plot(values_list, result_2[6], 'R2', '82')
draw_plot(values_list, result_2[7], 'mse', '82')
draw_scatter(model2.predict(all2[0]), all2[1], '82')
draw_scatter(model3.predict(all3[0]), all2[1], '73')

'''時間計算'''
end = time.time()
print(f'花費時間 : {end - start}')