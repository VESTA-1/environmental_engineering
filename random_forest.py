import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler
import time
from tqdm import tqdm

start = time.time()

'''data載入'''
data_name = 'train1'
data_path = f'./data/{data_name}.csv'
ori_data = pd.read_csv(data_path, index_col=0)
scaler = MinMaxScaler()
x_data = ori_data.iloc[:,1:].to_numpy().astype('float32')
scaler.fit(x_data)
x_data = scaler.transform(x_data)
y_data = ori_data.iloc[:,:1].to_numpy().ravel()

'''全域變數'''
values_list = list(i * 5 /1000 for i in range(2,201,1))
recorder = list(i for i in range(1, len(ori_data)+1))
random_seed = 42
split_size = 0.3

def model_evaluate(model, in_data, split_size):
    all, train, test = split_train_test(in_data, split_size)
    model.fit(train[0], train[1])
    y_all_pred = model.predict(all[0])
    all_r2 = r2_score(all[1], y_all_pred)
    all_mse = mean_squared_error(y_data, y_all_pred)
    y_train_pred = model.predict(train[0])
    train_r2 = r2_score(train[1], y_train_pred)
    train_mse = mean_squared_error(train[1], y_train_pred)
    y_test_pred = model.predict(test[0])
    test_r2 = r2_score(test[1], y_test_pred)
    test_mse = mean_squared_error(test[1], y_test_pred)
    return {'train': (train_mse, train_r2), 'test': (test_mse, test_r2), 'all': (all_mse, all_r2)}


'''randomForest模型'''
def random_forest_model(in_data, seed, **kwargs):
    model = RandomForestRegressor(random_state=seed)
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

'''交叉驗證'''
def cross_evaluate(model, x, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, x, y, cv=kf, scoring=['neg_mean_squared_error', 'r2'])
    mse_scores = -scores['test_neg_mean_squared_error']  # 將負的MSE轉換為正數
    r2_scores = scores['test_r2']
    print(f'mse : {mse_scores}')
    print(f'r2 : {r2_scores}')
    mean_mse = np.mean(mse_scores)
    mean_r2 = np.mean(r2_scores)
    return mean_mse, mean_r2


'''使用Gridcv來尋找最好的參數'''
def find_best_model(model, x, y):
    print('尋找最佳參數')
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4, 6]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x, y)
    best_params = grid_search.best_params_
    print(f'最佳的參數 {best_params}')
    return best_params


'''分割資料集'''
def split_train_test(data, size):
    scaler = MinMaxScaler()
    x_data = data.iloc[:,1:].to_numpy().astype('float32')
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)
    y_data = data.iloc[:,:1].to_numpy()
    y_data = y_data.ravel()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=size, random_state=random_seed)
    return (x_data, y_data), (x_train, y_train), (x_test, y_test)


'''計算每個特徵的重要性'''
# def feature_select_importance(feature_count):
#     temp_feature = [item[0]for item in feature_scores[:feature_count]]
#     print(temp_feature)
#     exit()
#     return data


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
                all_scores.append((999,-1))
                continue
            temp_data = in_data[temp_features]
            model, train_eval, test_eval = random_forest_model(temp_data, seed=random_seed, params=params, split_size=s)
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


def feature_selece_by_pareto(in_data, value):
    result = ['TCE_change']
    result += in_data[in_data['corr_div'] < value]['ASV'].to_list()
    return result


'''畫出根據比例算出的R2和MSE'''
def draw_plot(corr, r2, measure, title):
    plt.title(title)
    plt.plot(corr, r2)
    plt.xlabel("Pareto percent")
    plt.ylabel(measure)
    plt.tight_layout()
    # plt.savefig(f'./result/{data_name}/RF/{title}_{measure}.png')
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
    # plt.savefig(f'./result/{data_name}/RF/{title}_scatter.png')
    plt.show()


'''建立模型'''
model, _, _ = random_forest_model(ori_data, random_seed)

'''使用kfold來判斷方法好不好'''
# mse, r2 = cross_evaluate(model, x_data, y_data)
# print(f'cross_val : {mse}, {r2}')

'''超參數尋找'''
params = find_best_model(model, x_data, y_data)
model, _, _ = random_forest_model(ori_data, random_seed, params = params)

'''評估結果'''
# result = model_evaluate(model, ori_data, split_size)
# print(result)

'''特徵選擇'''
correlation = cal_corr(ori_data)
result = feature_select(params, ori_data, correlation)
result_3 = result[0.3]
result_2 = result[0.2]
model3 = result_3[-1]
model2 = result_2[-1]
model2_data = ori_data[feature_selece_by_pareto(correlation, result_2[4])]
model3_data = ori_data[feature_selece_by_pareto(correlation, result_3[4])]
all2, _, _ = split_train_test(model2_data, split_size)
all3, _, _ = split_train_test(model3_data, split_size)

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
