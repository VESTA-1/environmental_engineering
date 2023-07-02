import xgboost
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

'''data載入'''
data_name = 'train1'
data_path = f'./data/{data_name}.csv'
ori_data = pd.read_csv(data_path, index_col=0)

'''特徵選取'''
feature_list = ['TCE_change']
feature_sel_path = f'./result/{data_name}/feature_select_p.csv'
feature = pd.read_csv(feature_sel_path, index_col=0)
feature_p = feature.iloc[1].to_dict()

'''閥值跟回合數'''
rand_seed = 42
values_p = list(i * 5 /1000 for i in range(2,201,1))
values_c = list(i / 100 for i in range(40,-50,-1))
recorder = list(i for i in range(1, len(ori_data)+1))

def draw_plot(corr, r2, measure):
    plt.plot(corr, r2)
    plt.xlabel("Pareto percent")
    plt.ylabel(measure)
    plt.tight_layout()
    # plt.savefig(f'./result/train3/feature_select/{measure}.png')
    plt.show()


def data_feature_select_p(in_data ,value):
    feature_dict = {}
    feature_list = ['TCE_change']
    for e in feature_p:
        if feature_p[e] <= value: #將每個feature的柏拉圖的數值都和設定的閥值做比較
            feature_dict[e] = feature_p[e]
    for e in list(feature_dict):
        feature_list.append(e)

    out_data = in_data[feature_list]
    return feature_list, out_data


def draw_scatter(pred, true): 
    plt.scatter(recorder, pred, label="Pred samples", c="#8A2BE2")
    plt.scatter(recorder, true, label="True samples", c="#d95f02")
    plt.title('Scatter')
    plt.xlabel("number")
    plt.ylabel("BDE-209")
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_best_percent(in_data, measure):
    all_model = {}
    all_scores = []
    for v in values_p:
        print(f'目前計算{v}%的特徵')
        temp_feautre, temp_data = data_feature_select_p(in_data, v) #根據計算出特徵
        model, result = decision_tree(temp_data) #將選出的特徵拿去訓練
        if measure == 'R2':
            all_scores.append(result['test'][1]) #將所有資料拿去預測的R2值當作分數
            # all_scores.append(result['train'][1]) #將所有資料拿去預測的R2值當作分數
        else:
            all_scores.append(result['test'][0])
            # all_scores.append(result['train'][0])
        all_model[v] = (model, temp_feautre, temp_data, result) #將訓練好的模型、所用特徵、分數儲存
    if measure == 'R2':
        max_value = max(all_scores) #找到最高分數的
    else:
        max_value = min(all_scores) #找到最高分數的
    max_index = all_scores.index(max_value) #找到最高分數的位置，就可以知道是多少percent了
    return (all_scores, values_p[max_index], all_model[values_p[max_index]])


def decision_tree(in_data):
    '''資料載入'''
    x_data = in_data.iloc[:,1:].to_numpy().astype('float32')
    y_data = in_data.iloc[:,:1].to_numpy()
    # y_data = scaler.fit_transform(y_data)
    y_data = y_data.ravel()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=rand_seed)

    model = DecisionTreeRegressor(max_depth = 3, random_state = 42)
    '''用train去擬合,並預測目標'''
    model.fit(x_train, y_train)
    model_GPR_train_predictions = model.predict(x_train)
    model_GPR_test_predictions = model.predict(x_test)


    '''計算R2和MSE'''
    train_r2 = r2_score(y_train, model_GPR_train_predictions)
    train_mse = mean_squared_error(y_train, model_GPR_train_predictions)
    test_r2 = r2_score(y_test, model_GPR_test_predictions)
    test_mse = mean_squared_error(y_test, model_GPR_test_predictions)

    return model, {'train':(train_mse, train_r2), 'test':(test_mse, test_r2)}


best = get_best_percent(ori_data, 'MSE')
best_value = best[1]
model = best[2][0]
new_feature = best[2][1]
new_data = best[2][2]
new_result = best[2][3]
print(best_value)
print(new_result)
draw_plot(values_p, best[0], 'MSE')
