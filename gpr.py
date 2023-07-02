from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process.kernels import RationalQuadratic
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# warnings.filterwarnings('ignore')
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
# temp_rand = random.sample(range(1, 746), 7)
rand_seed = 42
values_p = list(i * 5 /1000 for i in range(2,201,1))
values_c = list(i / 100 for i in range(40,-50,-1))
recorder = list(i for i in range(1, len(ori_data)+1))
scaler = MinMaxScaler()


def data_cleaning(data):
    zero_count  = data[data == 0].count()
    mean = np.mean(zero_count.to_numpy())
    drop_list = zero_count[zero_count > mean]
    new_data = data.drop(drop_list.index.tolist(), axis=1)
    zero_count = new_data[new_data == 0].count()
    return new_data


'''切割訓練跟測試'''
split_size = 0.3
ori_data = data_cleaning(ori_data)
x_data = ori_data.iloc[:,1:].to_numpy().astype('float32')
y_data = ori_data.iloc[:,:1].to_numpy()
y_data = y_data.ravel()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=split_size, random_state=rand_seed)

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


def get_best_percent(in_data, measure):
    all_model = {}
    all_scores = []
    for v in values_p:
        print(f'目前計算{v}%的特徵')
        temp_feautre, temp_data = data_feature_select_p(in_data, v) #根據計算出特徵
        model, result = GPR(temp_data) #將選出的特徵拿去訓練
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
    

def GPR(in_data):
    '''資料載入'''
    x_data = in_data.iloc[:,1:].to_numpy().astype('float32')
    y_data = in_data.iloc[:,:1].to_numpy()
    # y_data = scaler.fit_transform(y_data)
    y_data = y_data.ravel()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=rand_seed)
    # if not find_best:
    #     print(f'x_data {x_data}')
    #     print(f'y_data {y_data}')

    # '''第一筆資料這些超參數是用交叉驗證找出來的'''
    # kernel = RationalQuadratic(alpha = 1, length_scale = 1, length_scale_bounds = (1e-50, 1e100), alpha_bounds = (1e-5, 1e100))
    # model_GPR = GaussianProcessRegressor(kernel = kernel, alpha = 0.1, normalize_y = True, random_state=rand_seed)
    # model_GPR_kf = GaussianProcessRegressor(kernel = kernel, alpha = 0.1, normalize_y = True, random_state=rand_seed)

    '''第二筆資料這些超參數是用交叉驗證找出來的'''
    # kernel = RationalQuadratic(alpha = 0.3, length_scale=0.3, length_scale_bounds = (1e-150, 1e100), alpha_bounds = (1e-5, 1e100))
    # model_GPR = GaussianProcessRegressor(kernel = kernel, alpha = 0.5, normalize_y = True, random_state=rand_seed)
    # param_grid = {'alpha' : [0.1], 'kernel': [RationalQuadratic(alpha = 1, length_scale = 1, length_scale_bounds = (1e-50, 1e100), alpha_bounds = (1e-5, 1e100))]}
    # grid_search = GridSearchCV(model_GPR, param_grid, cv=5, scoring='neg_mean_squared_error', error_score='raise')
    # grid_search.fit(x_train, y_train)
    # model_GPR = grid_search.best_estimator_

    '''第三筆資料這些超參數是用交叉驗證找出來的'''
    # kernel = RationalQuadratic(alpha = 0.3, length_scale = 0.3, length_scale_bounds = (1e-50, 1e125), alpha_bounds = (1e-5, 1e125))
    # model_GPR = GaussianProcessRegressor(kernel = kernel, alpha = 1, normalize_y = True, random_state=rand_seed)
    # model_GPR_kf = GaussianProcessRegressor(kernel = kernel, alpha = 1, normalize_y = True, random_state=rand_seed)

    '''沒有使用交叉驗證去找超參數'''
    kernel = RationalQuadratic()
    model_GPR = GaussianProcessRegressor(kernel = kernel, random_state=rand_seed)
    model_GPR_kf = GaussianProcessRegressor(kernel = kernel, random_state=rand_seed)


    '''用train去擬合,並預測目標'''
    model_GPR.fit(x_train, y_train)
    model_GPR_train_predictions = model_GPR.predict(x_train)
    model_GPR_test_predictions = model_GPR.predict(x_test)


    '''計算R2和MSE'''
    train_r2 = r2_score(y_train, model_GPR_train_predictions)
    train_mse = mean_squared_error(y_train, model_GPR_train_predictions)
    test_r2 = r2_score(y_test, model_GPR_test_predictions)
    test_mse = mean_squared_error(y_test, model_GPR_test_predictions)


    '''交叉驗證(Cross validation)'''
    # k = 5
    # kf = KFold(n_splits=k, shuffle = True, random_state=rand_seed)
    # model_GPR_kf.fit(x_train, y_train)
    # scores = cross_validate(model_GPR_kf, x_train, y_train, cv=kf, scoring=['neg_mean_squared_error', 'r2'])
    # mse_scores = -scores['test_neg_mean_squared_error']  # 將負的MSE轉換為正數
    # r2_scores = scores['test_r2']
    # model_GPR_test_predictions = model_GPR.predict(x_test)
    # mean_mse = np.mean(mse_scores)
    # mean_r2 = np.mean(r2_scores)
    # test_r2 = r2_score(y_test, model_GPR_test_predictions)
    # test_mse = mean_squared_error(y_test, model_GPR_test_predictions)
    # print(f"Mean MSE: {mean_mse}")
    # print(f"Mean R2: {mean_r2}")

    # return model_GPR_kf,  {'train':(mean_mse, mean_r2), 'test':(test_mse, test_r2)}
    return model_GPR, {'train':(train_mse, train_r2), 'test':(test_mse, test_r2)}


def draw_plot(corr, r2, measure):
    plt.plot(corr, r2)
    plt.xlabel("Pareto percent")
    plt.ylabel(measure)
    plt.tight_layout()
    # plt.savefig(f'./result/train3/feature_select/{measure}.png')
    plt.show()


def draw_scatter(pred, true): 
    plt.scatter(recorder, pred, label="Pred samples", c="#8A2BE2")
    plt.scatter(recorder, true, label="True samples", c="#d95f02")
    plt.title('Scatter')
    plt.xlabel("number")
    plt.ylabel("BDE-209")
    plt.legend()
    plt.tight_layout()
    plt.show()


'''根據特徵出現次數做資料清理'''
def data_cleaning(in_data):
    x_data = in_data.iloc[:,1:]
    zero_count  = x_data[x_data != 0].count() #計算所有特徵非0的次數
    print(zero_count)
    drop_list = zero_count[zero_count < 30] #當特徵出現次數少於10，將準備丟掉
    new_data = x_data.drop(drop_list.index.tolist(), axis=1)
    print(new_data)
    zero_count = new_data[new_data == 0].count()
    

'''計算特徵跟預測目標的關係'''
def cal_corr(in_data):
    corr_dict = in_data.corr().apply(lambda x:abs(x)).iloc[0].to_dict()
    corr_list = sorted(corr_dict.items(), key=lambda x:x[1])
    corr_csv = pd.DataFrame(columns = ['ASV', 'corr'], data=corr_list)
    corr_csv.drop(corr_csv.tail(1).index, inplace=True)
    # corr_csv.to_csv('./result/train1/feature_select.csv', encoding='utf-8')
    print(corr_csv)
    # return corr_csv


'''使用柏拉圖的比例做特徵選取'''
# best = get_best_percent(ori_data, 'R2')
# best_value = best[1]
# model = best[2][0]
# new_feature = best[2][1]
# new_data = best[2][2]
# new_result = best[2][3]
# print(best_value)
# print(new_result)
# draw_plot(values_p, best[0], 'R2')
# new_data.to_csv('./result/train1_3/feature_select/feature_select_01_result.csv')


'''原始資料跑GPR模型'''
# init_model, result = GPR(ori_data)
# print(result)
# x_data = ori_data.iloc[:,1:].to_numpy().astype('float32')
# y_data = ori_data.iloc[:,:1].to_numpy().ravel()
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=rand_seed)
# draw_scatter(init_model.predict(x_data), y_data)


'''看特徵跟預測目標的相關係數'''
# cal_corr(ori_data)


'''用第一筆資料訓練好的模型來測試第二筆資料'''
# second_data = pd.read_csv('./data/train_2_3.csv')
# for i, e in enumerate(new_feature):
#     if i != 0:
#         new_feature[i] = str(e).replace("_","")
#         if len(new_feature[i]) == 5:
#             new_feature[i] = 'ASV0' + new_feature[i][-2:]
#         elif len(new_feature[i]) == 5:
#             new_feature[i] = 'ASV0' + new_feature[i][-2:]
# second_data = second_data[new_feature]
# x_data = second_data.iloc[:,1:].to_numpy().astype('float32')
# y_data = second_data.iloc[:,:1].to_numpy().ravel()
# model_GPR_predictions = model.predict(x_data)
# all_r2 = r2_score(y_data, model_GPR_predictions)
# all_mse = mean_squared_error(y_data, model_GPR_predictions)
# print(all_mse, all_r2)


'''使用Gridcv來尋找最好的參數'''
def find_gpr_params(x, y):
    '''尋找參數'''
    model_GPR = GaussianProcessRegressor(normalize_y = True, random_state=rand_seed)
    alpha_list = list(i/10 for i in range(1,11))
    kernel_list = list(RationalQuadratic(alpha = i/10, length_scale = i/10, length_scale_bounds = (1e-100, 1e100), alpha_bounds = (1e-100, 1e100)) for i in range(1,11))
    param_grid = {'alpha': alpha_list, 'kernel': kernel_list}
    grid_search = GridSearchCV(model_GPR, param_grid, cv=5, scoring='neg_mean_squared_error', error_score='raise')
    grid_search.fit(x, y)
    best_params = grid_search.best_params_
    return best_params


def evaluate(model, x, y):
    '''交叉驗證'''
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, x, y, cv=kf, scoring=['neg_mean_squared_error', 'r2'])
    mse_scores = -scores['test_neg_mean_squared_error']  # 將負的MSE轉換為正數
    r2_scores = scores['test_r2']
    mean_mse = np.mean(mse_scores)
    mean_r2 = np.mean(r2_scores)
    return mean_mse, mean_r2


besst_params = find_gpr_params(x_train, y_train)
model = GaussianProcessRegressor()
model.set_params(**besst_params, random_state=rand_seed)
mse, r2 = evaluate(model, x_train, y_train)
print(mse, r2)
y_pred = model.predict(x_test)
test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
print(test_mse, test_r2)