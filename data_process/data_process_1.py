import pandas as pd
import numpy as np


def read_data(data_name):
    data = pd.read_excel(data_name,index_col=0)
    return data

def original_data_process(data):
    temp_dict = {}
    temp_name = []
    temp_result = []
    y = data.loc['總氯化乙烯濃度'][:90] #抓取總氯化以烯的數值
    index = data.loc['OTU ID'][:90] #抓取實驗的ID
    feature = data.loc[list(f'ASV_{i}' for i in range(1, 747)), :90] #抓取所有ASV的數值(從1到746個)，總共90筆資料
    feature = feature.transpose() #轉置
    result = pd.concat([index, y, feature], axis=1) #將x、y、ID合併
    result.to_csv('data/data1.csv', encoding='utf-8_sig', index=False) #儲存檔案
    features = pd.DataFrame()
    for i in range(0, 30):
        name = result.iloc[i][0][:4]
        temp1 = result.iloc[i] #抓取第一個時間點的數據
        temp2 = result.iloc[i+30] #抓取第二個時間點的數據
        temp3 = result.iloc[i+60] #抓取第三個時間點的數據
        '''計算微生物的差異值'''
        feature1 = (temp2[2:].to_frame().set_axis([i*3],axis=1) + 1) / (temp1[2:].to_frame().set_axis([i*3],axis=1) + 1)
        feature2 = (temp3[2:].to_frame().set_axis([i*3+1],axis=1) + 1) / (temp2[2:].to_frame().set_axis([i*3+1],axis=1) + 1)
        feature3 = (temp3[2:].to_frame().set_axis([i*3+2],axis=1) + 1) / (temp1[2:].to_frame().set_axis([i*3+2],axis=1) + 1)
        feature1 = feature1.applymap(lambda x: np.log(x))
        feature2 = feature2.applymap(lambda x: np.log(x))
        feature3 = feature3.applymap(lambda x: np.log(x))
        features = pd.concat([features,feature1, feature2, feature3],axis=1)
        '''計算y的差異值'''
        result1 = temp2[1] - temp1[1]
        result2 = temp3[1] - temp2[1]
        result3 = temp3[1] - temp1[1]
        temp_name.append(name+'_1')
        temp_name.append(name+'_2')
        temp_name.append(name+'_3')
        temp_result.append(result1)
        temp_result.append(result2)
        temp_result.append(result3)
        temp_dict['ID'] = temp_name
        temp_dict['TCE_change'] = temp_result
    train_data = pd.DataFrame(temp_dict)
    train_data = pd.concat([train_data,features.transpose().sort_index()], axis=1)
    train_data.set_index('ID')
    train_data.to_csv('data/train1.csv', encoding='utf-8_sig', index=False)
    return train_data


origin_data = read_data('data/original_data_1.xlsx') #讀取檔案
train_data = original_data_process(origin_data) #進行資料處理
