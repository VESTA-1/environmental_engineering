import pandas as pd
import numpy as np


def read_data(data_name):
    data = pd.read_excel(data_name,index_col=0)
    return data

def original_data_process(data):
    index = data.loc['OTU ID']
    temp_dict = {}
    temp_name = []
    temp_result = []
    y = data.loc['總BDE-209濃度(µmol)'][:66]
    # y.reset_index()
    feature = data.loc[list('ASV_{}'.format(i) for i in range(1, 3163))]
    feature = feature.transpose()
    feature = feature[:66]
    result = pd.concat([y, feature], axis=1)
    result.insert(0, column='ID', value=index)
    result.to_csv('data/data4.csv', encoding='utf-8_sig', index=False)
    # result = pd.read_csv('./data/data3.csv')
    # exit()
    features = pd.DataFrame()
    for i in range(0, 33):
        name = result.iloc[i][0]
        # print(name)
        # exit()
        temp1 = result.iloc[i]
        temp2 = result.iloc[i+33]
        # print(temp1, temp2)
        # print(temp2[2:].to_frame().set_axis([i],axis=1))
        # print(temp2[:].to_frame().set_axis([i],axis=1), temp1[:].to_frame().set_axis([i],axis=1))
        # feature1 = temp2[2:].to_frame().set_axis([i*3],axis=1) - temp1[2:].to_frame().set_axis([i*3],axis=1)
        # feature2 = temp3[2:].to_frame().set_axis([i*3+1],axis=1) - temp2[2:].to_frame().set_axis([i*3+1],axis=1)
        # feature3 = temp3[2:].to_frame().set_axis([i*3+2],axis=1) - temp1[2:].to_frame().set_axis([i*3+2],axis=1)

        # feature1 = temp1[2:].to_frame().set_axis([i*3],axis=1)
        # feature2 = temp2[2:].to_frame().set_axis([i*3+1],axis=1)
        # feature3 = temp3[2:].to_frame().set_axis([i*3+2],axis=1)

        feature1 = (temp2[2:].to_frame().set_axis([i],axis=1) + 1) / (temp1[2:].to_frame().set_axis([i],axis=1) + 1)
        feature1 = feature1.applymap(lambda x: np.log(x))
        features = pd.concat([features,feature1],axis=1)
        
        # print(features.transpose())
        # temp1 = float(result.iloc[i][1])
        # temp2 = float(result.iloc[i+30][1])
        # temp3 = float(result.iloc[i+60][1])
        result1 = temp2[1] - temp1[1]
        temp_name.append(name)
        temp_result.append(result1)
        temp_dict['ID'] = temp_name
        temp_dict['TCE_change'] = temp_result
    # exit()
    train_data = pd.DataFrame(temp_dict)
    # print(features.transpose())
    # print(train_data)
    train_data = pd.concat([train_data,features.transpose().sort_index()], axis=1)
    train_data.set_index('ID')
    train_data.to_csv('data/train4.csv', encoding='utf-8_sig', index=False)
    return train_data


def data_cleaning(data):
    zero_count  = data[data == 0].count()
    drop_list = zero_count[zero_count > 30]
    new_data = data.drop(drop_list.index.tolist(), axis=1)
    print(new_data)
    zero_count = new_data[new_data == 0].count()
    # new_data.to_csv('data/data_cleaning_2.csv', encoding='utf-8_sig')

origin_data = read_data('data/original_data_4.xlsx')
# print(origin_data)
train_data = original_data_process(origin_data)
print(train_data)
# clean_data = data_cleaning(train_data)