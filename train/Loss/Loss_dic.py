import numpy as np
import os
import train.miscc.Data_IO as dio


def Loss_Dic_maker(Loss_dic, add_dic):
    keys_list = list(add_dic.keys())
    for keyy in keys_list:
        if type(add_dic[keyy]) != type([]):
            continue
        try:
            Loss_dic[keyy] += add_dic[keyy]
        except Exception as e:
            Loss_dic[keyy] = add_dic[keyy]
    return Loss_dic



def Loss_Dic_mean(Loss_dic):
    for key in list(Loss_dic.keys()):
        Loss_dic[key] = np.mean(np.array(Loss_dic[key]))
    return Loss_dic


def Loss_Dic_mean_show(Loss_dic_mean, dic_folder_path='F'):
    show_str = ''
    for i, key in enumerate(list(Loss_dic_mean.keys())):
        show_str += str(key) + ':' + str(np.round(Loss_dic_mean[key], 2)) + '  '
        if i % 5 == 4:
            print(show_str)
            show_str = ''
    print(show_str)

    if dic_folder_path != 'F':
        dic_list_path = os.path.join(dic_folder_path, 'loss_dict.pickle')
        try:
            dic_list = dio.load_pickle(dic_list_path)
        except Exception as e:
            dic_list = []
        Loss_dic_mean_save = Loss_dic_mean.copy()
        dic_list.append(Loss_dic_mean_save)
        dio.save_pickle(dic_list, dic_list_path, False)