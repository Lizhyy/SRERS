import numpy as np
from Config import config
cfg = config()



def Polyhedron_4_microphone_loc():
    """
    正四面体麦克风阵列，小金字塔，4麦克风
    :return:
    """
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])




def x_batch_switch(x_batch, switch_type):
    if cfg.INIT.IO_mesh_in_vertex_ebd == 'face':
        return x_batch_switch2(x_batch, switch_type)
    else:
        return x_batch_switch1(x_batch, switch_type)




def x_batch_switch1(x_batch, switch_type):
    if switch_type == 0:
        x_batch[:, [0, 1, 2]] = x_batch[:, [0, 1, 2]]
    elif switch_type == 1:
        x_batch[:, [0, 1, 2]] = x_batch[:, [0, 2, 1]]
    elif switch_type == 2:
        x_batch[:, [0, 1, 2]] = x_batch[:, [1, 0, 2]]
    elif switch_type == 3:
        x_batch[:, [0, 1, 2]] = x_batch[:, [1, 2, 0]]
    elif switch_type == 4:
        x_batch[:, [0, 1, 2]] = x_batch[:, [2, 0, 1]]
    elif switch_type == 5:
        x_batch[:, [0, 1, 2]] = x_batch[:, [2, 1, 0]]
    return x_batch




def test_switch(x_batch, switch_type):
    if switch_type == 0:
        x_batch[[0, 1, 2]] = x_batch[[0, 1, 2]]
    elif switch_type == 1:
        x_batch[[0, 1, 2]] = x_batch[[0, 2, 1]]
    elif switch_type == 2:
        x_batch[[0, 1, 2]] = x_batch[[1, 0, 2]]
    elif switch_type == 3:
        x_batch[[0, 1, 2]] = x_batch[[1, 2, 0]]
    elif switch_type == 4:
        x_batch[[0, 1, 2]] = x_batch[[2, 0, 1]]
    elif switch_type == 5:
        x_batch[[0, 1, 2]] = x_batch[[2, 1, 0]]
    return x_batch




def ebd_batch_switch(ebd_batch, switch_type):
    if switch_type == 0:
        ebd_batch[:, [0, 1, 2]] = ebd_batch[:, [0, 1, 2]]
    elif switch_type == 1:
        ebd_batch[:, [0, 1, 2]] = ebd_batch[:, [0, 2, 1]]
    elif switch_type == 2:
        ebd_batch[:, [0, 1, 2]] = ebd_batch[:, [1, 0, 2]]
    elif switch_type == 3:
        ebd_batch[:, [0, 1, 2]] = ebd_batch[:, [1, 2, 0]]
    elif switch_type == 4:
        ebd_batch[:, [0, 1, 2]] = ebd_batch[:, [2, 0, 1]]
    elif switch_type == 5:
        ebd_batch[:, [0, 1, 2]] = ebd_batch[:, [2, 1, 0]]
    return ebd_batch





def ebd_switch(ebd_list, switch_type):
    if switch_type == 0:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[0, 1, 2, 3, 4, 5]]
    elif switch_type == 1:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[0, 2, 1, 3, 5, 4]]
    elif switch_type == 2:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[1, 0, 2, 4, 3, 5]]
    elif switch_type == 3:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[1, 2, 0, 4, 5, 3]]
    elif switch_type == 4:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[2, 0, 1, 5, 3, 4]]
    elif switch_type == 5:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[2, 1, 0, 5, 4, 3]]
    return ebd_list




def x_batch_switch2(ebd_list, switch_type):
    if switch_type == 0:
        ebd_list[:, [4,5,6,7,8,9,10,11,12,13,14,15]] = ebd_list[:, [4,5,6,7,8,9,10,11,12,13,14,15]]
    elif switch_type == 1:
        ebd_list[:, [4,5,6,7,8,9,10,11,12,13,14,15]] = ebd_list[:, [4,6,5,7,9,8,10,12,11,13,15,14]]
    elif switch_type == 2:
        ebd_list[:, [4,5,6,7,8,9,10,11,12,13,14,15]] = ebd_list[:, [5,4,6,8,7,9,11,10,12,14,13,15]]
    elif switch_type == 3:
        ebd_list[:, [4,5,6,7,8,9,10,11,12,13,14,15]] = ebd_list[:, [5,6,4,8,9,7,11,12,10,14,15,13]]
    elif switch_type == 4:
        ebd_list[:, [4,5,6,7,8,9,10,11,12,13,14,15]] = ebd_list[:, [6,4,5,9,7,8,12,10,11,15,13,14]]
    elif switch_type == 5:
        ebd_list[:, [4,5,6,7,8,9,10,11,12,13,14,15]] = ebd_list[:, [6,5,4,9,8,7,12,11,10,15,14,13]]
    return ebd_list




def RIR_batch_switch(RIR_batch, switch_type):
    # if switch_type == 0:
    #     RIR_batch = np.array(RIR_batch[[0, 1, 2, 3], :])
    # elif switch_type == 1:
    #     RIR_batch = np.array(RIR_batch[[0, 1, 3, 2], :])
    # elif switch_type == 2:
    #     RIR_batch = np.array(RIR_batch[[0, 2, 1, 3], :])
    # elif switch_type == 3:
    #     RIR_batch = np.array(RIR_batch[[0, 2, 3, 1], :])
    # elif switch_type == 4:
    #     RIR_batch = np.array(RIR_batch[[0, 3, 1, 2], :])
    # elif switch_type == 5:
    #     RIR_batch = np.array(RIR_batch[[0, 3, 2, 1], :])

    if switch_type == 0:
        RIR_batch[:, [0, 1, 2, 3], :] = RIR_batch[:, [0, 1, 2, 3], :]
    elif switch_type == 1:
        RIR_batch[:, [0, 1, 2, 3], :] = RIR_batch[:, [0, 1, 3, 2], :]
    elif switch_type == 2:
        RIR_batch[:, [0, 1, 2, 3], :] = RIR_batch[:, [0, 2, 1, 3], :]
    elif switch_type == 3:
        RIR_batch[:, [0, 1, 2, 3], :] = RIR_batch[:, [0, 2, 3, 1], :]
    elif switch_type == 4:
        RIR_batch[:, [0, 1, 2, 3], :] = RIR_batch[:, [0, 3, 1, 2], :]
    elif switch_type == 5:
        RIR_batch[:, [0, 1, 2, 3], :] = RIR_batch[:, [0, 3, 2, 1], :]
    return RIR_batch



def ev_batch_switch(ev_batch, switch_type):
    if switch_type == 0:
        ev_batch[:, [0, 1, 2, 3], :, :] = ev_batch[:, [0, 1, 2, 3], :, :]
    elif switch_type == 1:
        ev_batch[:, [0, 1, 2, 3], :, :] = ev_batch[:, [0, 1, 3, 2], :, :]
    elif switch_type == 2:
        ev_batch[:, [0, 1, 2, 3], :, :] = ev_batch[:, [0, 2, 1, 3], :, :]
    elif switch_type == 3:
        ev_batch[:, [0, 1, 2, 3], :, :] = ev_batch[:, [0, 2, 3, 1], :, :]
    elif switch_type == 4:
        ev_batch[:, [0, 1, 2, 3], :, :] = ev_batch[:, [0, 3, 1, 2], :, :]
    elif switch_type == 5:
        ev_batch[:, [0, 1, 2, 3], :, :] = ev_batch[:, [0, 3, 2, 1], :, :]
    return ev_batch



if __name__ == "__main__":
    mic_geo_0 = Polyhedron_4_microphone_loc()
    # mic_loc_0_0 = Polyhedron_4_microphone_loc()
    pos0 = np.random.randn(3) * 100
    mic_loc_0 = mic_geo_0 + pos0
    # h_0 = mic_loc_0 @ mic_loc_0.T + 1

    mic_geo_1 = x_batch_switch1(np.array(mic_geo_0), 1)
    pos1 = test_switch(np.array(pos0), 1)
    mic_loc_1 = x_batch_switch1(np.array(mic_loc_0), 1)
    mic_loc_1 = RIR_batch_switch(np.array(mic_loc_1), 1)
    mic_geo_1_ = (mic_loc_1 - pos1).astype('int32')
    
    mic_geo_2 = x_batch_switch1(np.array(mic_geo_0), 2)
    pos2 = test_switch(np.array(pos0), 2)
    mic_loc_2 = x_batch_switch1(np.array(mic_loc_0), 2)
    mic_loc_2 = RIR_batch_switch(np.array(mic_loc_2), 2)
    mic_geo_2_ = (mic_loc_2 - pos2).astype('int32')
    
    mic_geo_3 = x_batch_switch1(np.array(mic_geo_0), 3)
    pos3 = test_switch(np.array(pos0), 3)
    mic_loc_3 = x_batch_switch1(np.array(mic_loc_0), 3)
    mic_loc_3 = RIR_batch_switch(np.array(mic_loc_3), 3)
    mic_geo_3_ = (mic_loc_3 - pos3).astype('int32')
    
    mic_geo_4 = x_batch_switch1(np.array(mic_geo_0), 4)
    pos4 = test_switch(np.array(pos0), 4)
    mic_loc_4 = x_batch_switch1(np.array(mic_loc_0), 4)
    mic_loc_4 = RIR_batch_switch(np.array(mic_loc_4), 4)
    mic_geo_4_ = (mic_loc_4 - pos4).astype('int32')
    
    mic_geo_5 = x_batch_switch1(np.array(mic_geo_0), 5)
    pos5 = test_switch(np.array(pos0), 5)
    mic_loc_5 = x_batch_switch1(np.array(mic_loc_0), 5)
    mic_loc_5 = RIR_batch_switch(np.array(mic_loc_5), 5)
    mic_geo_5_ = (mic_loc_5 - pos5).astype('int32')

    # mic_loc_6 = mic_loc_0
