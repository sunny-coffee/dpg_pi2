import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt 
def read_matlab_data(data_path, config):

    data = scio.loadmat(data_path)
    demos = data.get('demos')
    demo_list = list(demo.T for demo in demos.ravel())
    # plt.figure()
    # for demo in demo_list:
    #     plt.plot(demo[:,0],demo[:,1])
    # plt.show()
    for i,demo in enumerate(demo_list):
        vel = np.diff(demo, axis=0)/config.dt
        vel = np.insert(vel, 0, np.zeros(config.n_dim), axis=0)
        demo_list[i] = np.concatenate((demo,vel), axis=1)
    training_data = np.concatenate(demo_list, axis=0)
    control_pram_array = config.kp0 * np.ones((training_data.shape[0],config.n_dim_kp))
    training_data = np.concatenate((training_data, control_pram_array), axis=1)
    return training_data


if __name__ == "__main__":
    data_path = "./data/sineShape2"
    # print(read_matlab_data(data_path))
    # print(type(read_matlab_data(data_path)[0,0]))
    # print((read_matlab_data(data_path)[0,0]).shape)
    print(type(read_matlab_data(data_path)), read_matlab_data(data_path).shape)