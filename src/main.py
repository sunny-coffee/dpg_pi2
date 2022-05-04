from configurations import config_DPG_PI2
from elm_model import ELM
from dpg_pi2_trainer import DPGPI2Trainer
import matplotlib.pyplot as plt
import numpy as np
from read_matlab_data import read_matlab_data

demos_path  = "./data/sineShape2"
training_data = read_matlab_data(demos_path, config_DPG_PI2)

initELM = ELM(training_data, config_DPG_PI2)
trainer = DPGPI2Trainer(initELM, config_DPG_PI2)
result = trainer.train()

eval_sample = result.fin_sample
fig = plt.figure()
plt.plot(eval_sample.w[:,0],eval_sample.w[:,1],'b' )
plt.plot(eval_sample.ref_w[:,0], eval_sample.ref_w[:,1],'r')
for i in range(config_DPG_PI2.viapoints.shape[0]):
    plt.scatter(config_DPG_PI2.viapoints[i,0], config_DPG_PI2.viapoints[i,1])
# plt.scatter(config_DPG_PI2.dynamics_model.start[0], config_DPG_PI2.dynamics_model.start[1])
# plt.scatter(config_DPG_PI2.dynamics_model.goal[0], config_DPG_PI2.dynamics_model.goal[1])
plt.show()
