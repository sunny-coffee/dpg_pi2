import numpy as np

class Masspoint():
    def __init__(self, mass, damp, config, start, goal, policy):
        self.mass = mass
        self.damp = damp
        self.dt = config.dt
        self.start = start
        self.goal = goal
        self.n_dim = config.n_dim
        self.policy = policy
        self.n_steps = int(config.duration/config.dt)
        self.n_param = policy.theta0.shape[0]
        self.stiffness_learning = config.stiffness_learning
        self.constraints = config.constraints
        self.exsit_force_field = config.exsit_force_field
        self.n_dim_kp = config.n_dim_kp
        self.viapoints = config.viapoints
        self.constraints = config.constraints
        

    def run_all_steps(self, eps):
     
        w_epoch = np.zeros((self.n_steps, self.n_dim))
        wd_epoch = np.zeros((self.n_steps, self.n_dim))
        wdd_epoch = np.zeros((self.n_steps, self.n_dim))
        ref_w_epoch = np.zeros((self.n_steps, self.n_dim))
        ref_wd_epoch = np.zeros((self.n_steps, self.n_dim))
        control_parameter_epoch = np.zeros((self.n_steps, self.n_dim_kp))
        
        w = self.start 
        wd = np.zeros(self.n_dim) 
        ref_w = w
        # print(self.policy.theta)
        theta_eps = self.policy.theta + eps
        
        for n in range(self.n_steps):
            outputs = self.policy.predict(w, ref_w, theta_eps)
            # print(outputs) 
            ref_wd = np.squeeze(outputs[0:self.n_dim])
          
            isConstrained = 0 
            for i in range(len(self.constraints)):
                isConstrained = isConstrained or (self.constrains[i].min_w1<w[0] and w[0]<self.constrains[i].max_w1 and 
                                                self.constrains[i].min_w2<w[1] and w[1] < self.constrains[i].max_w2) 

            if isConstrained:
                ref_wd = ref_wd*0 
                wd = wd*0 

            ref_w = ref_w + ref_wd * self.dt 
            
            # if np.linalg.norm(ref_wd) > 1000 or np.linalg.norm(ref_w) > 1000: 
            #     if self.isEval:
            #         raise Exception("Evaluation过程中产生异常值,请重试!!")
            #     break
            
            ref_w_epoch[n, :] = ref_w 
            ref_wd_epoch[n, :] = ref_wd 
            
            if self.stiffness_learning == 1:
                
                kp =  outputs[self.n_dim:]
                kp = np.maximum(kp, 0) 
                kd = 2*np.sqrt(kp) 

                if self.exsit_force_field:
                    force = self.compute_field_force(w,10) 
                else:
                    force = 0 

                [w,wd,wdd,command] = self.run_one_step(w,wd,ref_w,ref_wd,kp,kd,force)    

                # if np.linalg.norm(wd) > 1000 or np.linalg.norm(w) > 1000: 
                #     if self.isEval:
                #         raise Exception("Evaluation过程中产生异常值,请重试!!")
                #     break
                
                w_epoch[n, :] = w 
                wd_epoch[n, :] = wd 
                wdd_epoch[n, :] = wdd 
                control_parameter_epoch[n, :] = kp 
            else:
                w_epoch[n, :] = ref_w 
                wd_epoch[n, :] = ref_wd 
                if n != 0:
                    wdd_epoch[n, :] = (wd[n, :] - wd[n-1, :])/self.dt 

        totCost, transCost, viapointCost, accelerationCost, stiffnessCost = self.compute_cost(w_epoch, wd_epoch, wdd_epoch, control_parameter_epoch)
        sample = Sample(w_epoch, wd_epoch, wdd_epoch, ref_w_epoch, ref_wd_epoch, control_parameter_epoch, eps ,theta_eps, totCost, transCost, viapointCost, accelerationCost, stiffnessCost)
        return sample
    

    def run_one_step(self, w, wd, ref_w, ref_wd, kp, kd, fieldforce):
        command = kp * (ref_w-w) + kd * (ref_wd-wd)
        wdd = (command - wd * self.damp + fieldforce)/self.mass
        wd = wdd * self.dt + wd
        w = wd * self.dt + w
        return w, wd, wdd, command


    def compute_field_force(self, position, gain):
        nPoints = 100
        forceFieldTraj = np.zeros((nPoints, 2))
        for cnt1 in range(nPoints):
            forceFieldTraj[cnt1, 1] = -10* (1 - cnt1/nPoints)
            forceFieldTraj[cnt1, 2] = np.sin(2*np.pi*(1 - cnt1/nPoints))

        distances = np.linalg.norm(position - forceFieldTraj, axis=1)
        minDist = np.min(distances)
        minIndex = np.where(distances == minDist)

        if minIndex != nPoints-1:
            tang = forceFieldTraj[minIndex+1, :] - forceFieldTraj[minIndex, :]
        else:
            tang = forceFieldTraj[minIndex, :] - forceFieldTraj[minIndex-1, :]
            
        force = np.dot(tang/np.linalg.norm(tang), gain * minDist* np.array([[0, 1], [-1, 0]]))
        if np.dot(force, (position - forceFieldTraj[minIndex, :])) < 0:
            force = -force

        return force

    def compute_cost(self, w, wd, wdd, control_parameter_epoch):
        # print(w.shape)
        # print(w[20,:])
        transCostSteps = np.zeros(self.n_steps)
        accelerationCostSteps = np.zeros(self.n_steps)
        stiffnessCostSteps = np.zeros(self.n_steps)
        viapointCost = 0

        for n in range(self.n_steps):
            if n != 0:
                transCostSteps[n] = np.linalg.norm(w[n, :] - w[n-1, :])

            accelerationCostSteps[n]  =  np.linalg.norm(wdd[n, :])
            stiffnessCostSteps[n] = np.linalg.norm(control_parameter_epoch[n, :])

        for i in range(self.viapoints.shape[0]):
            distances = np.linalg.norm(w - self.viapoints[i, :], axis=1)
            minDist = np.min(distances)
            viapointCost += minDist
        d_final  =  np.linalg.norm(w[-1, :] - self.goal)
        viapointCost += d_final

        viapointCostWeight = 1
        stiffnessCostWeight = 0.00001
        accelerationCostWeight = 0.0012
        transCostWeight = 0.001
        viapointCost = viapointCost * viapointCostWeight
        stiffnessCost = np.sum(stiffnessCostSteps) * stiffnessCostWeight
        accelerationCost = np.sum(accelerationCostSteps) * accelerationCostWeight
        transCost = np.sum(transCostSteps) * transCostWeight
        totCost = viapointCost + accelerationCost + stiffnessCost + transCost
        return totCost, transCost, viapointCost, accelerationCost, stiffnessCost

class Sample():
    def __init__(self, w, wd, wdd, ref_w, ref_wd, control_paramters, eps ,theta_eps, totCost, transCost, viapointCost, accelerationCost, stiffnessCost) -> None:
        self.w = w          # point mass pos
        self.wd = wd        # point mass vel
        self.wdd = wdd      # zeros(n_dim, n_steps);% point mass acc
        self.ref_w = ref_w
        self.ref_wd = ref_wd
        self.control_paramters = control_paramters
        self.eps = eps
        self.theta_eps = theta_eps
        self.totCost = totCost
        self.transCost = transCost
        self.viapointCost = viapointCost
        self.accelerationCost = accelerationCost
        self.stiffnessCost = stiffnessCost


class Masspoint_one_step():
    def __init__(self, mass, damp) -> None:
        self.mass = mass
        self.damp =damp
        self.dt = 0.1
        
    def run_one_step(self, w, wd, action, fieldforce):
        # print(kp,ref_w,kd,ref_wd)
        # print(kp.shape,ref_w.shape,kd.shape,ref_wd.shape)
        wdd = (action - wd * self.damp + fieldforce)/self.mass
        wd = wdd * self.dt + wd
        w = wd * self.dt + w
        return w, wd, wdd

# if __name__ == "__main__":
#     mass = 1
#     damp = 1
#     dt = 0.1
#     masspoint = Masspoint(mass, damp, dt)

#     kp = 20
#     kd = 20
#     fieldforce = 0
#     w = np.array([8, 10])
#     wd = np.array([2, 3])
#     ref_w = np.array([7, 9])
#     ref_wd = np.array([2, 4])
#     print(masspoint.generateNextState(w, wd, ref_w, ref_wd, kp, kd, fieldforce))
