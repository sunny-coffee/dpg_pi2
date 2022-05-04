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