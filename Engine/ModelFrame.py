import torch


class SystemModel:

    def __init__(self, f, Q, h, R, T, T_test, m, n, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        #   Motion Model
        ####################
        self.Target = None
        self.Input = None
        self.lengthMask = None
        self.m1x_0_rand = None
        self.y = None
        self.x = None
        self.m2x_0_batch = None
        self.x_prev = None
        self.m1x_0_batch = None
        self.m2x_0 = None
        self.m1x_0 = None
        self.f = f
        self.m = m
        self.Q = Q

        #########################
        #   Observation Model
        #########################
        self.H = h
        self.n = n
        self.R = R

        ################
        #   Sequence
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        #   Covariance Priors
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S

    #####################
    #    Init Sequence
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0
