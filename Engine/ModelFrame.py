import torch


# TODO: Finish the comments on the code to explain each variable


class SystemModel:

    def __init__(self, f=None, Q=None, h=None, R=None, T=None, T_test=None, m=None, n=None, prior_Q=None,
                 prior_Sigma=None, prior_S=None):
        ####################
        #   Motion Model
        ####################

        self.y = None  # Observation vector
        self.x = None  # Current state vector
        self.x_prev = None  # Previous state vector

        self.f = f  # State transition function
        self.m = m  # Dimension of the state vector
        self.Q = Q  # Process noise covariance matrix

        self.Input = None  #
        self.Target = None  #
        self.lengthMask = None  #

        self.m1x_0 = None  #
        self.m1x_0_rand = None  #
        self.m1x_0_batch = None  #

        self.m2x_0 = None  #
        self.m2x_0_batch = None  #

        #########################
        #   Observation Model
        #########################

        self.n = n  #
        self.H = h  #
        self.R = R  #

        ################
        #   Sequence
        ################

        self.T = T            #
        self.T_test = T_test  #

        #########################
        #   Covariance Priors
        #########################

        self.prior_Q = torch.eye(self.m) if prior_Q is None else prior_Q
        self.prior_S = torch.eye(self.n) if prior_S is None else prior_S
        self.prior_Sigma = torch.zeros((self.m, self.m)) if prior_Sigma is None else prior_Sigma

    #####################
    #    Init Sequence
    #####################

    def InitSequence(self, m1x_0, m2x_0) -> None:
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0
        return None
