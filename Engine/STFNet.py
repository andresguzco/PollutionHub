import torch.nn.functional as func
import torch.nn as nn
import torch


# TODO: Run the filter on each of the epochs to optimise it. Review the computatino of the oslostt


class STFNet(torch.nn.Module):
    ###################
    #   Constructor
    ###################

    def __init__(self):
        super().__init__()
        self.m1y = None                           # First moment of y
        self.KGain = None                         # Kalman Gain
        self.y_previous = None                    # Previous observation

        self.m1x_prior = None                     # State at time t given t -1
        self.m1x_prior_previous = None            # State at t - 1 given t - 2
        self.m1x_posterior = None                 # State at time t given t
        self.m1x_posterior_previous = None        # State at time t - 1 given t - 1

        self.m = None                             # Dimension of the state vector
        self.n = None                             # Dimension of the observation vector
        self.T = None                             # Time steps

        self.f = None                             # State transition function
        self.h = None                             # Observation transition function
        self.P = None                             # State covariance (P)
        self.Q = None                             # Process noise covariance (Q)
        self.R = None                             # Measurement noise covariance (R)

        self.FC4 = None                           # Fully connected (linear) layer of the neural network
        self.FC3 = None                           # Fully connected (linear) layer of the neural network
        self.FC2 = None                           # Fully connected (linear) layer of the neural network
        self.FC1 = None                           # Fully connected (linear) layer of the neural network

        self.d_output_FC4 = None                  # Dimensions of the output for FC3
        self.d_output_FC3 = None                  # Dimensions of the output for FC3
        self.d_output_FC2 = None                  # Dimensions of the output for FC2
        self.d_output_FC1 = None                  # Dimensions of the output for FC1

        self.d_input_FC4 = None                   # Dimensions of the input for FC3
        self.d_input_FC3 = None                   # Dimensions of the input for FC3
        self.d_input_FC2 = None                   # Dimensions of the input for FC2
        self.d_input_FC1 = None                   # Dimensions of the input for FC1

        self.d_hidden_FC4 = None                  # The dimensions of the hidden states for FC2
        self.d_hidden_FC3 = None                  # The dimensions of the hidden states for FC2
        self.d_hidden_FC2 = None                  # The dimensions of the hidden states for FC2
        self.d_hidden_FC1 = None                  # The dimensions of the hidden states for FC2

        self.device = None                        # Computing device
        self.batch_size = None                    # Size of the input batch
        self.seq_len_input = None                 # Size of the sequence length of the input data

    def NNBuild(
            self,
            m: int,
            n: int,
            use_cuda: bool,
            Q,
            Sigma,
            S,
            n_batch: int,
            dropout_probability: float,
            hidden_dims: int,
            T: int,
            M1_0
    ) -> None:
        # Device
        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        ##################################
        #   Initialize System Dynamics
        ##################################
        self.m = m
        self.n = n
        self.T = T

        self.h = torch.eye(self.n, self.n, device=self.device)
        self.f = torch.eye(self.m, self.m, device=self.device)

        self.Q = Q.to(self.device)
        self.R = Sigma.to(self.device)
        self.P = S.to(self.device)

        self.InitSTFNet(n_batch, dropout_probability, hidden_dims)

        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h @ self.m1x_posterior
        return None

    ######################################
    #   Initialize Kalman Gain Network
    ######################################

    def InitSTFNet(
            self,
            n_batch: int,
            dropout_probability: float,
            hidden_dims: int
    ) -> None:
        self.seq_len_input = 1          # Calculates time-step by time-step
        self.batch_size = n_batch       # Batch size

        # Fully connected 1
        self.d_input_FC1 = self.m
        self.d_output_FC1 = self.m ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_hidden_FC1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_probability),
            nn.Linear(self.d_hidden_FC1, self.d_output_FC1),
            nn.BatchNorm1d(self.d_output_FC1)
        ).to(self.device)

        # Fully connected 2
        self.d_input_FC2 = self.m
        self.d_output_FC2 = self.m ** 2
        self.d_hidden_FC2 = self.d_input_FC2 * hidden_dims
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2)
        ).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = 2 * self.n
        self.d_output_FC3 = self.m ** 2
        self.d_hidden_FC3 = self.d_input_FC3 * hidden_dims
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_hidden_FC3),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC3, self.d_output_FC3)
        ).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_output_FC1 + self.d_output_FC2 + self.d_output_FC3
        self.d_output_FC4 = self.m ** 2
        self.d_hidden_FC4 = self.d_input_FC4 * hidden_dims
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_hidden_FC4),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC4, self.d_output_FC4)
        ).to(self.device)
        return None

    ######################
    #   Compute Priors
    ######################

    def StepPrior(self, y) -> None:
        # Predict the 1-st moment of x
        self.stepStateTransitionEst(y)
        self.m1x_prior = self.f @ self.m1x_posterior
        # Predict the 1-st moment of y
        self.m1y = self.h @ self.m1x_prior
        return None

    ##############################
    # State Transition Estimation
    ##############################

    def stepStateTransitionEst(self, y: torch.tensor):
        # Both in size [batch_size, n]
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)

        # both in size [batch_size, m]
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_previous, 2)
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_previous, 2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        # Kalman Gain Network Step
        STF = self.STFstep(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape State transition function to a Matrix
        self.f = torch.reshape(STF, (self.batch_size, self.m, self.n))
        return None

    def computeKGain(self) -> None:
        P = self.P.squeeze(0).view(self.batch_size, self.m, self.m)      # State covariance
        R = self.R.squeeze(0).view(self.batch_size, self.n, self.n)  # Measurement noise covariance

        # Perform batched matrix operations for Kalman Gain computation
        HT = torch.transpose(self.h, dim0=1, dim1=2)                         # Transpose H for each batch item
        S = torch.bmm(self.h, torch.bmm(P, HT)) + R                          # Innovation covariance
        S_inv = torch.inverse(S)                                             # Inverse of S for each item in the batch
        KGain = torch.bmm(torch.bmm(P, HT), S_inv)                           # Kalman Gain for each item in the batch

        # Store the computed Kalman Gain
        self.KGain = KGain
        return None

    #######################
    #    STF Net Step
    #######################

    def STFNet_step(self, y: torch.tensor) -> torch.tensor:
        # Compute Priors
        self.StepPrior(y)

        # Compute Kalman Gain
        self.computeKGain()

        # Innovation
        dy = y - self.m1y  # [batch_size, n, 1]

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        # self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        # return
        return self.m1x_posterior

    ##################################
    # State Transition Function Step
    ##################################

    def STFstep(
            self,
            obs_diff: torch.tensor,
            obs_innov_diff: torch.tensor,
            fw_evol_diff: torch.tensor,
            fw_update_diff: torch.tensor
    ) -> torch.tensor:
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        #    Forward Flow
        ####################

        # FC 1
        in_FC1 = fw_update_diff
        out_FC1 = self.FC1(in_FC1)
        # FC 2
        in_FC2 = fw_evol_diff
        out_FC2 = self.FC2(in_FC2)
        # FC 3
        in_FC3 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC3 = self.FC3(in_FC3)
        # FC 4
        in_FC4 = torch.cat((out_FC1, out_FC2, out_FC3), 3)
        out_FC4 = self.FC4(in_FC4)

        return out_FC4

    ###############
    #   Forward
    ###############

    def forward(self, y: torch.tensor) -> torch.tensor:
        y = y.to(self.device)
        return self.STFNet_step(y)
