import torch.nn.functional as func
import torch.nn as nn
import torch


class KalmanNet(torch.nn.Module):
    """
    The `KalmanNetNN` class is a subclass of `torch.nn.Module` and represents a neural network implementation of the
    Kalman filter.

    Constructor:
        The constructor initializes the various attributes of the class. These attributes include the Kalman gain
        (`KGain`), the first moment of the observation (`m1y`), the prior first
    * moment of the state (`m1x_prior`), the observation noise covariance matrix (`h_Sigma`), the previous prior first
    moment of the state (`m1x_prior_previous`), the previous observation
    * (`y_previous`), the previous posterior first moment of the state (`m1x_posterior_previous`), the posterior first
    moment of the state (`m1x_posterior`), the state evolution function
    * (`f`), the observation function (`h`), the number of state variables (`m`), the number of observation variables
    (`n`), the time step (`T`), and various fully connected layers (`FC
    *7`, `d_output_FC7`, `d_input_FC7`, `FC6`, `d_output_FC6`, `d_input_FC6`, `FC5`, `d"""
    ###################
    #   Constructor
    ###################
    def __init__(self):
        super().__init__()
        self.KGain = None
        self.m1y = None
        self.m1x_prior = None
        self.h_Sigma = None
        self.m1x_prior_previous = None
        self.y_previous = None
        self.m1x_posterior_previous = None
        self.m1x_posterior = None
        self.f = None
        self.m = None
        self.h = None
        self.n = None
        self.T = None
        self.FC7 = None
        self.d_output_FC7 = None
        self.d_input_FC7 = None
        self.FC6 = None
        self.d_output_FC6 = None
        self.d_input_FC6 = None
        self.FC5 = None
        self.d_output_FC5 = None
        self.d_input_FC5 = None
        self.FC4 = None
        self.d_output_FC4 = None
        self.d_input_FC4 = None
        self.FC3 = None
        self.d_output_FC3 = None
        self.d_input_FC3 = None
        self.FC2 = None
        self.d_hidden_FC2 = None
        self.d_output_FC2 = None
        self.h_S = None
        self.d_input_FC2 = None
        self.FC1 = None
        self.d_output_FC1 = None
        self.d_input_FC1 = None
        self.GRU_S = None
        self.d_hidden_S = None
        self.d_input_S = None
        self.h_Q = None
        self.GRU_Sigma = None
        self.d_hidden_Sigma = None
        self.d_input_Sigma = None
        self.GRU_Q = None
        self.d_hidden_Q = None
        self.d_input_Q = None
        self.prior_S = None
        self.prior_Sigma = None
        self.prior_Q = None
        self.batch_size = None
        self.seq_len_input = None
        self.device = None

    def NNBuild(self, SysModel, args):
        """
        :param SysModel: The system model containing information about the dynamics and parameters of the system.
        :param args: Additional arguments for configuring the neural network.
        :return: None

        This method is used to build a neural network for Kalman gain estimation. It takes in the system model
        and additional arguments as parameters.

        The method first checks if the 'use_cuda' flag in the arguments is True. If it is, the device will be set
        to 'cuda' for GPU acceleration. Otherwise, it will be set to 'cpu'.

        Next, the method calls the 'InitSystemDynamics' method with the system model's f, h, m, and n parameters
        to initialize the system dynamics.

        After that, the method initializes the Kalman gain neural network by calling the 'InitKGainNet' method
        with the system model's prior Q, prior Sigma, prior S parameters, and the additional
        * arguments.

        Note: The commented lines regarding the number of neurons in the hidden layers are provided as references
        and can be adjusted based on specific requirements.
        """
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.InitSystemDynamics(SysModel.f, SysModel.H, SysModel.m, SysModel.n)

        # Number of neurons in the 1st hidden layer
        # H1_KNet = (SysModel.m + SysModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        # H2_KNet = (SysModel.m * SysModel.n) * 1 * (4)

        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    ######################################
    #   Initialize Kalman Gain Network
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        """
        Initializes the KGainNet model.

        :param prior_Q: Prior for the latent variable Q
        :param prior_Sigma: Prior for the latent variable Sigma
        :param prior_S: Prior for the latent variable S
        :param args: Additional arguments
        :return: None

        This method initializes the KGainNet model by setting the necessary parameters and creating the
        required layers and modules.

        **Parameters:**

        - `prior_Q` (Tensor): Prior for the latent variable Q
        - `prior_Sigma` (Tensor): Prior for the latent variable Sigma
        - `prior_S` (Tensor): Prior for the latent variable S
        - `args` (object): Additional arguments

        **Returns:**

        - `None`
        """
        self.seq_len_input = 1          # KNet calculates time-step by time-step
        self.batch_size = args.n_batch  # Batch size

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        # GRU to track Q
        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU()).to(self.device)

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2)).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU()).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU()).to(self.device)

        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU()).to(self.device)

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU()).to(self.device)

        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU()).to(self.device)

    ##################################
    #   Initialize System Dynamics
    ##################################
    def InitSystemDynamics(self, f, h, m, n):
        """
        :param f: The state evolution function that describes how the system's state evolves over time.
        :param h: The observation function that maps the system's state to the observed measurements.
        :param m: The dimension of the state vector.
        :param n: The dimension of the measurement vector.
        :return: None

        This method initializes the system dynamics by setting the state evolution function (f), the observation
        function (h), and the dimensions of the state vector (m) and the measurement
        * vector (n).
        """
        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n

    ###########################
    #   Initialize Sequence
    ###########################
    def InitSequence(self, M1_0, T):
        """
        Initialize the sequence with given parameters.

        :param M1_0: Initial value of M1_0.
        :param T: The target value.
        :return: None
        """
        self.T = T

        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    ######################
    #   Compute Priors
    ######################
    def step_prior(self):
        """
        Perform the prior step in the filtering process.

        :return: None
        """
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)

    ##############################
    #   Kalman Gain Estimation
    ##############################
    def step_KGain_est(self, y):
        """
        :param y: Tensor of shape [batch_size, m, 1]. Observation input.
        :return: None.

        This method calculates the Kalman Gain for a given set of observations.

        The method takes in a tensor `y` representing the observations. The tensor should have a shape of
        [batch_size, m, 1], where `batch_size` is the number of input samples, and `m` is the
        * number of observation dimensions.

        The method performs the following steps:
        1. Calculate the differences between the current observation `y` and the previous observation `self.y_previous`.
        2. Calculate the differences between the current observation `y` and the innovation term `self.m1y`.
        3. Calculate the differences between the current posterior state estimates `self.m1x_posterior` and
        the previous posterior state estimates `self.m1x_posterior_previous`.
        4. Calculate the differences between the current posterior state estimates `self.m1x_posterior` and
        the previous prior state estimates `self.m1x_prior_previous`.
        5. Normalize the above differences using the L2 norm along dimension 1.
        6. Perform the Kalman Gain step by calling the `KGain_step` method.
        7. Reshape the resulting Kalman Gain tensor to [batch_size, m, n] and store it in the `self.KGain` attribute.

        Note: This method assumes the presence of certain attributes (`self.y_previous`, `self.m1y`,
        `self.m1x_posterior`, `self.m1x_posterior_previous`, `self.m1x_prior_previous`) which are
        * used in the calculations.
        """
        # both in size [batch_size, n]
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
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    #######################
    #    Kalman Net Step
    #######################
    def KNet_step(self, y):
        """
        :param y: A tensor representing the observation values. Shape should be [batch_size, n, 1].
        :return: A tensor representing the updated posterior moment. Shape is [batch_size, n, 1].

        This method performs one step of the Kalman filter algorithm. It takes an observation value and updates
        the posterior moment based on the observation and previous state estimates.

        Steps:
        1. Compute Priors: The method calls the step_prior() method to compute the prior estimates.
        2. Compute Kalman Gain: The method calls the step_KGain_est() method to compute the Kalman Gain.
        3. Innovation: Compute the difference between the observation value and the previous state estimate.
        4. Compute the 1st posterior moment: Multiply the Kalman Gain by the innovation to obtain the posterior moment.
        5. Update previous state estimates: Update the previous state estimates with the current estimates.
        6. Update previous observation: Update the previous observation with the current observation value.
        7. Return the updated posterior moment.

        Example usage:
        ```
        # Create an instance of the class
        knet = KNet()

        # Generate observation value 'y'
        y = torch.tensor([[[0.5], [1.0], [1.5]]])

        # Perform one step of the Kalman filter algorithm
        result = knet.KNet_step(y)
        ```
        """
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

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

    ########################
    #    Kalman Gain Step
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        """
        Applies the KGain step in the forward and backward flow.

        :param obs_diff: The observation difference.
        :param obs_innov_diff: The observation innovation difference.
        :param fw_evol_diff: The forward evolution difference.
        :param fw_update_diff: The forward update difference.
        :return: The output of FC2 layer.

        """
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

        # FC 5
        in_FC5 = fw_update_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        # FC 6
        in_FC6 = fw_evol_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        #    Backward Flow
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2

    ###############
    #   Forward
    ###############
    def forward(self, y):
        """
        :param y: The input data to be propagated forward in the network.
        :return: The output of the forward propagation step.
        """
        y = y.to(self.device)
        return self.KNet_step(y)

    #########################
    #    Init Hidden State
    #########################
    def init_hidden_KNet(self):
        """
        Initialize hidden states for the KNet model.

        :return: None
        """
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size,
                                                                   1)  # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size,
                                                                           1)  # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size,
                                                                   1)  # batch size expansion
