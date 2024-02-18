import torch.nn as nn
import torch
import time

import Engine.ModelFrame
import Interface.config


class ExtendedKalmanFilter:
    """
    Class for Extended Kalman Filter implementation.

    Args:
        SystemModel (object): Object containing system model parameters and functions
        args (object): Object containing additional arguments

    Attributes:
        sigma (Tensor): Placeholder for Sigma
        x (Tensor): Placeholder for x
        i (int): Counter
        KG_array (Tensor): Placeholder for KG_array
        batch_size (int): Batch size
        m2x_0_batch (Tensor): Placeholder for m2x_0_batch
        m1x_0_batch (Tensor): Placeholder for m1x_0_batch
        batched_H_T (Tensor): Placeholder for batched_H_T
        batched_H (Tensor): Placeholder for batched_H
        batched_F_T (Tensor): Placeholder for batched_F_T
        batched_F (Tensor): Placeholder for batched_F
        dy (Tensor): Placeholder for dy
        m2x_posterior (Tensor): Placeholder for m2x_posterior
        m1x_posterior (Tensor): Placeholder for m1x_posterior
        KG (Tensor): Placeholder for KG
        m2y (Tensor): Placeholder for m2y
        m1y (Tensor): Placeholder for m1y
        m2x_prior (Tensor): Placeholder for m2x_prior
        m1x_prior (Tensor): Placeholder for m1x_prior
        device (torch.device): Device to be used (cuda or cpu)
        f (function): Process model function
        m (int): Dimension of x
        Q (Tensor): Process noise covariance matrix
        h (function): Observation model function
        n (int): Dimension of y
        R (Tensor): Observation noise covariance matrix
        T (int): Sequence length
        T_test (int): Test sequence length

    Methods:
        Predict: Predicts the states of the system using the process model
        KGain: Computes the Kalman Gain
        Innovation: Computes the innovation term
        Correct: Corrects the predicted states using the Kalman Gain and innovation term
        Update: Updates the state estimates
        UpdateJacobians: Updates the Jacobian matrices
        Init_batched_sequence: Initializes the batched sequence
        GenerateBatch: Generates batched observations and updates the state estimates
    """

    def __init__(self, SystemModel: Engine.ModelFrame.SystemModel, args: Interface.config.general_settings()):
        # Device
        self.sigma = None
        self.x = None
        self.i = None
        self.KG_array = None
        self.batch_size = None
        self.m2x_0_batch = None
        self.m1x_0_batch = None
        self.batched_H_T = None
        self.batched_H = None
        self.batched_F_T = None
        self.batched_F = None
        self.dy = None
        self.m2x_posterior = None
        self.m1x_posterior = None
        self.KG = None
        self.m2y = None
        self.m1y = None
        self.m2x_prior = None
        self.m1x_prior = None
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Process model
        self.f = SystemModel.f
        self.m = SystemModel.m

        self.Q = SystemModel.Q.to(self.device)
        # Observation model
        self.h = SystemModel.H
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)

        # Sequence length
        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

    # Predict
    def Predict(self):
        """Predicts the moments of x and y.
        :return: None
        """
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior).to(self.device)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(self.m1x_posterior, self.f), getJacobian(self.m1x_prior, self.h))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)
        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        """
        Calculate the Kalman Gain(KG).

        The KG is calculated using the following steps:
        1. Perform matrix multiplication between the m2x_prior and batched_H_T tensors.
        2. Perform matrix multiplication between the resulting tensor and the inverse of the m2y tensor.
        3. Save the KG in the KG_array tensor at the corresponding position.
        4. Increment the value of 'i'.

        :return: None
        """
        self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

        # Save KalmanGain
        self.KG_array[:, :, :, self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        """
        :param y: the value to subtract from self.m1y
        :return: None
        """
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        """
        Compute the 1-st and 2-nd posterior moments.

        :return: None
        """
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self, y):
        """
        Updates the filter using the given measurement.

        :param y: The measurement value.
        :return: A tuple containing the updated posterior means of the two states, m1x_posterior and m2x_posterior.
        """
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    #########################

    def UpdateJacobians(self, F, H):
        """
        :param F: The batched F matrix
        :param H: The batched H matrix
        :return: None

        This method updates the Jacobians for the given F and H matrices.

        The F and H matrices are batched matrices that represent Jacobians for some mathematical operations.
         The method transfers the F and H matrices to the device specified by self.device.
        *. It then calculates the transpose of F and H matrices and assigns them to self.batched_F_T
        and self.batched_H_T respectively.

        Note that the method does not return any value.
        """
        self.batched_F = F.to(self.device)
        self.batched_F_T = torch.transpose(F, 1, 2)
        self.batched_H = H.to(self.device)
        self.batched_H_T = torch.transpose(H, 1, 2)

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        """
        Initialize a batched sequence.

        :param m1x_0_batch: The initial value for m1x_0_batch. Shape: [batch_size, m, 1]
        :param m2x_0_batch: The initial value for m2x_0_batch. Shape: [batch_size, m, m]
        :return: None
        """
        self.m1x_0_batch = m1x_0_batch  # [batch_size, m, 1]
        self.m2x_0_batch = m2x_0_batch  # [batch_size, m, m]

    ######################
    #   Generate Batch
    ######################
    def GenerateBatch(self, y):
        """
        Generate batched data.

        :param y: Input data.
        :return: None.
        """
        y = y.to(self.device)
        self.batch_size = y.shape[0]  # batch size
        T = y.shape[2]  # sequence length (maximum length if randomLength=True)

        # Pre allocate KG array
        self.KG_array = torch.zeros([self.batch_size, self.m, self.n, T]).to(self.device)
        self.i = 0  # Index for KG_array allocation

        # Allocate Array for 1st and 2nd order moments (use zero padding)
        self.x = torch.zeros(self.batch_size, self.m, T).to(self.device)
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T).to(self.device)

        # Set 1st and 2nd order moments for t=0
        self.m1x_posterior = self.m1x_0_batch.to(self.device)
        self.m2x_posterior = self.m2x_0_batch.to(self.device)

        # Generate in a batched manner
        for t in range(0, T):
            yt = torch.unsqueeze(y[:, :, t], 2)
            xt, sigmat = self.Update(yt)
            self.x[:, :, t] = torch.squeeze(xt, 2)
            self.sigma[:, :, :, t] = sigmat


def getJacobian(x, g):
    """
    Currently, pytorch does not have a built-in function to compute Jacobian matrix
    in a batched manner, so we have to iterate over the batch dimension.

    input x (torch.tensor): [batch_size, m/n, 1]
    input g (function): function to be differentiated
    output Jac (torch.tensor): [batch_size, m, m] for f, [batch_size, n, m] for h
    """

    # TODO: Compare methods
    # Method 1: using autograd.functional.jacobian
    # batch_size = x.shape[0]
    # Jac_x0 = torch.squeeze(autograd.functional.jacobian(g, torch.unsqueeze(x[0,:,:],0)))
    # Jac = torch.zeros([batch_size, Jac_x0.shape[0], Jac_x0.shape[1]])
    # Jac[0,:,:] = Jac_x0
    # for i in range(1,batch_size):
    #     Jac[i,:,:] = torch.squeeze(autograd.functional.jacobian(g, torch.unsqueeze(x[i,:,:],0)))
    # Method 2: using F, H directly
    _, Jac = g(x, jacobian=True)
    return Jac


def EKFTest(args, SysModel, test_input, test_target, test_init=None, test_lengthMask=None):
    """
    :param args: Dictionary containing arguments for the method.
    :param SysModel: System model object.
    :param test_input: Tensor containing the input data for testing.
    :param test_target: Tensor containing the target data for testing.
    :param test_init: Tensor containing the initial values for testing.
    :param test_lengthMask: Tensor containing the length masks for testing.
    :return: List containing the MSE values, average MSE, average MSE in dB, KG array, and EKF output.

    This method performs testing using the Extended Kalman Filter (EKF) algorithm. It takes various parameters such
    as the system model, test input and target data, initialization settings
    *, and length masks. It returns a list of values including MSE values, average MSE, average MSE in dB, KG array,
     and EKF output.
    """
    # Number of test samples
    N_T = test_target.size()[0]

    # Loss
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.zeros(N_T)

    # Allocate empty tensor for output
    # EKF_out = torch.zeros([N_T, SysModel.m, test_input.size()[2]])               # N_T x m x T
    # KG_array = torch.zeros([N_T, SysModel.m, SysModel.n, test_input.size()[2]])  # N_T x m x n x T

    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel, args)

    # Init and Forward Computation
    EKF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1))
    EKF.GenerateBatch(test_input)

    end = time.time()
    t = end - start

    KG_array = EKF.KG_array
    EKF_out = EKF.x

    # MSE loss
    for j in range(N_T):  # Cannot use batch due to different length and std computation
        if args.randomLength:
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[j, :, test_lengthMask[j]],
                                            test_target[j, :, test_lengthMask[j]]).item()
        else:
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[j, :, :], test_target[j, :, :]).item()

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_linear_std = torch.std(MSE_EKF_linear_arr, unbiased=True)

    # Confidence interval
    EKF_std_dB = 10 * torch.log10(MSE_EKF_linear_std + MSE_EKF_linear_avg) - MSE_EKF_dB_avg

    print("Extended Kalman Filter - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("Extended Kalman Filter - STD:", EKF_std_dB, "[dB]")

    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]
