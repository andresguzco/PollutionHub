import torch.nn as nn
import torch
import time


class KalmanFilter:
    """

    This class implements a Kalman Filter algorithm for state estimation. It contains methods for predicting
    the moments of the state variables, computing the Kalman Gain, updating the state
    * estimate based on new measurements, and initializing the batched sequence.

    Constructor:
        `__init__(self, SystemModel, args)`:
            - Initializes the KalmanFilter object with the SystemModel and args.
            - The SystemModel is an object that contains the system matrices (F, H, Q, and R) and other parameters.
            - The args is an object that contains arguments such as whether to use CUDA or CPU.

    Methods:
        `Predict(self)`:
            - Predicts the moments of the variables x and y based on the prior knowledge.

        `KGain(self)`:
            - Calculates the Kalman gain using the given parameters.

        `Innovation(self, y)`:
            - Computes the innovation (difference between the measurement and predicted measurement).

        `Correct(self)`:
            - Computes the posterior moments based on the Kalman gain and innovation.

        `Update(self, y) -> (tensor, tensor)`:
            - Updates the state estimate based on the new measurement.
            - Returns the updated state estimate, consisting of m1x_posterior and m2x_posterior.

        `Init_batched_sequence(self, m1x_0_batch, m2x_0_batch)`:
            - Initializes the batched sequence with the given initial values.

        `GenerateBatch(self, y)`:
            - Generates the state estimates for a batch of measurements.
            - y: batch of observations [batch_size, n, T]

    Attributes:
        - sigma: Tensor representing the covariance matrix.
        - x: Tensor representing the state variables.
        - batched_H_T: Tensor representing the transposed batched H matrix.
        - batched_H: Tensor representing the batched H matrix.
        - batched_F_T: Tensor representing the transposed batched F matrix.
        - batched_F: Tensor representing the batched F matrix.
        - batch_size: Integer representing the batch size.
        - m2x_0_batch: Tensor representing the initial values for m2x_0 in the batched sequence.
        - m1x_0_batch: Tensor representing the initial values for m1x_0 in the batched sequence.
        - m2x_posterior: Tensor representing the 2nd posterior moment of x.
        - m1x_posterior: Tensor representing the 1st posterior moment of x.
        - dy: Tensor representing the difference between the measurement and predicted measurement.
        - KG: Tensor representing the Kalman gain.
        - m2y: Tensor representing the 2nd moment of y.
        - m1y: Tensor representing the 1st moment of y.
        - m2x_prior: Tensor representing the 2nd prior moment of x.
        - m1x_prior: Tensor representing the 1st prior moment of x.
        - device: Device on which the calculations are performed.
        - F: SystemModel's F matrix.
        - m: SystemModel's m.
        - Q: SystemModel's Q matrix.
        - H: SystemModel's H matrix.
        - n: SystemModel's n.
        - R: SystemModel's R matrix.
        - T: SystemModel's T.
        - T_test: SystemModel's T_test.

    """

    def __init__(self, SystemModel, args):
        # Device
        self.sigma = None
        self.x = None
        self.batched_H_T = None
        self.batched_H = None
        self.batched_F_T = None
        self.batched_F = None
        self.batch_size = None
        self.m2x_0_batch = None
        self.m1x_0_batch = None
        self.m2x_posterior = None
        self.m1x_posterior = None
        self.dy = None
        self.KG = None
        self.m2y = None
        self.m1y = None
        self.m2x_prior = None
        self.m1x_prior = None
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.F = SystemModel.F
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)

        self.H = SystemModel.H
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

    # Predict

    def Predict(self):
        """

        Predict method is used to predict the moments of the variables x and y based on the prior knowledge.

        :return: None

        """
        # Predict the 1-st moment of x
        self.m1x_prior = torch.bmm(self.batched_F, self.m1x_posterior).to(self.device)

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.bmm(self.batched_H, self.m1x_prior)

        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        """
        Calculates the Kalman gain using the given parameters.

        :return: None
        """
        self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)

        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

    # Innovation
    def Innovation(self, y):
        """
        :param y: The value of y to be used in the calculation
        :return: None

        """
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        """
        Compute the 1st and 2nd posterior moments.

        :return: None
        """
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self, y):
        """
        Updates the state estimate based on the new measurement.

        :param y: The new measurement.
        :return: The updated state estimate, consisting of m1x_posterior and m2x_posterior.
        """
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        """
        Initialize the batched sequence with the given initial values.

        :param m1x_0_batch: Initial values for variable m1x_0 in the batched sequence. Shape: [batch_size, m, 1]
        :param m2x_0_batch: Initial values for variable m2x_0 in the batched sequence. Shape: [batch_size, m, m]
        :return: None
        """
        self.m1x_0_batch = m1x_0_batch  # [batch_size, m, 1]
        self.m2x_0_batch = m2x_0_batch  # [batch_size, m, m]

    ######################
    #   Generate Batch
    ######################
    def GenerateBatch(self, y):
        """
        :param y: Input batch tensor of shape (batch_size, n, T) where batch_size is the number of examples,
        n is the number of measurements, and T is the sequence length. :return: None

        This method generates a batch of data points using the input batch tensor.

        The input tensor `y` is converted to the device specified by `self.device`.

        The batch size is set to the number of examples in `y`, and the sequence length `T` is determined based on
        the shape of `y`.

        The matrix `self.F` is broadcasted to match the batch size and expanded dimensions. The resulting batched F
        is stored in `self.batched_F`.

        The transpose of `self.batched_F` is stored in `self.batched_F_T`.

        The matrix `self.H` is broadcasted to match the batch size and expanded dimensions. The resulting batched H
        is stored in `self.batched_H`.

        The transpose of `self.batched_H` is stored in `self.batched_H_T`.

        An array `self.x` of zeros with shape (batch_size, self.m, T) is allocated on the device.

        An array `self.sigma` of zeros with shape (batch_size, self.m, self.m, T) is allocated on the device.

        The 1st order moment `self.m1x_posterior` is set to the initial values stored in `self.m1x_0_batch` and moved
        to the device.

        The 2nd order moment `self.m2x_posterior` is set to the initial values stored in `self.m2x_0_batch` and moved
        to the device.

        A loop is executed for each time step `t` from 0 to `T-1`.

        For each time step:
        - The `yt` tensor is obtained by selecting the measurements at time step `t` from `y`.

        - The `xt` and `sigmat` tensors are obtained by calling the `Update` method with `yt`.

        - The dimensions of `xt` are reduced by squeezing the third dimension.

        - The `xt` tensor is stored in the `self.x` array at the corresponding time step `t`.

        - The `sigmat` tensor is stored in the `self.sigma` array at the corresponding time step `t`.
        """
        y = y.to(self.device)
        self.batch_size = y.shape[0]  # batch size
        T = y.shape[2]  # sequence length (maximum length if randomLength=True)

        # Batched F and H
        self.batched_F = self.F.view(1, self.m, self.m).expand(self.batch_size, -1, -1).to(self.device)
        self.batched_F_T = torch.transpose(self.batched_F, 1, 2).to(self.device)
        self.batched_H = self.H.view(1, self.n, self.m).expand(self.batch_size, -1, -1).to(self.device)
        self.batched_H_T = torch.transpose(self.batched_H, 1, 2).to(self.device)

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


def KFTest(args, SysModel, test_input, test_target, allStates=True, \
           randomInit=False, test_init=None, test_lengthMask=None):
    """
    :param args: The arguments for the KFTest method.
    :param SysModel: The system model for the Kalman Filter.
    :param test_input: The test input for the Kalman Filter.
    :param test_target: The target output for the Kalman Filter.
    :param allStates: A boolean value indicating whether to consider all states or not. Defaults to True.
    :param randomInit: A boolean value indicating whether to randomly initialize the Kalman Filter sequence.
    :param test_init: The test initialization for the Kalman Filter. Defaults to None.
    :param test_lengthMask: The test length mask for the Kalman Filter. Defaults to None.
    :return: A list containing the MSE loss, average MSE, average MSE in dB, and the Kalman Filter output.
    """
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.zeros(args.N_T)
    # allocate memory for KF output
    KF_out = torch.zeros(args.N_T, SysModel.m, args.T_test)
    if not allStates:
        loc = torch.tensor([True, False, False])  # for position only
        if SysModel.m == 2:
            loc = torch.tensor([True, False])  # for position only

    start = time.time()

    KF = KalmanFilter(SysModel, args)
    # Init and Forward Computation
    if randomInit:
        KF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(args.N_T, -1, -1))
    else:
        KF.Init_batched_sequence(SysModel.m1x_0.view(1, SysModel.m, 1).expand(args.N_T, -1, -1),
                                 SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(args.N_T, -1, -1))
    KF.GenerateBatch(test_input)

    end = time.time()
    t = end - start
    KF_out = KF.x
    # MSE loss
    for j in range(args.N_T):  # cannot use batch due to different length and std computation
        if allStates:
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j, :, test_lengthMask[j]],
                                               test_target[j, :, test_lengthMask[j]]).item()
            else:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j, :, :], test_target[j, :, :]).item()
        else:  # mask on state
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j, loc, test_lengthMask[j]],
                                               test_target[j, loc, test_lengthMask[j]]).item()
            else:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j, loc, :], test_target[j, loc, :]).item()

    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, unbiased=True)

    # Confidence interval
    KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - STD:", KF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out]
