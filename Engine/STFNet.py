from scipy.optimize import minimize
import torch.nn as nn
import torch


class STFNet(nn.Module):
    def __init__(self, state_dim, time_steps, output_dim):
        super(STFNet, self).__init__()
        self.T = time_steps
        self.output_dim = output_dim
        self.x_dim = state_dim
        self.network = nn.Sequential(
            nn.Linear(self.x_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim ** 2)
        )

    def forward(self, Y, Z):
        params = self.estimate_parameters(Y, Z)
        # Ensure parameters used in Kalman Filter are detached
        Q = torch.eye(self.output_dim) * params[0]
        H = torch.eye(self.output_dim) * params[1]
        alpha_hat, _ = self.KalmanFilter(Y.detach(), Z.detach(), Q.detach(), H.detach())
        return alpha_hat

    @staticmethod
    def predict(alpha_prev, P_prev, Q, T_t):
        # Ensure tensors used in prediction are detached
        alpha_pred = torch.matmul(T_t.detach(), alpha_prev.detach())
        first_part = torch.matmul(T_t.detach(), P_prev.detach())
        P_new = torch.matmul(first_part, T_t.detach().t()) + Q.detach()
        return alpha_pred, P_new

    @staticmethod
    def update(alpha_pred, P_pred, y, H):
        # Detach tensors used in update step
        K = torch.matmul(P_pred.detach(), torch.inverse(P_pred.detach() + H.detach()))
        alpha_upd = alpha_pred.detach().clone()
        alpha_upd += torch.matmul(K, (y.detach() - alpha_pred.detach()).t()).t()
        P_upd = torch.eye(K.size(0)).detach().clone() - K.matmul(P_pred.detach())
        return alpha_upd, P_upd

    def KalmanFilter(self, Y, Z, Q, H):
        n_timesteps = len(Y)
        n_states = self.output_dim

        # Ensure transformation matrix T is detached
        T_flat = self.network(Z).detach()
        T = T_flat.reshape(n_timesteps, self.output_dim, self.output_dim)

        # Initialize storage for states, ensuring detachment
        alpha_estimates = torch.zeros(n_timesteps, n_states).detach()
        P_estimates = torch.zeros(n_timesteps, n_states, n_states).detach()

        # Initial state, detached
        alpha_estimates[0, :] = torch.ones(self.output_dim).detach()
        P_estimates[0, :, :] = torch.eye(self.output_dim).detach()

        # Run Kalman filter with detached tensors
        for t in range(1, n_timesteps):
            alpha_t, P_t = self.predict(
                alpha_prev=alpha_estimates[t-1, :],
                P_prev=P_estimates[t-1, :, :],
                Q=Q,
                T_t=T[t-1, :, :]
            )

            alpha_estimates[t, :], P_estimates[t, :, :] = self.update(
                alpha_pred=alpha_t,
                P_pred=P_t,
                y=Y[t, :],
                H=H
            )

        return alpha_estimates, P_estimates

    def log_likelihood(self, params, Y_numpy, Z_numpy):
        # Ensure Q and H used in log likelihood are detached
        Q = torch.eye(self.output_dim) * params[0]
        H = torch.eye(self.output_dim) * params[1]
        Y = torch.tensor(Y_numpy).detach()
        Z = torch.tensor(Z_numpy).detach()

        # Run the Kalman filter with detached tensors
        alpha_estimates, P_estimates = self.KalmanFilter(Y, Z, Q.detach(), H.detach())

        # Compute log-likelihood with detached tensors
        log_likelihood = 0
        inv_H = torch.inverse(H.detach())

        for t in range(len(Y)):
            log_likelihood += 0.5 * torch.matmul(
                Y[t].detach() - alpha_estimates[t].detach(),
                torch.matmul(inv_H, Y[t].detach() - alpha_estimates[t].detach())
            )

        return float(-log_likelihood - (0.5 * torch.log(torch.det(2 * torch.pi * H.detach()))) * len(Y))

    def estimate_parameters(self, Y, Z, initial_guess=None):
        if initial_guess is None:
            initial_guess = [1.0, 1.0]

        result = minimize(
            fun=self.log_likelihood,
            x0=initial_guess,
            args=(Y.detach().numpy(), Z.detach().numpy()),
            method='L-BFGS-B'
        )
        return result.x
