import torch.nn as nn
import torch


class STFNet(nn.Module):
    def __init__(
            self,
            state_dim,
            output_dim
    ):
        super(STFNet, self).__init__()
        self.output_dim = output_dim
        self.x_dim = state_dim
        self.network = nn.Sequential(
            nn.Linear(self.x_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, (self.output_dim ** 2) + 2)
        )

    def forward(self, Y, Z):
        output = self.network(Z)
        T_flat = output[:, :-2].detach()
        params = output[:, -2:].exp().detach()

        T = T_flat.reshape(output.shape[0], self.output_dim, self.output_dim)
        H = torch.zeros(output.shape[0], self.output_dim, self.output_dim)
        Q = torch.zeros(output.shape[0], self.output_dim, self.output_dim)

        for i in range(output.shape[1]):
            H[i, :, :] = torch.eye(self.output_dim) * params[i, 0]
            Q[i, :, :] = torch.eye(self.output_dim) * params[i, 1]

        return self.KalmanFilter(Y, T, H, Q)

    @staticmethod
    @torch.no_grad()
    def predict(alpha_prev, P_prev, Q, T_t):
        alpha_pred = torch.matmul(T_t, alpha_prev)
        first_part = torch.matmul(T_t, P_prev)
        P_new = torch.matmul(first_part, T_t.t()) + Q
        return alpha_pred, P_new

    @staticmethod
    @torch.no_grad()
    def update(alpha_pred, P_pred, y, H):
        K = torch.matmul(P_pred, torch.inverse(P_pred + H))
        alpha_upd = alpha_pred + torch.matmul(K, (y - alpha_pred).t()).t()
        P_upd = torch.eye(K.size(0)) - K.matmul(P_pred)
        return alpha_upd, P_upd

    @torch.no_grad()
    def KalmanFilter(self, Y, T, Q, H):
        n_timesteps = len(Y)
        n_states = self.output_dim

        alpha_estimates = torch.zeros(n_timesteps, n_states)
        P_estimates = torch.zeros(n_timesteps, n_states, n_states)

        alpha_estimates[0, :] = torch.ones(self.output_dim)
        P_estimates[0, :, :] = torch.eye(self.output_dim)

        for t in range(1, n_timesteps):
            alpha_t, P_t = self.predict(
                alpha_prev=alpha_estimates[t-1, :],
                P_prev=P_estimates[t-1, :, :],
                Q=Q[t, :, :],
                T_t=T[t-1, :, :]
            )

            alpha_estimates[t, :], P_estimates[t, :, :] = self.update(
                alpha_pred=alpha_t,
                P_pred=P_t,
                y=Y[t, :],
                H=H[t, :, :]
            )

        return alpha_estimates, P_estimates
