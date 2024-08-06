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
            nn.Linear(512, self.output_dim ** 2),
            nn.ReLU()
        )
        self.vars = torch.Parameter(torch.randn(2))

    def forward(self, Y, Z):
        output = self.network(Z)
        T_flat = output.detach()
        params = self.vars

        T = T_flat.reshape(output.shape[0], self.output_dim, self.output_dim) 
        H = torch.eye(self.output_dim).unsqueeze(0).expand(output.shape[0], -1, -1) * params[0]
        Q = torch.eye(self.output_dim).unsqueeze(0).expand(output.shape[0], -1, -1) * params[1]

        y_out = self._KalmanFilter(Y, T, H, Q)
        return y_out

    def predict(self, alpha_prev, P_prev, Q, T_t):
        alpha_pred = torch.matmul(T_t, alpha_prev)
        first_part = torch.matmul(T_t, P_prev)
        P_new = torch.matmul(first_part, T_t.t()) + Q
        return alpha_pred, P_new


    def update(self, alpha_pred, P_pred, y, H):
        K = torch.matmul(P_pred, torch.inverse(P_pred + H))
        alpha_upd = alpha_pred + torch.matmul(K, (y - alpha_pred).t()).t()
        P_upd = torch.eye(K.size(0)) - K.matmul(P_pred)
        return alpha_upd, P_upd

    def _KalmanFilter(self, Y, T, Q, H):
        n_timesteps = len(Y)
        n_states = self.output_dim

        alpha_estimates = [torch.ones(1, self.output_dim)]
        P_estimates = [torch.eye(self.output_dim).unsqueeze(0)]

        for t in range(1, n_timesteps):
            alpha_t, P_t = self.predict(
                alpha_prev=alpha_estimates[-1].squeeze(0),
                P_prev=P_estimates[-1].squeeze(0),
                Q=Q[t, :, :],
                T_t=T[t-1, :, :]
            )

            alpha_updated, P_updated = self.update(
                alpha_pred=alpha_t,
                P_pred=P_t,
                y=Y[t, :],
                H=H[t, :, :]
            )

            alpha_estimates.append(alpha_updated.unsqueeze(0))
            P_estimates.append(P_updated.unsqueeze(0))

        alpha_estimates = torch.cat(alpha_estimates, dim=0)
        P_estimates = torch.cat(P_estimates, dim=0)

        return alpha_estimates, P_estimates
