import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import StochasticTwoLevelDataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def save_model(model, model_name):
    torch.save(model.func.state_dict(), './saved_models/{}_func.pt'.format(model_name))
    torch.save(model.dec.state_dict(), './saved_models/{}_dec.pt'.format(model_name))
    torch.save(model.rec.state_dict(), './saved_models/{}_rec.pt'.format(model_name))
    torch.save(model.epsilon, './saved_models/{}_epsilon.pt'.format(model_name))

def load_model(model, model_name):
    '''
    model.func.load_state_dict(
        torch.load('saved_models/trained_closed_3_6_48-48_fine_tuned_old_func.pt', map_location=torch.device('cpu')))
    model.dec.load_state_dict(torch.load('saved_models/trained_closed_3_6_48-48_fine_tuned_old_dec.pt', map_location=torch.device('cpu')))
    model.rec.load_state_dict(torch.load('saved_models/trained_closed_3_6_48-48_fine_tuned_old_rec.pt', map_location=torch.device('cpu')))
    try:
        model.epsilon = torch.load('saved_models/trained_closed_3_6_48-48_fine_tuned_old_epsilon.pt.pt', map_location=torch.device('cpu'))
    except:
        pass'''
    model = latent_ode(obs_dim=3, latent_dim=6, nhidden=50,
                        rnn_nhidden=50, lr=3e-3, batch=1080)

    model.load_state_dict(torch.load('saved_models/test_50_50_0.003',
                                     map_location=torch.device('cpu')
                                     ))
    #model.eval()
    model.func.eval()
    model.dec.eval()
    model.rec.eval()

def load(type):
    if type == 'open':
        data = torch.load('saved_datasets/open_6_53_53_0.007.pt')
        model = latent_ode(batch=1080, obs_dim=3, latent_dim=6, nhidden=53, rnn_nhidden=53, lr=0.007, beta=1, extra_decode=True)
        load_model(model, 'open_6_53_53_0.007')
    elif type == 'closed':
        data = torch.load('saved_datasets/closed_50_50_0.003.pt', map_location=torch.device('cpu'))
        model = latent_ode(batch=64, obs_dim=3, latent_dim=6, nhidden=48, rnn_nhidden=48, lr=0.004, extra_decode=True)
        load_model(model, 'closed_6_48_48-0.004')
    elif type == 'two':
        data = torch.load('saved_datasets/two_8_170_170_0.002.pt') 
        model = latent_ode(batch=1080, obs_dim=4, latent_dim=8, nhidden=170, rnn_nhidden=170, lr=0.002, extra_decode=True)
        load_model(model, 'two_8_170_170_0.002')

    return data, model


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.nfe = 0
        self.ode = nn.Sequential(
            nn.Linear(latent_dim, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, latent_dim)

        )

    def forward(self, t, x):
        self.nfe += 1
        result = self.ode(x)
        return result.to(device)


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=6, obs_dim=3, nhidden=48, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1).to(device)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out.to(device), h.to(device)

    def initHidden(self, batch=0):
        if batch == 0:
            return torch.zeros(self.nbatch, self.nhidden).to(device)
        else:
            return torch.zeros(batch, self.nhidden).to(device)


class Decoder(nn.Module):

    def __init__(self, latent_dim=6, obs_dim=3, nhidden=48, extra=False):
        super(Decoder, self).__init__()
        self.extra = extra
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.tanh(out)
        if self.extra:
            out = self.fc2(out)
            out = self.tanh(out)
            out = self.fc3(out)
        else:
            out = self.fc3(out)
        return out.to(device)


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(device)
    const = torch.log(const)
    return -.5 * (const + logvar.to(device) + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1).to(device)
    v2 = torch.exp(lv2).to(device)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class latent_ode(torch.nn.Module):
    def __init__(self, obs_dim=2, latent_dim=4, nhidden=20, rnn_nhidden=25, data=None, lr=1e-2, batch=1000, beta=1,
                 extra_decode=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        self.rnn_nhidden = rnn_nhidden
        self.beta = beta
        self.data = data
        self.epsilon = None
        self.func = LatentODEfunc(latent_dim, nhidden)
        self.rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch)
        self.dec = Decoder(latent_dim, obs_dim, nhidden, extra=extra_decode)
        self.params = (list(self.func.parameters()) + list(self.dec.parameters()) + list(self.rec.parameters()))
        self.lr = lr
        self.optimizer = optim.Adam(self.params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def train(self, trajs, ts, num_epochs):
        # dataset parameters
        num_ts = ts.size(0)
        beta = self.beta
        os.makedirs('./plots/train', exist_ok=True)
        for itr in (range(num_epochs)):
            self.rec.to(device)
            self.func.to(device)
            self.optimizer.zero_grad()
            h = self.rec.initHidden()
            self.rec.to(device)
            for t in reversed(range(num_ts)):
                obs = trajs[:, t, :]
                out, h = self.rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            qz0_mean = qz0_mean.to(device)
            qz0_logvar = qz0_logvar.to(device)
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.to(device)

            # forward in time and solve ode for reconstructions
            pred_z = odeint(self.func, z0, ts).permute(1, 0, 2)
            pred_x = self.dec(pred_z)
            pred_x = pred_x.to(device)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()) + 0.3
            noise_logvar = 2. * torch.log(noise_std_)
            noise_logvar = noise_logvar.to(device)
            logpx = log_normal_pdf(
                trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = beta * normal_kl(qz0_mean, qz0_logvar,
                                           pz0_mean, pz0_logvar).sum(-1)
            lagrange = (1 - pred_x[:, :, 0] ** 2 - pred_x[:, :, 1] ** 2 - pred_x[:, :, 2] ** 2).sum(-1)
            loss = torch.mean(-logpx + analytic_kl + 0.6 * torch.abs(lagrange), dim=0).to(device)
            loss.backward()
            av_mse, *_ = self.MSE(trajs, ts)
            self.optimizer.step()

            if itr == num_epochs:
                self.epsilon = epsilon
            print('Epoch: {}, elbo: {:.4f}, mse: {:.4f}'.format(itr, loss, av_mse))

            if itr >= 4500:
                self.scheduler.step()


    def encode(self, trajs, ts, reconstruct=True):
        if (reconstruct):
            with torch.no_grad():
                num_ts = ts.size(0)
                # sample from trajectorys' approx. posterior
                h = self.rec.initHidden(batch=trajs.shape[0]).to(device)
                for t in reversed(range(num_ts)):
                    obs = trajs[:, t, :]
                    out, h = self.rec.forward(obs, h)
                qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
                z0 = qz0_mean  # self.epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        else:
            with torch.no_grad():
                num_ts = ts.size(0)
                # sample from trajectorys' approx. posterior
                h = self.rec.initHidden(batch=trajs.shape[0])
                for t in reversed(range(num_ts)):
                    obs = trajs[:, t, :]
                    out, h = self.rec.forward(obs, h)
                qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        return z0.to(device)

    def decode(self, z0, ts):
        with torch.no_grad():
            if len(z0.shape) == 1:
                pred_z = odeint(self.func, z0, ts)
            else:
                pred_z = odeint(self.func, z0, ts).permute(1, 0, 2)
            pred_x = self.dec(pred_z)
        return pred_x.to(device)

    def latent_dynamics(self, trajs, enc_ts, dec_ts, recontruct=True):
        z0 = self.encode(trajs, enc_ts, recontruct)
        print('z0', z0.shape)
        with torch.no_grad():
            if len(z0.shape) == 1:
                pred_z = odeint(self.func, z0, dec_ts)
            else:
                pred_z = odeint(self.func, z0, dec_ts).permute(1, 0, 2)
        return pred_z.to(device)

    def MSE(self, trajs, train_ts):
        z0 = self.encode(trajs, train_ts)
        pred_x = self.decode(z0, train_ts)

        mse_errors = np.mean((trajs.cpu().numpy() - pred_x.cpu().numpy()) ** 2, axis=1)
        mse_errors = np.mean(mse_errors, axis=1)
        avg_mse = np.mean(mse_errors)

        return avg_mse, mse_errors
