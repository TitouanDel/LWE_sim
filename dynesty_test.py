#%% librairies
# %reload_ext autoreload
# %autoreload 2

import os
import sys
sys.path.append("..")
sys.path.append("../LIFT")

import numpy as np
import cupy as cp
# import pandas as pd
import matplotlib.pyplot as plt
import dynesty
from dynesty.pool import Pool
from dynesty import plotting as dyplot
# from machine_learning.LIFT_dataset_diverse1_5 import GenerateVLT
from VLT_pupil import PupilVLT, CircPupil
from initialize_VLT import GenerateVLT
from DIP.DIP import DIP
from LIFT.modules.Zernike import Zernike
import torch

'''
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.2, 8.1, 10.1])
yerr = np.array([0.1, 0.2, 0.1, 0.3, 0.2])*10

model = lambda x, m, b: m * x + b

# Define the log-likelihood function
def log_likelihood(params):
    m, b, lnf = params
    inv_sigma2 = 1.0 / (yerr**2 + model(x,m,b)**2 * np.exp(2*lnf))
    return -0.5 * (np.sum((y - model(x,m,b))**2 * inv_sigma2 - np.log(inv_sigma2)))

# Define the prior transform
def prior_transform(u):
    m = 5.0 * u[0] - 2
    b = 10.0 * u[1] - 5.0
    lnf = -5.0 * u[2]
    return m, b, lnf


# if __name__ == "__main__":

# with Pool(10, log_likelihood, prior_transform) as pool:
    # sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim=3, pool=pool)
    # sampler.run_nested()

# Set up the nested sampler
sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=3, nlive=1000)
# Run the sampler
sampler.run_nested()

# Get the results
results = sampler.results

# Print the posterior mean and standard deviation of the parameters
print("m = {} +/- {}".format(np.mean(results.samples[:, 0]), np.std(results.samples[:, 0])))
print("b = {} +/- {}".format(np.mean(results.samples[:, 1]), np.std(results.samples[:, 1])))
print("lnf = {} +/- {}".format(np.mean(results.samples[:, 2]), np.std(results.samples[:, 2])))

plt.errorbar(x, y, yerr=yerr, fmt='o')
plt.xlim(x.min(), x.max())

# draw fitted line from posterior samples
for i in range(1000):
    m, b, lnf = results.samples[np.random.randint(len(results.samples))]
    plt.plot(x, model(x,m,b), color='k', alpha=0.01)
plt.xlabel('x')
plt.ylabel('y')
'''

#%% Set up
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

im_size = 41
phase_size = 80 

vlt_pupil = PupilVLT(phase_size, vangle=[0, 0], petal_modes=False, rotation_angle=0)
# plt.imshow(vlt_pupil)
# plt.show()
petal_modes = PupilVLT(phase_size, vangle=[0, 0], petal_modes=True, rotation_angle=0)

tel = GenerateVLT(im_size, pupil=vlt_pupil, source_spectrum= [('H', 14)], f=8*64, reflectivity=0.385, sampling_time=0.1/20.0, num_samples=10*20, gpu=True)

# Basis used for diversity
modes_num_Z = 32
div_basis = Zernike(modes_num_Z)
div_basis.computeZernike(tel)
astig_shift = 400 #e-9
div_mode = 4

# Basis with Zernike + petal modes
Z = Zernike(10)
Z.computeZernike(tel)

computed_modes = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # 
Z.modesFullRes = cp.concatenate((Z.modesFullRes, cp.asarray(petal_modes)), axis=2)
Z.modesFullRes = Z.modesFullRes[:,:,computed_modes]
Z.nModes = 19

modes = torch.tensor(Z.modesFullRes).to(device).double()



# ACHTUNG! In this example PSF is not normalized to 1.0 over the image, but to total number of photons
dip = DIP(tel, device, None) #, 'sum')
dip = dip.double() # double for better precision
dip.diversity   = torch.tensor(div_basis.Mode(div_mode) * astig_shift).to(device)
# Z.modesFullRes = cp.expand_dims(cp.asarray(petal_modes), axis=-1)

dip.modal_basis = torch.tensor(Z.modesFullRes, dtype=torch.float64, device=device)
dip.tel_pupil   = torch.tensor(tel.pupil, dtype=torch.float64, device=dip.device)


def GetOPD(coefs):
    OPD_all = dip.modal_basis[:,:,:coefs.shape[-1]] @ coefs.T + dip.diversity.unsqueeze(-1)
    OPD = (OPD_all).permute([2,0,1])
    return OPD


def model_numpy(params):
    A = torch.tensor(params).unsqueeze(0).to(device).double()
    PSF = dip(OPD=GetOPD(A))
    return PSF.squeeze().cpu().numpy()


def GetNoisyPSF(PSF):
    PSF_noisy, _ = tel.det.getFrame(PSF, noise=True, integrate=False)
    return PSF_noisy.mean(axis=2), PSF_noisy.var(axis=2), PSF_noisy


A_truth = np.array([-3.81671928e-09, -5.61932618e-09,
        -2.35217670e-08,  1.00989629e-08,  8.25292456e-09,  5.39639890e-09,
        -2.32034918e-08,  2.51097325e-08,  9.09484647e-08,-2.33747369e-08,  
        3.43749511e-08, -4.14646515e-08,  2.20387174e-08, -7.71892242e-08, 
        -2.35462189e-08,  5.44901192e-09,  8.76293396e-08, 4.47899150e-09,
        4.45222367e-10])*1e9 

PSF_mean, PSF_var, PSF_cube = GetNoisyPSF(model_numpy(A_truth))
R_n = 1.0 /np.clip(PSF_var, a_min=1e-8, a_max=PSF_var.max()) # Clipping to avoid unadequate values
R_n = R_n.mean()

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].set_title('Simulated noiseless PSF')
ax[1].set_title('Simulated noisy PSF (DITs avg.)') 
ax[2].set_title(r'Pixels variance map ($R_n$)')
ax[0].imshow(model_numpy(A_truth))

if hasattr(PSF_mean,'device'):
    ax[1].imshow(PSF_mean.get())
    ax[2].imshow(PSF_var.get())
else:
    ax[1].imshow(PSF_mean)
    ax[2].imshow(PSF_var)
plt.show()

'''
# The same log-likelihood, but with the heteroscedasticy parameter included
def log_likelihood(params):
    x, lnf = params[:-1], params[-1]
    inv_sigma2 = 1.0 / (Y_err**2 + model(*x)**2 * np.exp(2 * lnf))
    return -0.5 * (np.sum( (Y-model(*x))**2 * inv_sigma2 - np.log(inv_sigma2) ))

# Define the prior transform
def prior_transform(u):
    a   = 300.0 * u[:-1] - 150.0
    lnf = -5.0  * u[-1]
    return *a, lnf #lnf is the heteroscedastic noise parameter, may be excluded
'''

def log_likelihood(model, x, Y, inv_sigma2):
    # inv_sigma2 = 1.0 / (Y_err**2)
    # just to make this function universal for both frameworks
    log_func = torch.log if isinstance(Y, torch.Tensor) else np.log
    return ( -0.5 * inv_sigma2 * (Y-model(x))**2 - log_func(inv_sigma2/(2*np.pi)) ).sum()

# Define the prior transform, it maps an arbitrary range (here -150...150 [nm]) of values to -1...1
def prior_transform(u):
    return 300.0*u - 150.0

    
#%% nested sampling
# lslice
# lwalk

print("Starting nested sampling...")

sampler = dynesty.NestedSampler(
    # loglikelihood   = lambda x: log_likelihood(model_numpy, x, model_numpy(A_truth), np.ones_like(PSF_mean)),
    loglikelihood   = lambda x: log_likelihood(model_numpy, x, PSF_mean,  R_n),
    prior_transform = prior_transform,
    ndim            = len(A_truth),
    nlive           = 2000
)

sampler.run_nested()
results = sampler.results


# with Pool(15, log_likelihood, prior_transform) as pool:
#     sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim=6, pool=pool)
#     sampler.run_nested()

#%% plot
# labels = [Z.modeName(i) for i in range(len(A_truth))] #+ ['lnf']
labels = [f'Petal {computed_modes[i]}' for i in range(len(A_truth))] #+ ['lnf']

# initialize figure
# fig, axes = plt.subplots(len(A_truth), len(A_truth), figsize=(10, 10))

# fg, ax = dyplot.cornerpoints(results, cmap='plasma', kde=False, truths=[*A_truth, 0], fig=(fig, axes), labels=labels)
fg, ax = dyplot.cornerpoints(results, cmap='plasma', kde=False, truths=A_truth, labels=labels)


'''
# initialize figure
fig, axes = plt.subplots(6, 6, figsize=(10, 10))
# axes = axes.reshape((5, 5))  # reshape axes

# plot initial run (res1; left)
fg, ax = dyplot.cornerplot(res1, color='blue', truths=[*A, 0],
                           truth_color='black', show_titles=True,
                           smooth=0.03, quantiles=[0.16, 0.5, 0.84],
                           labels=labels,
                           fig=(fig, axes))


'''
# %% Compute Fischer Information Matrix
from torch.autograd.functional import hessian, jacobian

model_torch = lambda params: dip(OPD=GetOPD(params))

# Redifine all these quantities so that PyTorch can understand
A_truth_torch  = torch.tensor(A_truth).unsqueeze(0).to(device).double()
R_n_torch      = torch.tensor(R_n).to(device).double()
PSF_cube_torch = torch.tensor(PSF_cube).to(device).double()

H_1 = []
H_2 = []
for i in range(PSF_cube.shape[-2]):
    data = torch.tensor(PSF_cube[...,i]).to(device).double().unsqueeze(0)
    H_1.append(-hessian(lambda x: log_likelihood(model_torch, x, data, R_n_torch), A_truth_torch).squeeze())
    
H_1 = torch.stack(H_1).mean(axis=0).cpu()
H_1_np = H_1.numpy() #units of FIM are [nm^-2] ?

plt.imshow(H_1_np, vmin=-np.abs(H_1).max(), vmax=np.abs(H_1).max(), cmap='coolwarm')
plt.colorbar()
plt.show()

J = torch.inverse(H_1)
CRLB = torch.diag(J)
print("CRLB=", CRLB.numpy())
# %%
