#%%
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../LIFT')

import matplotlib.pyplot as plt
import numpy as np
from warnings import warn
import torch

import cupy as cp

from LIFT.modules.Telescope import Telescope 
from LIFT.modules.Detector import Detector
from LIFT.modules.Source import Source
from LIFT.modules.Zernike import Zernike
from LIFT.modules.LIFT import LIFT
from VLT_pupil import CircPupil, PupilVLT
from initialize_VLT import GenerateVLT
#%% Initializing optical system
samples = 200

vlt_pupil = PupilVLT(samples, vangle=[0, 0], petal_modes=False, rotation_angle=0)
plt.imshow(vlt_pupil)
plt.show()
petal_modes = PupilVLT(samples, vangle=[0, 0], petal_modes=True, rotation_angle=0)

# Display petal modes
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for i, ax in enumerate(axes.flat):
    idx = i % 12
    
    ax.imshow(petal_modes[:, :, idx], cmap='gray')
    ax.set_title(f'Image {idx+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()

vlt = GenerateVLT(img_resolution=41, pupil=vlt_pupil, source_spectrum= [('H', 15.5)], f=8*64, reflectivity=0.385, sampling_time=0.1/20.0, num_samples=10*20, gpu=True)

#%% Initialize modal basis

modes_num_Z = 10
Z_basis = Zernike(modes_num_Z)
Z_basis.computeZernike(vlt)
Z_basis.modesFullRes = cp.concatenate((Z_basis.modesFullRes, cp.asarray(petal_modes)), axis=2)
Z_basis.nModes += 12

astig_shift = 200e-9 #[m]
astig_diversity = Z_basis.Mode(4) * astig_shift

#%% Generating PSF
LO_dist_law = lambda x, A, B, C: A / cp.exp(B*cp.abs(x)) + C
N_modes_simulated = Z_basis.nModes
x = cp.arange(N_modes_simulated)
LO_distribution = LO_dist_law(x, *[70, 0.2, 10])*0
LO_distribution[2:10] = 50
coeff_std = 100
LO_distribution[11:22] = coeff_std
LO_distribution *= 1e-9 # [nm] -> [m]

coefs_LO = cp.random.normal(0, LO_distribution, N_modes_simulated)

def npy(x):
    if isinstance(x, cp.ndarray):
        return x.get()
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x

plt.title('Simulated LO coefficients')

plt.bar(npy(x), npy(coefs_LO)*1e9, label='Random LO coefs')
plt.plot(npy(x), npy(LO_distribution)*1e9, label='STD per mode', color='gray')

plt.xticks(x.get())
plt.xlim(0, N_modes_simulated-1)
plt.grid()
plt.legend()
plt.ylabel('Coefficient value [nm RMS]')
plt.xlabel('Mode number')
plt.show()

def PSFfromCoefs(coefs, diversity):
    vlt.src.OPD = Z_basis.wavefrontFromModes(vlt,coefs) + diversity # Don't forget to add the diversity term
    PSF = vlt.ComputePSF()
    vlt.src.OPD *= 0.0 # zero out just in case
    return PSF

PSF_noiseless = PSFfromCoefs(coefs_LO, astig_diversity)
PSF_noisy_DITs, _ = vlt.det.getFrame(PSF_noiseless, noise=True, integrate=False) # Adding noise to the PSF and generating a sequence of frames

PSF_noiseless = npy(PSF_noiseless)

R_n = PSF_noisy_DITs.var(axis=2)    # LIFT flux-weighting matrix
PSF_data = PSF_noisy_DITs.mean(axis=2) # input PSF
PSF_data = cp.array(PSF_data)
R_n = cp.array(R_n)
#%%
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].set_title('Simulated noiseless PSF')
ax[1].set_title('Simulated noisy PSF (DITs avg.)')
ax[2].set_title(r'Pixels variance map ($R_n$)')
ax[0].imshow(PSF_noiseless)

if hasattr(PSF_data,'device'):
    ax[1].imshow(PSF_data.get())
    ax[2].imshow(R_n.get())
else:
    ax[1].imshow(PSF_data)
    ax[2].imshow(R_n)
plt.show()

#%% LIFT
estimator = LIFT(vlt, Z_basis, astig_diversity, 20)

#Choice of PSF to estimate : noiseless or data
PSF_sim = PSF_data

modes_LIFT = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # Selected Zernike modal coefficients
# Note! Sometime, for ML-estimation it is better to exlude an order coincding with the diversity term (e.g. 4th order in this case) to reduce the cross coupling between modes

coefs_mean = cp.zeros([max(modes_LIFT)+1]) # Initial guess for the mean value of the modal coefficients (for MAP estimator)
coefs_var  = LO_distribution**2 # Initial guess for the variance of the modal coefficients (for MAP estimator)

#coefs_LIFT,  PSF_LIFT, _  = estimator.Reconstruct(PSF_sim, R_n=None, mode_ids=modes_LIFT, optimize_norm='sum')
coefs_LIFT,  PSF_LIFT, _  = estimator.Reconstruct(PSF_sim, R_n=R_n, mode_ids=modes_LIFT, optimize_norm='sum')


def GenerateWF(coefs, diversity=0.0):
    return (Z_basis.wavefrontFromModes(vlt, coefs)+ diversity) * 1e9 # [nm]

calc_WFE = lambda WF: np.std(WF[vlt_pupil == 1]) if not hasattr(WF, 'get') else cp.std(WF[vlt_pupil == 1])

WF_0    = GenerateWF(coefs_LO,   astig_diversity)
WF_LIFT = GenerateWF(coefs_LIFT, astig_diversity)

if hasattr(WF_0,     'device'): WF_0     = WF_0.get()
if hasattr(WF_LIFT,  'device'): WF_LIFT  = WF_LIFT.get()
if hasattr(PSF_sim,  'device'): PSF_sim  = PSF_sim.get()
if hasattr(PSF_LIFT, 'device'): PSF_LIFT = PSF_LIFT.get()

d_WF  = WF_0 - WF_LIFT
d_WFE = calc_WFE(d_WF)
WFE_0 = calc_WFE(WF_0)

c_lim = np.max([np.max(np.abs(WF_0)), np.max(np.abs(WF_LIFT)), np.max(np.abs(d_WF))])

# Compare PSFs
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(PSF_noiseless)
ax[0].set_title('Simulated noiseless PSF')
ax[1].imshow(PSF_LIFT)
ax[1].set_title('Estimated PSF (LIFT)')
ax[0].axis('off')
ax[1].axis('off')
plt.show()

# Compare wavefronts
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(WF_0, vmin=-c_lim, vmax=c_lim, cmap='seismic')
ax[0].set_title('Simulated WF')
ax[1].imshow(WF_LIFT, vmin=-c_lim, vmax=c_lim, cmap='seismic')
ax[1].set_title('Estimated WF')
ax[2].imshow(d_WF, vmin=-c_lim, vmax=c_lim, cmap='seismic')
ax[2].set_title('WF difference')
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
# Add colorbar
cax = fig.add_axes([0.92, 0.27, 0.02, 0.45])
fig.colorbar(ax[2].imshow(d_WF, vmin=-c_lim, vmax=c_lim, cmap='seismic'), cax=cax)
cax.set_ylabel('[nm RMS]')
plt.show()

print(f'Introduced WF: {WFE_0:.2f}, WFE: {d_WFE:.2f} [nm RMS]')

plt.title('LIFT(20) WF coefs estimation (noisy)')
index = np.arange(len(coefs_LO))
bar_width = 0.35
plt.bar(npy(index), npy(coefs_LO)*1e9, bar_width, label='Random LO coefs')
plt.bar(npy(index+bar_width), npy(coefs_LIFT)*1e9, bar_width, label='Estimated coefs', color='red')
plt.plot(npy(x), npy(LO_distribution)*1e9, label='STD per mode', color='gray')

plt.xticks(x.get())
plt.xlim(0, N_modes_simulated-1)
plt.grid()
plt.legend()
plt.ylabel('Coefficient value [nm RMS]')
plt.xlabel('Mode number')
plt.xlim(0, 22)
plt.show()

# LWE_coeff_std.append(coeff_std)
# LWE_WFE_noiseless.append(d_WFE)

# %%
# plt.plot(LWE_coeff_std, LWE_WFE_noiseless, marker='o', linestyle='-')
# plt.xlabel('LWE_coeff_std (nm)')
# plt.ylabel('LWE_WFE_noiseless (nm)')
# plt.grid(True)
# plt.ylim(0.4, 1)
# plt.title('LIFT(20) error on WF estimation (noiseless)')

# plt.show()

#%%DIP imports
sys.path.append('../../DIP')
from DIP.DIP import DIP
from DIP.utils import EarlyStopping


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dip = DIP(vlt, device, 'sum')
dip.diversity   = torch.atleast_3d(torch.tensor(astig_diversity)*1e9).to(device)
dip.modal_basis = torch.tensor(Z_basis.modesFullRes, dtype=torch.float32, device=device)

PSF_torch = torch.tensor(PSF_sim/PSF_sim.sum()).float().to(device).unsqueeze(0)

#%%DIP test
from scipy.ndimage import median_filter
from torch import nn, optim
inv_R_n_torch = median_filter(1.0 / R_n.get(), 4) + 1.0
inv_R_n_torch = torch.tensor(inv_R_n_torch).float().to(device).unsqueeze(0)

modes_DIP = modes_LIFT
num_modes_DIP = len(modes_DIP)

assert num_modes_DIP <= N_modes_simulated, 'Number of modes to estimate is larger than the number of modes simulated'

def coefs2WF(coefs):
    OPD_all = (dip.modal_basis[:,:, modes_DIP] * coefs.view(1,1,-1)).sum(dim=2, keepdim=True) + dip.diversity
    OPD = (OPD_all).permute([2,0,1])
    return OPD

coefs_DIP_defocus = torch.zeros(num_modes_DIP, requires_grad=True, device=device)

loss_fn = nn.L1Loss(reduction='sum')
early_stopping = EarlyStopping(patience=10, tolerance=0.0001, relative=False)
optimizer = optim.LBFGS([coefs_DIP_defocus], history_size=20, max_iter=5, line_search_fn="strong_wolfe")    

def criterion():
    loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus))*inv_R_n_torch, PSF_torch*inv_R_n_torch)
    #loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus)), PSF_torch)
    return loss # add whatever regularizer you want here
    
verbose = True

for i in range(200):
    optimizer.zero_grad()
    loss = criterion() 
    early_stopping(loss)
    loss.backward()
    optimizer.step( lambda: criterion() )
    if verbose: print(f'Loss (iter. {i}/200): {loss.item()}', end='\r')

    if early_stopping.stop:
        if verbose: print('Stopped at it.', i, 'with loss:', loss.item())
        break

with torch.no_grad():       
    PSF_DIP = npy( dip(OPD = (WF_DIP:=coefs2WF(coefs_DIP_defocus))).squeeze() )
    WF_DIP  = npy( WF_DIP.squeeze() )

for pos in [0, 1, 10]:
    coefs_DIP_defocus = torch.cat((coefs_DIP_defocus[:pos], torch.zeros(1, device=device), coefs_DIP_defocus[pos:]))

# Compare PSFs
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(PSF_noiseless)
ax[0].set_title('Simulated noiseless PSF')
ax[1].imshow(PSF_DIP)
ax[1].set_title('Estimated PSF (DIP)')
ax[0].axis('off')
ax[1].axis('off')
plt.show()

d_WFE = calc_WFE(d_WF := WF_0-WF_DIP)

# Compare wavefronts
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(WF_0, vmin=-c_lim, vmax=c_lim, cmap='seismic')
ax[0].set_title('Simulated WF')
ax[1].imshow(WF_DIP, vmin=-c_lim, vmax=c_lim, cmap='seismic')
ax[1].set_title('Estimated WF')
ax[2].imshow(d_WF, vmin=-c_lim, vmax=c_lim, cmap='seismic')
ax[2].set_title('WF difference')
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
# Add colorbar
cax = fig.add_axes([0.92, 0.27, 0.02, 0.45])
fig.colorbar(ax[2].imshow(d_WF, vmin=-c_lim, vmax=c_lim, cmap='seismic'), cax=cax)
cax.set_ylabel('[nm RMS]')
plt.show()

print(f'Introduced WF: {WFE_0:.2f}, WFE: {d_WFE:.2f} [nm RMS]')

plt.title('DIP WF coefs estimation (noisy)')
index = np.arange(22)
bar_width = 0.35
plt.bar(npy(index), npy(coefs_LO)*1e9, bar_width, label='Random LO coefs')
plt.bar(npy(index+bar_width), npy(coefs_DIP_defocus), bar_width, label='Estimated coefs', color='red')
plt.plot(npy(x), npy(LO_distribution)*1e9, label='STD per mode', color='gray')

plt.xticks(x.get())
plt.xlim(0, N_modes_simulated)
plt.grid()
plt.legend()
plt.ylabel('Coefficient value [nm RMS]')
plt.xlabel('Mode number')
plt.show()

mse = np.mean((npy(coefs_LO)*1e9 - npy(coefs_DIP_defocus)) ** 2)
print("MSE entre coefs_LO et coefs_DIP_defocus :", mse, '[nm RMS]')
# %%
