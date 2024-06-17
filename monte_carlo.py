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

from LIFT.modules.Zernike import Zernike
from LIFT.modules.LIFT import LIFT
from VLT_pupil import CircPupil, PupilVLT
from initialize_VLT import GenerateVLT

sys.path.append('../../DIP')
from DIP.DIP import DIP
from DIP.utils import EarlyStopping
from scipy.ndimage import median_filter
from torch import nn, optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def npy(x):
    if isinstance(x, cp.ndarray):
        return x.get()
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x
#%% Initializing optical system
samples = 80

vlt_pupil = PupilVLT(samples, vangle=[0, 0], petal_modes=False, rotation_angle=0)
# plt.imshow(vlt_pupil)
# plt.show()
petal_modes = PupilVLT(samples, vangle=[0, 0], petal_modes=True, rotation_angle=0)

# Display petal modes
# fig, axes = plt.subplots(3, 4, figsize=(12, 9))

# for i, ax in enumerate(axes.flat):
#     idx = i % 12
    
#     ax.imshow(petal_modes[:, :, idx], cmap='gray')
#     ax.set_title(f'Image {idx+1}')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

vlt = GenerateVLT(img_resolution=41, pupil=vlt_pupil, source_spectrum= [('H', 14)], f=8*64, reflectivity=0.385, sampling_time=0.1/20.0, num_samples=10*20, gpu=True)
#%% Initialize modal basis
modes_num_Z = 32
div_basis = Zernike(modes_num_Z)
div_basis.computeZernike(vlt)
astig_shift = 400e-9 #[m]
astig_diversity = div_basis.Mode(4) * astig_shift

Z_basis = Zernike(10)
Z_basis.computeZernike(vlt)

Z_basis.modesFullRes = cp.concatenate((Z_basis.modesFullRes, cp.asarray(petal_modes)), axis=2)
Z_basis.nModes += 12

def GenerateWF(coefs, diversity=0.0):
    return (Z_basis.wavefrontFromModes(vlt, coefs)+ diversity) * 1e9 # [nm]


dip = DIP(vlt, device, 'sum')
dip.diversity   = torch.atleast_3d(torch.tensor(astig_diversity)*1e9).to(device)
dip.modal_basis = torch.tensor(Z_basis.modesFullRes, dtype=torch.float32, device=device)

#%% Coef distrib
n_psf = n_lift_failed =  n_dip_failed = 0

LO_dist_law = lambda x, A, B, C: A / cp.exp(B*cp.abs(x)) + C
N_modes_simulated = Z_basis.nModes
x = cp.arange(N_modes_simulated)
LO_distribution = LO_dist_law(x, *[70, 0.2, 10])
#LO_distribution[[0,1,10]] = 0
LO_distribution[0:11]=0
LO_distribution[2:10]=20
coeff_tilt = 55
coeff_piston = 150
LO_distribution[14:21] = coeff_tilt
LO_distribution[11:14] = coeff_piston
LO_distribution *= 1e-9 # [nm] -> [m]

#%% MonteCarlo
def PSFfromCoefs(coefs, diversity):
    vlt.src.OPD = Z_basis.wavefrontFromModes(vlt,coefs) + diversity # Don't forget to add the diversity term
    PSF = vlt.ComputePSF()
    vlt.src.OPD *= 0.0 # zero out just in case
    return PSF

def coefs2WF(coefs):
    OPD_all = (dip.modal_basis[:,:, modes_DIP] * coefs.view(1,1,-1)).sum(dim=2, keepdim=True) + dip.diversity
    OPD = (OPD_all).permute([2,0,1])
    return OPD

WFE_intro = [] 
WFE_DIP = []
WFE_LIFT = []
PV = []

for k in range(1):
    print("iteration number:", k, end='\r')

    coefs_LO = cp.random.normal(0, LO_distribution, N_modes_simulated)
    # coefs_LO[11:14] = cp.clip(coefs_LO[11:14], -700e-9, 700e-9)
    # coefs_LO[14:22] = cp.clip(coefs_LO[14:22], -300e-9, 300e-9)
    # coefs_LO = cp.clip(coefs_LO, -500e-9, 500e-9)

    PSF_noiseless = PSFfromCoefs(coefs_LO, astig_diversity)
    PSF_noisy_DITs, _ = vlt.det.getFrame(PSF_noiseless, noise=True, integrate=False) # Adding noise to the PSF and generating a sequence of frames

    PSF_noiseless = npy(PSF_noiseless)

    R_n = PSF_noisy_DITs.var(axis=2)  
    R_n = cp.array(R_n)
    PSF_data = PSF_noisy_DITs.mean(axis=2) # input PSF
    PSF_data = cp.array(PSF_data)


    estimator = LIFT(vlt, Z_basis, astig_diversity, 20)
    
    #Choice of PSF to estimate : noiseless or data
    PSF_sim = PSF_data

    #modes_LIFT = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # Selected Zernike modal coefficients
    modes_LIFT = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    # Note! Sometime, for ML-estimation it is better to exlude an order coincding with the diversity term (e.g. 4th order in this case) to reduce the cross coupling between modes

    coefs_mean = cp.zeros([max(modes_LIFT)+1]) # Initial guess for the mean value of the modal coefficients (for MAP estimator)
    coefs_var  = LO_distribution**2 # Initial guess for the variance of the modal coefficients (for MAP estimator)

    # coefs_LIFT,  PSF_LIFT, _  = estimator.Reconstruct(PSF_sim, R_n=None, mode_ids=modes_LIFT, optimize_norm='sum')
    coefs_LIFT,  PSF_LIFT, _  = estimator.Reconstruct(PSF_sim, R_n=R_n, mode_ids=modes_LIFT, optimize_norm='sum')

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
    WFE_intro.append(WFE_0)
    PV.append(np.round(np.max(WF_0) - np.min(WF_0)))
    


    c_lim = np.max([np.max(np.abs(WF_0)), np.max(np.abs(WF_LIFT)), np.max(np.abs(d_WF))])
    print("LIFT WFE:", d_WFE)
    WFE_LIFT.append(d_WFE)
    if d_WFE>10:
        n_lift_failed +=1
    
    PSF_torch = torch.tensor(PSF_sim/PSF_sim.sum()).float().to(device).unsqueeze(0)

    inv_R_n_torch = median_filter(1.0 / R_n.get(), 4) + 1.0
    inv_R_n_torch = torch.tensor(inv_R_n_torch).float().to(device).unsqueeze(0)

    modes_DIP = modes_LIFT
    num_modes_DIP = len(modes_DIP)

    assert num_modes_DIP <= N_modes_simulated, 'Number of modes to estimate is larger than the number of modes simulated'



    coefs_DIP_defocus = torch.zeros(num_modes_DIP, requires_grad=True, device=device)

    loss_fn = nn.L1Loss(reduction='sum')
    early_stopping = EarlyStopping(patience=5, tolerance=0.00001, relative=False)
    optimizer = optim.LBFGS([coefs_DIP_defocus], history_size=20, max_iter=5, line_search_fn="strong_wolfe")    

    def criterion():
        loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus))*inv_R_n_torch, PSF_torch*inv_R_n_torch)
        #loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus)), PSF_torch)
        return loss # add whatever regularizer you want here
        
    verbose = False

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
        coefs_DIP_defocus = 1e-9*torch.cat((coefs_DIP_defocus[:pos], torch.zeros(1, device=device), coefs_DIP_defocus[pos:]))

    d_WFE_DIP = calc_WFE(d_WF := WF_0-WF_DIP)
    print("DIP WFE:", d_WFE_DIP)
    WFE_DIP.append(d_WFE_DIP)
    if d_WFE_DIP>10:
        n_dip_failed +=1
#%% Display graph
print("LIFT error rate:", n_lift_failed*100/(k+1), "%")
print("DIP error rate:", n_dip_failed*100/(k+1), "%")

print('WFE_avg_LIFT=', np.mean(WFE_LIFT))
print('WFE_avg_DIP=', np.mean(WFE_DIP))

# Tracé des nuages de points
plt.scatter(WFE_intro, WFE_LIFT, label='LIFT')
plt.scatter(WFE_intro, WFE_DIP, label='DIP')

# Ajout de légendes et d'étiquettes
plt.xlabel('Introduced WFE [nm rms]')
plt.ylabel('WFE after reconstruction [nm rms]')
plt.title('Dependency of WFE after estimation on introduced WFE')
plt.legend()
plt.ylim([0, 10])
# Affichage du graphique
plt.show()

plt.scatter(PV, WFE_LIFT, label='LIFT')
plt.scatter(PV, WFE_DIP, label='DIP')

# Ajout de légendes et d'étiquettes
plt.xlabel('Introduced PV [nm rms]')
plt.ylabel('WFE after reconstruction [nm rms]')
plt.title('Dependency of WFE after estimation on introduced PV')
plt.legend()
plt.ylim([0, 10])
# Affichage du graphique
plt.show()
#%% Loop on noise
K=100
mag = [10, 12, 13, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75, 16, 16.25, 16.5, 17]
J = len(mag)

div = [4, 11, 21]
l_div = len(div)

WFE_avg_LIFT = np.zeros((J, l_div, 2))
WFE_avg_DIP = np.zeros((J, l_div, 2))

error_thresh = 20
astig_shift = 600e-9 #[m]

calc_WFE = lambda WF: np.std(WF[vlt_pupil == 1]) if not hasattr(WF, 'get') else cp.std(WF[vlt_pupil == 1])


for j in range(J):
    print("iteration number:", j)
    # mag[j] = 10+(7/(J-1))*j
    vlt = GenerateVLT(img_resolution=41, pupil=vlt_pupil, source_spectrum= [('H', mag[j])], f=8*64, reflectivity=0.385, sampling_time=0.1/20.0, num_samples=10*20, gpu=True)
    n_lift_failed = np.zeros(l_div)
    n_dip_failed = np.zeros(l_div)

    for k in range(K):

        coefs_LO = cp.random.normal(0, LO_distribution, N_modes_simulated)
        # coefs_LO[11:14] = cp.clip(coefs_LO[11:14], -700e-9, 700e-9)
        # coefs_LO[14:22] = cp.clip(coefs_LO[14:22], -300e-9, 300e-9)
        WFE_DIP = np.zeros(l_div)
        WFE_LIFT = np.zeros(l_div)


        for i_div in range(l_div):
            # Diversity
            modes_num_Z = 32
            div_basis = Zernike(modes_num_Z)
            div_basis.computeZernike(vlt)
            astig_diversity = div_basis.Mode(div[i_div]) * astig_shift
            
            # Generate PSF
            PSF_noiseless = PSFfromCoefs(coefs_LO, astig_diversity)
            PSF_noisy_DITs, _ = vlt.det.getFrame(PSF_noiseless, noise=True, integrate=False) # Adding noise to the PSF and generating a sequence of frames

            PSF_noiseless = npy(PSF_noiseless)

            R_n = PSF_noisy_DITs.var(axis=2)  
            R_n = cp.array(R_n)
            PSF_data = PSF_noisy_DITs.mean(axis=2) # input PSF
            PSF_data = cp.array(PSF_data)
            PSF_sim = PSF_data
            
            estimator = LIFT(vlt, Z_basis, astig_diversity, 20)

            #modes_LIFT = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # Selected Zernike modal coefficients
            modes_LIFT = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

            coefs_mean = cp.zeros([max(modes_LIFT)+1]) # Initial guess for the mean value of the modal coefficients (for MAP estimator)
            coefs_var  = LO_distribution**2 # Initial guess for the variance of the modal coefficients (for MAP estimator)

            coefs_LIFT,  PSF_LIFT, _  = estimator.Reconstruct(PSF_sim, R_n=R_n, mode_ids=modes_LIFT, optimize_norm='sum')


            WF_0    = GenerateWF(coefs_LO,   astig_diversity)
            WF_LIFT = GenerateWF(coefs_LIFT, astig_diversity)

            if hasattr(WF_0,     'device'): WF_0     = WF_0.get()
            if hasattr(WF_LIFT,  'device'): WF_LIFT  = WF_LIFT.get()
            if hasattr(PSF_sim,  'device'): PSF_sim  = PSF_sim.get()
            if hasattr(PSF_LIFT, 'device'): PSF_LIFT = PSF_LIFT.get()

            d_WF  = WF_0 - WF_LIFT
            d_WFE = calc_WFE(d_WF)
            WFE_0 = calc_WFE(WF_0)
            WFE_intro.append(WFE_0)
            # PV.append(np.round(np.max(WF_0) - np.min(WF_0)))

            WFE_LIFT[i_div] = d_WFE
            if d_WFE > error_thresh:
                n_lift_failed[i_div] +=1
            
            PSF_torch = torch.tensor(PSF_sim/PSF_sim.sum()).float().to(device).unsqueeze(0)

            inv_R_n_torch = median_filter(1.0 / R_n.get(), 4) + 1.0
            inv_R_n_torch = torch.tensor(inv_R_n_torch).float().to(device).unsqueeze(0)

            modes_DIP = modes_LIFT
            num_modes_DIP = len(modes_DIP)

            assert num_modes_DIP <= N_modes_simulated, 'Number of modes to estimate is larger than the number of modes simulated'

            dip = DIP(vlt, device, 'sum')
            dip.diversity   = torch.atleast_3d(torch.tensor(astig_diversity)*1e9).to(device)
            dip.modal_basis = torch.tensor(Z_basis.modesFullRes, dtype=torch.float32, device=device)

            coefs_DIP_defocus = torch.zeros(num_modes_DIP, requires_grad=True, device=device)

            loss_fn = nn.L1Loss(reduction='sum')
            early_stopping = EarlyStopping(patience=5, tolerance=0.00001, relative=False)
            optimizer = optim.LBFGS([coefs_DIP_defocus], history_size=20, max_iter=5, line_search_fn="strong_wolfe")    

            def criterion():
                loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus))*inv_R_n_torch, PSF_torch*inv_R_n_torch)
                #loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus)), PSF_torch)
                return loss # add whatever regularizer you want here
                
            verbose = False

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
                coefs_DIP_defocus = 1e-9*torch.cat((coefs_DIP_defocus[:pos], torch.zeros(1, device=device), coefs_DIP_defocus[pos:]))

            d_WFE_DIP = calc_WFE(d_WF := WF_0-WF_DIP)
            # print("DIP WFE:", d_WFE_DIP)
            WFE_DIP[i_div] = d_WFE_DIP
            if d_WFE_DIP> error_thresh:
                n_dip_failed[i_div] +=1
            
            if WFE_LIFT[i_div]< error_thresh : WFE_avg_LIFT[j, i_div, 0] += WFE_LIFT[i_div]
            if WFE_DIP[i_div]< error_thresh : WFE_avg_DIP[j, i_div, 0] += WFE_DIP[i_div]

    WFE_avg_LIFT[j, :, 1] = n_lift_failed*100/K
    WFE_avg_DIP[j, :, 1] = n_dip_failed*100/K
        
    WFE_avg_LIFT[j, :, 0] /= K-n_lift_failed
    WFE_avg_DIP[j, :, 0] /= K-n_dip_failed

    
# WFE_avg_LIFT[:,:,0] /= 100 - n_lift_failed
# WFE_avg_DIP[:,:,0] /= 100 - n_dip_failed
# %% Graphs
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for i in range(WFE_avg_LIFT.shape[1]):
    axs[0].plot(mag, WFE_avg_LIFT[:, i, 0], label=f'Z{div[i]}')
axs[0].set_title('LIFT average WFE for each diversity mode')
axs[0].set_xlabel('Magnitude')
axs[0].set_ylabel('Average WFE after LIFT estimation [nm]')
axs[0].legend()
axs[0].grid()

for i in range(WFE_avg_LIFT.shape[1]):
    axs[1].plot(mag, WFE_avg_LIFT[:, i, 1], label=f'Z{div[i]}')
axs[1].set_title('Error rate for each diversity mode')
axs[1].set_xlabel('Magnitude')
axs[1].set_ylabel('Error rate with LIFT estimation [%]')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for i in range(WFE_avg_DIP.shape[1]):
    axs[0].plot(mag, WFE_avg_DIP[:, i, 0], label=f'Z{div[i]}')
axs[0].set_title('DIP average WFE for each diversity mode')
axs[0].set_xlabel('Magnitude')
axs[0].set_ylabel('Average WFE after DIP estimation [nm]')
axs[0].legend()
axs[0].grid()

for i in range(WFE_avg_DIP.shape[1]):
    axs[1].plot(mag, WFE_avg_DIP[:, i, 1], label=f'Z{div[i]}')
axs[1].set_title('Error rate for each diversity mode')
axs[1].set_xlabel('Magnitude')
axs[1].set_ylabel('Error rate with DIP estimation [%]')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
 # %% Loop on NCPA
K=100
mag = 15
ncpa = [5, 10, 30, 50, 70, 90, 110, 130, 150]
J = len(ncpa)
vlt = GenerateVLT(img_resolution=41, pupil=vlt_pupil, source_spectrum= [('H', mag)], f=8*64, reflectivity=0.385, sampling_time=0.1/20.0, num_samples=10*20, gpu=True)

div = [4, 11, 21]
l_div = len(div)

WFE_avg_LIFT = np.zeros((J, l_div, 2))
WFE_avg_DIP = np.zeros((J, l_div, 2))

error_thresh = 20
astig_shift = 400e-9 #[m]

calc_WFE = lambda WF: np.std(WF[vlt_pupil == 1]) if not hasattr(WF, 'get') else cp.std(WF[vlt_pupil == 1])


for j in range(J):
    print("iteration number:", j)

    LO_distribution[2:11]= ncpa[j]*1e-9
    n_lift_failed = np.zeros(l_div)
    n_dip_failed = np.zeros(l_div)

    for k in range(K):

        coefs_LO = cp.random.normal(0, LO_distribution, N_modes_simulated)
        # coefs_LO[11:14] = cp.clip(coefs_LO[11:14], -700e-9, 700e-9)
        # coefs_LO[14:22] = cp.clip(coefs_LO[14:22], -300e-9, 300e-9)
        WFE_DIP = np.zeros(l_div)
        WFE_LIFT = np.zeros(l_div)


        for i_div in range(l_div):
            # Diversity
            modes_num_Z = 32
            div_basis = Zernike(modes_num_Z)
            div_basis.computeZernike(vlt)
            astig_diversity = div_basis.Mode(div[i_div]) * astig_shift
            
            # Generate PSF
            PSF_noiseless = PSFfromCoefs(coefs_LO, astig_diversity)
            PSF_noisy_DITs, _ = vlt.det.getFrame(PSF_noiseless, noise=True, integrate=False) # Adding noise to the PSF and generating a sequence of frames

            PSF_noiseless = npy(PSF_noiseless)

            R_n = PSF_noisy_DITs.var(axis=2)  
            R_n = cp.array(R_n)
            PSF_data = PSF_noisy_DITs.mean(axis=2) # input PSF
            PSF_data = cp.array(PSF_data)
            PSF_sim = PSF_data
            
            estimator = LIFT(vlt, Z_basis, astig_diversity, 20)

            modes_LIFT = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # Selected Zernike modal coefficients
            # modes_LIFT = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

            coefs_mean = cp.zeros([max(modes_LIFT)+1]) # Initial guess for the mean value of the modal coefficients (for MAP estimator)
            coefs_var  = LO_distribution**2 # Initial guess for the variance of the modal coefficients (for MAP estimator)

            coefs_LIFT,  PSF_LIFT, _  = estimator.Reconstruct(PSF_sim, R_n=R_n, mode_ids=modes_LIFT, optimize_norm='sum')


            WF_0    = GenerateWF(coefs_LO,   astig_diversity)
            WF_LIFT = GenerateWF(coefs_LIFT, astig_diversity)

            if hasattr(WF_0,     'device'): WF_0     = WF_0.get()
            if hasattr(WF_LIFT,  'device'): WF_LIFT  = WF_LIFT.get()
            if hasattr(PSF_sim,  'device'): PSF_sim  = PSF_sim.get()
            if hasattr(PSF_LIFT, 'device'): PSF_LIFT = PSF_LIFT.get()

            d_WF  = WF_0 - WF_LIFT
            d_WFE = calc_WFE(d_WF)
            WFE_0 = calc_WFE(WF_0)
            WFE_intro.append(WFE_0)
            # PV.append(np.round(np.max(WF_0) - np.min(WF_0)))

            WFE_LIFT[i_div] = d_WFE
            if d_WFE > error_thresh:
                n_lift_failed[i_div] +=1
            
            PSF_torch = torch.tensor(PSF_sim/PSF_sim.sum()).float().to(device).unsqueeze(0)

            inv_R_n_torch = median_filter(1.0 / R_n.get(), 4) + 1.0
            inv_R_n_torch = torch.tensor(inv_R_n_torch).float().to(device).unsqueeze(0)

            modes_DIP = modes_LIFT
            num_modes_DIP = len(modes_DIP)

            assert num_modes_DIP <= N_modes_simulated, 'Number of modes to estimate is larger than the number of modes simulated'

            dip = DIP(vlt, device, 'sum')
            dip.diversity   = torch.atleast_3d(torch.tensor(astig_diversity)*1e9).to(device)
            dip.modal_basis = torch.tensor(Z_basis.modesFullRes, dtype=torch.float32, device=device)

            coefs_DIP_defocus = torch.zeros(num_modes_DIP, requires_grad=True, device=device)

            loss_fn = nn.L1Loss(reduction='sum')
            early_stopping = EarlyStopping(patience=5, tolerance=0.00001, relative=False)
            optimizer = optim.LBFGS([coefs_DIP_defocus], history_size=20, max_iter=5, line_search_fn="strong_wolfe")    

            def criterion():
                loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus))*inv_R_n_torch, PSF_torch*inv_R_n_torch)
                #loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus)), PSF_torch)
                return loss # add whatever regularizer you want here
                
            verbose = False

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
                coefs_DIP_defocus = 1e-9*torch.cat((coefs_DIP_defocus[:pos], torch.zeros(1, device=device), coefs_DIP_defocus[pos:]))

            d_WFE_DIP = calc_WFE(d_WF := WF_0-WF_DIP)
            # print("DIP WFE:", d_WFE_DIP)
            WFE_DIP[i_div] = d_WFE_DIP
            if d_WFE_DIP> error_thresh:
                n_dip_failed[i_div] +=1
            
            if WFE_LIFT[i_div]< error_thresh : WFE_avg_LIFT[j, i_div, 0] += WFE_LIFT[i_div]
            if WFE_DIP[i_div]< error_thresh : WFE_avg_DIP[j, i_div, 0] += WFE_DIP[i_div]

    WFE_avg_LIFT[j, :, 1] = n_lift_failed*100/K
    WFE_avg_DIP[j, :, 1] = n_dip_failed*100/K
        
    WFE_avg_LIFT[j, :, 0] /= K-n_lift_failed
    WFE_avg_DIP[j, :, 0] /= K-n_dip_failed

 # %%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for i in range(WFE_avg_LIFT.shape[1]):
    axs[0].plot(ncpa, WFE_avg_LIFT[:, i, 0], label=f'Z{div[i]}')
axs[0].set_title('LIFT average WFE for each diversity mode')
axs[0].set_xlabel('STD of NCPA coefficients')
axs[0].set_ylabel('Average WFE after LIFT estimation [nm]')
axs[0].legend()
axs[0].grid()

for i in range(WFE_avg_LIFT.shape[1]):
    axs[1].plot(ncpa, WFE_avg_LIFT[:, i, 1], label=f'Z{div[i]}')
axs[1].set_title('Error rate for each diversity mode')
axs[1].set_xlabel('STD of NCPA coefficients')
axs[1].set_ylabel('Error rate with LIFT estimation [%]')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for i in range(WFE_avg_DIP.shape[1]):
    axs[0].plot(ncpa, WFE_avg_DIP[:, i, 0], label=f'Z{div[i]}')
axs[0].set_title('DIP average WFE for each diversity mode')
axs[0].set_xlabel('STD of NCPA coefficients')
axs[0].set_ylabel('Average WFE after DIP estimation [nm]')
axs[0].legend()
axs[0].grid()

for i in range(WFE_avg_DIP.shape[1]):
    axs[1].plot(ncpa, WFE_avg_DIP[:, i, 1], label=f'Z{div[i]}')
axs[1].set_title('Error rate for each diversity mode')
axs[1].set_xlabel('STD of NCPA coefficients')
axs[1].set_ylabel('Error rate with DIP estimation [%]')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
# %%
K=100
mag = [10, 12, 13, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75, 16, 16.25, 16.5, 17]
J = len(mag)

div = [4, 4]
l_div = len(div)

WFE_avg_LIFT = np.zeros((J, l_div, 2))
WFE_avg_DIP = np.zeros((J, l_div, 2))

error_thresh = 50
astig_shift = 600e-9 #[m]

calc_WFE = lambda WF: np.std(WF[vlt_pupil == 1]) if not hasattr(WF, 'get') else cp.std(WF[vlt_pupil == 1])


for j in range(J):
    print("iteration number:", j)
    # mag[j] = 10+(7/(J-1))*j
    vlt = GenerateVLT(img_resolution=41, pupil=vlt_pupil, source_spectrum= [('H', mag[j])], f=8*64, reflectivity=0.385, sampling_time=0.1/20.0, num_samples=10*20, gpu=True)
    n_lift_failed = np.zeros(l_div)
    n_dip_failed = np.zeros(l_div)

    for k in range(K):

        coefs_LO = cp.random.normal(0, LO_distribution, N_modes_simulated)
        # coefs_LO[11:14] = cp.clip(coefs_LO[11:14], -700e-9, 700e-9)
        # coefs_LO[14:22] = cp.clip(coefs_LO[14:22], -300e-9, 300e-9)
        WFE_DIP = np.zeros(l_div)
        WFE_LIFT = np.zeros(l_div)


        for i_div in range(l_div):
            # Diversity
            modes_num_Z = 32
            div_basis = Zernike(modes_num_Z)
            div_basis.computeZernike(vlt)
            astig_diversity = div_basis.Mode(div[i_div]) * astig_shift
            
            # Generate PSF
            PSF_noiseless = PSFfromCoefs(coefs_LO, astig_diversity)
            PSF_noisy_DITs, _ = vlt.det.getFrame(PSF_noiseless, noise=True, integrate=False) # Adding noise to the PSF and generating a sequence of frames

            PSF_noiseless = npy(PSF_noiseless)

            R_n = PSF_noisy_DITs.var(axis=2)  
            R_n = cp.array(R_n)
            PSF_data = PSF_noisy_DITs.mean(axis=2) # input PSF
            PSF_data = cp.array(PSF_data)
            PSF_sim = PSF_data
            
            estimator = LIFT(vlt, Z_basis, astig_diversity, 20)

            if i_div==0:
                modes_LIFT = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # Selected Zernike modal coefficients
            else:
                modes_LIFT = [2, 3, 4, 5, 6, 7, 8, 9]

            coefs_mean = cp.zeros([max(modes_LIFT)+1]) # Initial guess for the mean value of the modal coefficients (for MAP estimator)
            coefs_var  = LO_distribution**2 # Initial guess for the variance of the modal coefficients (for MAP estimator)

            coefs_LIFT,  PSF_LIFT, _  = estimator.Reconstruct(PSF_sim, R_n=R_n, mode_ids=modes_LIFT, optimize_norm='sum')


            WF_0    = GenerateWF(coefs_LO,   astig_diversity)
            WF_LIFT = GenerateWF(coefs_LIFT, astig_diversity)

            if hasattr(WF_0,     'device'): WF_0     = WF_0.get()
            if hasattr(WF_LIFT,  'device'): WF_LIFT  = WF_LIFT.get()
            if hasattr(PSF_sim,  'device'): PSF_sim  = PSF_sim.get()
            if hasattr(PSF_LIFT, 'device'): PSF_LIFT = PSF_LIFT.get()

            d_WF  = WF_0 - WF_LIFT
            d_WFE = calc_WFE(d_WF)
            WFE_0 = calc_WFE(WF_0)
            WFE_intro.append(WFE_0)
            # PV.append(np.round(np.max(WF_0) - np.min(WF_0)))

            WFE_LIFT[i_div] = d_WFE
            if d_WFE > error_thresh:
                n_lift_failed[i_div] +=1
            
            PSF_torch = torch.tensor(PSF_sim/PSF_sim.sum()).float().to(device).unsqueeze(0)

            inv_R_n_torch = median_filter(1.0 / R_n.get(), 4) + 1.0
            inv_R_n_torch = torch.tensor(inv_R_n_torch).float().to(device).unsqueeze(0)

            modes_DIP = modes_LIFT
            num_modes_DIP = len(modes_DIP)

            assert num_modes_DIP <= N_modes_simulated, 'Number of modes to estimate is larger than the number of modes simulated'

            dip = DIP(vlt, device, 'sum')
            dip.diversity   = torch.atleast_3d(torch.tensor(astig_diversity)*1e9).to(device)
            dip.modal_basis = torch.tensor(Z_basis.modesFullRes, dtype=torch.float32, device=device)

            coefs_DIP_defocus = torch.zeros(num_modes_DIP, requires_grad=True, device=device)

            loss_fn = nn.L1Loss(reduction='sum')
            early_stopping = EarlyStopping(patience=5, tolerance=0.00001, relative=False)
            optimizer = optim.LBFGS([coefs_DIP_defocus], history_size=20, max_iter=5, line_search_fn="strong_wolfe")    

            def criterion():
                loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus))*inv_R_n_torch, PSF_torch*inv_R_n_torch)
                #loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus)), PSF_torch)
                return loss # add whatever regularizer you want here
                
            verbose = False

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
                coefs_DIP_defocus = 1e-9*torch.cat((coefs_DIP_defocus[:pos], torch.zeros(1, device=device), coefs_DIP_defocus[pos:]))

            d_WFE_DIP = calc_WFE(d_WF := WF_0-WF_DIP)
            # print("DIP WFE:", d_WFE_DIP)
            WFE_DIP[i_div] = d_WFE_DIP
            if d_WFE_DIP> error_thresh:
                n_dip_failed[i_div] +=1
            
            if WFE_LIFT[i_div]< error_thresh : WFE_avg_LIFT[j, i_div, 0] += WFE_LIFT[i_div]
            if WFE_DIP[i_div]< error_thresh : WFE_avg_DIP[j, i_div, 0] += WFE_DIP[i_div]

    WFE_avg_LIFT[j, :, 1] = n_lift_failed*100/K
    WFE_avg_DIP[j, :, 1] = n_dip_failed*100/K
        
    WFE_avg_LIFT[j, :, 0] /= K-n_lift_failed
    WFE_avg_DIP[j, :, 0] /= K-n_dip_failed
#%%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))


axs[0].plot(mag, WFE_avg_LIFT[:, 0, 0], label='Petal modes estimated')
axs[0].plot(mag, WFE_avg_LIFT[:, 1, 0], label='Petal modes not estimated')

axs[0].set_title('LIFT average WFE')
axs[0].set_xlabel('Magnitude')
axs[0].set_ylabel('Average WFE after LIFT estimation [nm]')
axs[0].legend()
axs[0].grid()

axs[1].plot(mag, WFE_avg_LIFT[:, 0, 1], label='Petal modes estimated')
axs[1].plot(mag, WFE_avg_LIFT[:, 1, 1], label='Petal modes not estimated')
axs[1].set_title('LIFT Error rate')
axs[1].set_xlabel('Magnitude')
axs[1].set_ylabel('Error rate with LIFT estimation [%]')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(mag, WFE_avg_DIP[:, 0, 0], label='Petal modes estimated')
axs[0].plot(mag, WFE_avg_DIP[:, 1, 0], label='Petal modes not estimated')
axs[0].set_title('DIP average WFE')
axs[0].set_xlabel('Magnitude')
axs[0].set_ylabel('Average WFE after DIP estimation [nm]')
axs[0].legend()
axs[0].grid()

axs[1].plot(mag, WFE_avg_DIP[:, 0, 1], label='Petal modes estimated')
axs[1].plot(mag, WFE_avg_DIP[:, 1, 1], label='Petal modes not estimated')
axs[1].set_title('DIP Error rate')
axs[1].set_xlabel('Magnitude')
axs[1].set_ylabel('Error rate with DIP estimation [%]')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
# %% Monte Carlo for coef variance
coefs_LO = cp.array([ 0.00000000e+00,  0.00000000e+00, -3.81671928e-09, -5.61932618e-09,
       -2.35217670e-08,  1.00989629e-08,  8.25292456e-09,  5.39639890e-09,
       -2.32034918e-08,  2.51097325e-08,  0.00000000e+00,  9.09484647e-08,
       -2.33747369e-08,  3.43749511e-08, -4.14646515e-08,  2.20387174e-08,
       -7.71892242e-08, -2.35462189e-08,  5.44901192e-09,  8.76293396e-08,
        4.47899150e-09,  -1.45222367e-8])
K=200
est_diff = cp.zeros((K,22,2)) #each line is a realisation, each column is a coefficient, first layer is LIFT and second layer is DIP

dip = DIP(vlt, device, 'sum')
dip.diversity   = torch.atleast_3d(torch.tensor(astig_diversity)*1e9).to(device)
dip.modal_basis = torch.tensor(Z_basis.modesFullRes, dtype=torch.float32, device=device)

for k in range(K):
    print("iteration number:", k, end='\r')

    PSF_noisy_DITs, _ = vlt.det.getFrame(PSF_noiseless, noise=True, integrate=False) # Adding noise to the PSF and generating a sequence of frames


    R_n = PSF_noisy_DITs.var(axis=2)  
    R_n = cp.array(R_n)
    PSF_data = PSF_noisy_DITs.mean(axis=2) # input PSF
    PSF_data = cp.array(PSF_data)


    estimator = LIFT(vlt, Z_basis, astig_diversity, 20)
    
    #Choice of PSF to estimate : noiseless or data
    PSF_sim = PSF_data

    modes_LIFT = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # Selected Zernike modal coefficients
    # modes_LIFT = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    # Note! Sometime, for ML-estimation it is better to exlude an order coincding with the diversity term (e.g. 4th order in this case) to reduce the cross coupling between modes

    coefs_mean = cp.zeros([max(modes_LIFT)+1]) # Initial guess for the mean value of the modal coefficients (for MAP estimator)
    coefs_var  = LO_distribution**2 # Initial guess for the variance of the modal coefficients (for MAP estimator)

    # coefs_LIFT,  PSF_LIFT, _  = estimator.Reconstruct(PSF_sim, R_n=None, mode_ids=modes_LIFT, optimize_norm='sum')
    coefs_LIFT,  PSF_LIFT, _  = estimator.Reconstruct(PSF_sim, R_n=R_n, mode_ids=modes_LIFT, optimize_norm='sum')


    est_diff[k,:,0] = cp.array(coefs_LIFT)-coefs_LO
 

    
    PSF_torch = torch.tensor(PSF_sim/PSF_sim.sum()).float().to(device).unsqueeze(0)

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
    early_stopping = EarlyStopping(patience=5, tolerance=0.00001, relative=False)
    optimizer = optim.LBFGS([coefs_DIP_defocus], history_size=20, max_iter=5, line_search_fn="strong_wolfe")    

    def criterion():
        loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus))*inv_R_n_torch, PSF_torch*inv_R_n_torch)
        #loss = loss_fn( dip(OPD=coefs2WF(coefs_DIP_defocus)), PSF_torch)
        return loss # add whatever regularizer you want here
        
    verbose = False

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

    est_diff[k,:,1] = cp.array(npy(coefs_DIP_defocus))*1e-9-coefs_LO

bias_LIFT = cp.mean(est_diff[:,:,0], axis=0)
bias_DIP = cp.mean(est_diff[:,:,1], axis=0)

var_LIFT = cp.var(est_diff[:,:,0], axis=0)
var_DIP = cp.var(est_diff[:,:,1], axis=0)

# %% plots
index = np.arange(len(coefs_LO))
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
plt.xticks(index)
bar_width = 0.3
CRLB = np.array([0, 0, 1694.1804,  7543.5981,  4187.1464,  1345.4589,   638.7708,  1694.4784,
          826.1569, 0,  492.3287, 14383.7024,  4231.9061, 16313.5605,  1793.0408,
         1992.6196,  1242.0047,  1988.6812,  2124.3887,  1124.3003,  2623.9681,
         1624.0494])

axs[0].bar(index - bar_width, var_LIFT.get() * 1e18, label='LIFT', color='orange', alpha=0.6, width=bar_width)
axs[0].bar(index, var_DIP.get() * 1e18, label='DIP', color='purple', alpha=0.6, width=bar_width)
axs[0].bar(index + bar_width, CRLB, label='CRLB', color='black', alpha=0.6, width=bar_width)

axs[0].set_title('Estimation variance of coefficients')
axs[0].set_ylabel('Variance [nm^2]')
axs[0].legend()
axs[0].grid()

axs[1].bar(index - bar_width, bias_LIFT.get() * 1e18, label='LIFT', color='orange', alpha=0.6, width=bar_width)
axs[1].bar(index, bias_DIP.get() * 1e18, label='DIP', color='purple', alpha=0.6, width=bar_width)

axs[1].set_title('Estimation bias of coefficients')
axs[1].set_ylabel('Bias [m]')
axs[1].legend()
axs[1].grid()

axs[2].bar(index[:10], npy(coefs_LO[:10]) * 1e9, label='Low order Zernike coefs',color='blue', alpha=0.6, width=0.8)
axs[2].bar(index[10:], npy(coefs_LO[10:]) * 1e9, label='Petal modes coefs', color='green', alpha=0.6, width=0.8)
axs[2].set_title('Random coefficients')
axs[2].set_xlabel('Mode number')
axs[2].set_ylabel('Coefficient value [nm]')
axs[2].legend()
axs[2].grid(axis='x')

for ax in axs:
    ax.set_xlim(left=1.5, right=21.5)
# Ajuster la mise en page
plt.tight_layout()
plt.show()
