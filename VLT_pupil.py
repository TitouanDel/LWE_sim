# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:07:00 2021

@author: akuznets
"""

import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


def CircPupil(samples, D=8.0, centralObstruction=1.12):
    x      = np.linspace(-1/2, 1/2, samples)*D
    xx,yy  = np.meshgrid(x,x)
    circle = np.sqrt(xx**2 + yy**2)
    obs    = circle >= centralObstruction/2
    pupil  = circle < D/2 
    return pupil * obs


def PupilVLT(samples, vangle=[0, 0], petal_modes=False, rotation_angle=0):
    pupil_diameter = 8.0
    secondary_diameter = 1.12
    alpha = 101
    spider_width = 0.039

    # Calculate shift of the obscuration
    rad_vangle = np.deg2rad(vangle[0]/60)
    shx = np.cos(np.deg2rad(vangle[1])) * 101.4 * np.tan(rad_vangle)
    shy = np.sin(np.deg2rad(vangle[1])) * 101.4 * np.tan(rad_vangle)

    # Create coordinate matrices
    delta = pupil_diameter / samples
    ext = 2 * np.max(np.abs([shx, shy])) + 1
    grid_range = np.arange(-(pupil_diameter + ext - 2 * delta) / 2, (pupil_diameter + ext) / 2 + delta, delta)
    x, y = np.meshgrid(grid_range, grid_range)

    # Mask for pupil and central obstruction
    mask = (np.sqrt((x - shx)**2 + (y - shy)**2) <= pupil_diameter / 2) & (np.sqrt(x**2 + y**2) >= secondary_diameter / 2)

    # Rotation function
    def rotate(x, y, angle):
        rad_angle = np.deg2rad(angle)
        x_rot = x * np.cos(rad_angle) - y * np.sin(rad_angle)
        y_rot = x * np.sin(rad_angle) + y * np.cos(rad_angle)
        return x_rot, y_rot

    # Rotate coordinates
    x_rot, y_rot = rotate(x, y, rotation_angle)

    # Function to create spider petals
    def create_petal(condition):
        petal = np.zeros_like(x_rot, dtype=bool)
        petal[condition] = True
        return petal & mask

    # Calculate spider petals with rotation
    alpha_rad = np.deg2rad(alpha)
    slope = np.tan(alpha_rad / 2)
    
    petal_conditions = [
        np.where(
            (( -y_rot > spider_width/2 + slope*(-x_rot - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x_rot<0)  & (y_rot<=0)) | \
            (( -y_rot > spider_width/2 + slope*( x_rot - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x_rot>=0) & (y_rot<=0))
        ),
        np.where(
            (( -y_rot < spider_width/2 + slope*( x_rot - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x_rot>0) & (y_rot<=0)) | \
            ((  y_rot < spider_width/2 + slope*( x_rot - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x_rot>0) & (y_rot>0))
        ),
        np.where(
            ((  y_rot > spider_width/2 + slope*(-x_rot - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x_rot<=0) & (y_rot>0)) | \
            ((  y_rot > spider_width/2 + slope*( x_rot - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x_rot>0)  & (y_rot>0))
        ),
        np.where(
            (( -y_rot < spider_width/2 + slope*(-x_rot - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x_rot<0) & (y_rot<0)) |\
            ((  y_rot < spider_width/2 + slope*(-x_rot - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x_rot<0) & (y_rot>=0))
        )
    ]
    petals = [create_petal(condition) for condition in petal_conditions]

    # Resize petals
    lim_x = [(np.fix((shy + ext / 2) / delta)).astype(int), (-np.fix((-shy + ext / 2) / delta)).astype(int)]
    lim_y = [(np.fix((shx + ext / 2) / delta)).astype(int), (-np.fix((-shx + ext / 2) / delta)).astype(int)]
    resized_petals = [resize(petal[lim_x[0]:lim_x[1], lim_y[0]:lim_y[1]], (samples, samples), anti_aliasing=False) for petal in petals]

    if petal_modes:
        limits = [(( -0.5,  0.5 ),  (-0.25, 0.75)),
                  (( -0.75, 0.25),  (-0.5,  0.5 )),
                  (( -0.5,  0.5 ),  (-0.75, 0.25)),
                  (( -0.25, 0.75),  (-0.5,  0.5 ))]

        def normalize_petal_mode(petal, coord, full_pupil):
            mode = petal.astype('double') * coord
            mode -= mode.min()
            mode /= (mode.max() + mode.min())
            mode -= 0.5
            mode[np.where(petal==False)] = 0.0
            mode[np.where(petal==True)] -= mode[np.where(petal==True)].mean()
            # mode /= mode[np.where(petal==True)].std()
            mode /= mode[np.where(full_pupil==True)].std()
            return mode

        tips, tilts  = [], []
  
        for i in range(4):
            xx, yy = np.meshgrid(np.linspace(*limits[i][0],  samples), np.linspace(*limits[i][1], samples))
            xx_rot, yy_rot = rotate(xx, yy, rotation_angle)
            tips.append(  normalize_petal_mode(resized_petals[i], yy_rot, sum(resized_petals)) )
            tilts.append( normalize_petal_mode(resized_petals[i], xx_rot, sum(resized_petals)) )

        return np.dstack([*resized_petals, *tips, *tilts])

    else:
        return sum(resized_petals)
