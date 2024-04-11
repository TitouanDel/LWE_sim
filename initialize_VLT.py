import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../LIFT')

from LIFT.modules.Telescope import Telescope 
from LIFT.modules.Source import Source
from LIFT.modules.Detector import Detector


def GenerateVLT(img_resolution, pupil, source_spectrum, f=8*64, reflectivity=0.385, sampling_time=0.1/20.0, num_samples=10*20, gpu=False):
    #D = 8.0 # [m]
    #pixel_size = 24e-6 # [m]
    #ang_pixel = 12.3 # IRLOS pixel angular size on sky is approximately 12.3 [mas]
    #f = D * 78.0 # pixel_size / ang_pixel * 206264806.71915 # [m]

    sampling_time = 0.1 / 20.0 # [s]
    #exposure_time = sampling_time * num_samples
    #to have exaclty 100000 photons per aperture per second at J_mag=15
    
    tel = Telescope(img_resolution    = img_resolution,
                    pupil             = pupil,
                    diameter          = 8,
                    focalLength       = f,
                    pupilReflectivity = reflectivity, 
                    gpu_flag          = gpu)
    
    det = Detector(pixel_size     = 24e-6,
                    sampling_time = sampling_time,
                    samples       = num_samples,
                    RON           = 0.7,
                    QE            = 1.0)

    det.GPU = True
    det.object = None
    det * tel

    ngs_poly = Source(source_spectrum)
    ngs_poly * tel
    return tel