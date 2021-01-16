
import torchvision.io
import av
#from scipy.fftpack import fftshift, ifftshift
from phasepack.tools import rayleighmode as _rayleighmode
from phasepack.tools import lowpassfilter as _lowpassfilter
from phasepack.filtergrid import filtergrid

# Try and use the faster Fourier transform functions from the pyfftw module if
# available
from phasepack.tools import fft2, ifft2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import torch

def normalise(img):
  return (img - img.min())/(img.max() - img.min())

def integrated_backscatter_energy(img): #img is numpy image with 1 channel
  ibs= np.cumsum(img ** 2,0)
  return ibs

def indices(i, rows):
  ret = np.zeros((rows-i+1,))
  for i in range(ret.shape[0]):
    ret[i] = ret[i] + i
  return ret

#print(indices(1,3))

def shadow(img):
  rows = img.shape[0]
  cols = img.shape[1]
  stdImg = round(rows/4)
  sh = np.zeros_like(img)

  for j in range(cols):
    for i in range(rows):
        gaussWin= np.exp(-((indices(i+1,rows))**2)/(2*(stdImg**2)))
        #print(gaussWin)
        sh[i,j] = np.sum(np.multiply(img[i:rows,j], np.transpose(gaussWin)) / np.sum(gaussWin))
        #print(sh[i,j])
        
  return sh

def analyticEstimator(img, nscale=5, minWaveLength=10, mult=2.1, sigmaOnf=0.55, k=2.,\
                 polarity=0, noiseMethod=-1):

    if img.dtype not in ['float32', 'float64']:
        img = np.float64(img)
        imgdtype = 'float64'
    else:
        imgdtype = img.dtype

    if img.ndim == 3:
        img = img.mean(2)
    rows, cols = img.shape

    epsilon = 1E-4  # used to prevent /0.
    IM = fft2(img)  # Fourier transformed image

    zeromat = np.zeros((rows, cols), dtype=imgdtype)

    # Matrix for accumulating weighted phase congruency values (energy).
    totalEnergy = zeromat.copy()

    # Matrix for accumulating filter response amplitude values.
    sumAn = zeromat.copy()

    radius, u1, u2 = filtergrid(rows, cols)

    # Get rid of the 0 radius value at the 0 frequency point (at top-left
    # corner after fftshift) so that taking the log of the radius will not
    # cause trouble.
    radius[0, 0] = 1.

    H = (1j * u1 - u2) / radius


    lp = _lowpassfilter([rows, cols], .4, 10)
    # Radius .4, 'sharpness' 10
    logGaborDenom = 2. * np.log(sigmaOnf) ** 2.

    for ss in range(nscale):
        wavelength = minWaveLength * mult ** ss
        fo = 1. / wavelength  # Centre frequency of filter

        logRadOverFo = np.log(radius / fo)
        logGabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        logGabor *= lp      # Apply the low-pass filter
        logGabor[0, 0] = 0.  # Undo the radius fudge

        IMF = IM * logGabor   # Frequency bandpassed image
        f = np.real(ifft2(IMF))  # Spatially bandpassed image

        # Bandpassed monogenic filtering, real part of h contains convolution
        # result with h1, imaginary part contains convolution result with h2.
        h = ifft2(IMF * H)

        # Squared amplitude of the h1 and h2 filters
        hAmp2 = h.real * h.real + h.imag * h.imag

        # Magnitude of energy
        sumAn += np.sqrt(f * f + hAmp2)

        # At the smallest scale estimate noise characteristics from the
        # distribution of the filter amplitude responses stored in sumAn. tau
        # is the Rayleigh parameter that is used to describe the distribution.
        if ss == 0:
            # Use median to estimate noise statistics
            if noiseMethod == -1:
                tau = np.median(sumAn.flatten()) / np.sqrt(np.log(4))

            # Use the mode to estimate noise statistics
            elif noiseMethod == -2:
                tau = _rayleighmode(sumAn.flatten())

        # Calculate the phase symmetry measure

        # look for 'white' and 'black' spots
        if polarity == 0:
            totalEnergy += np.abs(f) - np.sqrt(hAmp2)

        # just look for 'white' spots
        elif polarity == 1:
            totalEnergy += f - np.sqrt(hAmp2)

        # just look for 'black' spots
        elif polarity == -1:
            totalEnergy += -f - np.sqrt(hAmp2)


    if noiseMethod >= 0:
        T = noiseMethod


    else:
        totalTau = tau * (1. - (1. / mult) ** nscale) / (1. - (1. / mult))

        # Calculate mean and std dev from tau using fixed relationship
        # between these parameters and tau. See
        # <http://mathworld.wolfram.com/RayleighDistribution.html>
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2.)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2.)

        # Noise threshold, must be >= epsilon
        T = np.maximum(EstNoiseEnergyMean + k * EstNoiseEnergySigma,
                       epsilon)
    #print(totalEnergy,'!!!!!!!!!\n')
    phaseSym = np.maximum(totalEnergy - T, 0)
    #print(phaseSym,'||||||||||||\n')
    phaseSym /= sumAn + epsilon

    #print(type(f), f.shape, f)
    #print(type(hAmp2), hAmp2.shape, hAmp2)

    LP = (1 - np.arctan2(np.sqrt(hAmp2),f))
    FS = phaseSym  #????????????
    LE = (hAmp2 + f*f)

    return LP, FS, LE  #, totalEnergy, T

def bone_prob_map(img, minwl = 10):
  #sh = normalise(shadow(img))
  ibs = normalise(integrated_backscatter_energy(img))
  #shibs = normalise(np.multiply(ibs, sh))
  #shibs = shibs * (shibs >= shibs.mean())
  LP,FS,LE = analyticEstimator(normalise(img) ** 4, minWaveLength = minwl)

  #final = normalise( sh * (1-ibs) * LE * FS )
  final = normalise( normalise(LP) * normalise(FS) * (1-ibs))
  #meanFinal = (final*(final > 0)).mean()
  #final = final * (final > 1.5*meanFinal)
  return final

"""
Assume that this class generates pairs of adjacents frames (not necessarily consecutive,
depending on 'sample_rate' variable) of US video sequences 
(with similar visual qualities, due to them being from the same video as well as same jittering applied to both........) 

"""

class USDataset(Dataset):
  '''
  root: Folder in which LUS videos are located
  sample_rate: The gap between the sampled 'adjacent' LUS images in terms of frame, i.e, 
              if sample_rate = 4 and video is of 25 FPS and length 10 seconds, this Dataset
              might give a tuple of 200th & 204th frame as a sample 
  '''
  def __init__ (self, root , train=True, sample_rate = 1, infer=False):
    #super().__init__()
    self.root = root
    #self.imgs = os.listdir(root + '/Images/')
    self.sample_rate = sample_rate
    self.infer = infer
    
    if train is True:
      self.transform = transforms.Compose([
                                    transforms.ToPILImage(), 
                                    transforms.Resize((256,256)),
    #TODO: To test whether used jittering configurations have exaggerated effects of LUS images
                                    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),

                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x + 0.01*torch.randn_like(x)),
                                    transforms.Lambda(lambda x: torch.clamp(x,min=0,max=1))
                                    #transforms.Normalize(0.5,0.5)
                                    ])

    else:
      self.transform = transforms.Compose([
                                           transforms.ToPILImage(), 
                                           transforms.Resize((256,256)),
                                          transforms.ToTensor(), 
                                          #transforms.Normalize(0.5, 0.5)
                                        ])

    self.hflip = transforms.RandomHorizontalFlip(p=1)


  def __len__(self):
    return 1024

  def __getitem__(self, idx):
    
    '''
    Assuming that 'root' contains only LUS videos and .txt files
    '''
    #np.random.seed()
    vid_name = np.random.choice(os.listdir(self.root))
    while '.txt' in vid_name or '.gif' in vid_name or vid_name == '.config':
      vid_name = np.random.choice(os.listdir(self.root))


    num_sec, fps = torchvision.io.read_video_timestamps(filename = self.root + vid_name, pts_unit = 'pts')
    #currently, num_sec is the total number of frames
    if num_sec == None or fps == None:
      print('None generated by the file/video:', vid_name,'!!!!!\n')
    '''
    if self.infer == True:  
      print("vid name = ",vid_name)
      print("num_sec = ", num_sec)
      print("len(num_sec) =", len(num_sec))
      print("FPS = ", fps)
    '''
    extra_sec = len(num_sec) % fps
    num_sec = int(len(num_sec)/fps)   #TODO: gave division with 'None' value at a random point during training
    #if self.infer == True:
    #  print("Corrected num_sec = ",num_sec)
    
    time_window = np.random.randint(0,num_sec)

    frames, _, _ = torchvision.io.read_video(filename = self.root + vid_name, start_pts = time_window , end_pts = time_window + 1.0 , pts_unit='sec')

    #if self.infer == True:
    #  print("frames' shape = ", frames.shape)

    #if frames.shape[0] < self.sample_rate:
    #  print("Exception!")

    #To choose the frames in the 1 second window, i.e, if the video is of 25 FPS, then 'frames' tensor above would have 25 channels.
    #Can also use the 'fps' variable instead of 'frames.shape[0]'
    frame_stamp = np.random.randint(0,frames.shape[0] - self.sample_rate)  

    img_1 = self.transform(frames[frame_stamp].clone().detach().permute(2,0,1))
    img_2 = self.transform(frames[frame_stamp + self.sample_rate].clone().detach().permute(2,0,1))


    img_1 = img_1[0].unsqueeze(0).repeat_interleave(10,0)
    img_2 = img_2[0].unsqueeze(0).repeat_interleave(10,0)

    #print(img_1.shape, img_2.shape)
    
    #ta = time.time()
    for i in range(img_1.shape[0]):
      #img_1[i] = torch.Tensor(bone_prob_map(img_1[i].numpy(), minwl = 5*(i+1))) * img_1[i]
      #img_2[i] = torch.Tensor(bone_prob_map(img_2[i].numpy(), minwl = 5*(i+1))) * img_2[i]
      img_1[i] = torch.Tensor(bone_prob_map(img_1[i].numpy(), minwl = 3 + 3*i)) * img_1[i]
      img_2[i] = torch.Tensor(bone_prob_map(img_2[i].numpy(), minwl = 3 + 3*i)) * img_2[i]
    del frames
    return img_1, img_2
