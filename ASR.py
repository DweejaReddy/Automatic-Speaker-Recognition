from __future__ import division
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift, dct
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io.wavfile import read
import warnings
np.seterr(all='ignore')
def hertz_to_mel(freq):
 return 1125*np.log(1 + freq/700)

def mel_to_hertz(m):
 return 700*(np.exp(m/1125) - 1)

#calculate mel frequency filter bank
def mel_filterbank(nfft, nfiltbank, fs):
  #set limits of mel scale from 300Hz to 8000Hz
  lower_mel = hertz_to_mel(300)
  upper_mel = hertz_to_mel(8000)
  mel = np.linspace(lower_mel, upper_mel, nfiltbank+2)
  hertz = [mel_to_hertz(m) for m in mel]
  fbins = [int(hz * (nfft/2+1)/fs) for hz in hertz]
  fbank = np.empty((nfft//2+1,nfiltbank))
  for i in range(1,nfiltbank+1):
    for k in range(int(nfft/2 + 1)):
      if k < fbins[i-1]:
        fbank[k, i-1] = 0
      elif k >= fbins[i-1] and k < fbins[i]:
        fbank[k,i-1] = (k - fbins[i-1])/(fbins[i] - fbins[i-1])
      elif k >= fbins[i] and k <= fbins[i+1]:
        fbank[k,i-1] = (fbins[i+1] - k)/(fbins[i+1] - fbins[i])
      else:
        fbank[k,i-1] = 0
  return fbank

def mfcc(s,fs, nfiltbank):

  #divide into segments of 25 ms with overlap of 10ms
  nSamples = np.int32(0.025*fs)
  overlap = np.int32(0.01*fs)
  nFrames = np.int32(np.ceil(len(s)/(nSamples-overlap)))
  #zero padding to make signal length long enough to have nFrames
  padding = ((nSamples-overlap)*nFrames) - len(s)
  if padding > 0:
    signal = np.append(s, np.zeros(padding))
  else:
    signal = s
  segment = np.empty((nSamples, nFrames))
  start = 0
  for i in range(nFrames):
    segment[:,i] = signal[start:start+nSamples]
    start = (nSamples-overlap)*i
  #compute periodogram
  nfft = 512
  periodogram = np.empty((nFrames,nfft//2 + 1))
  for i in range(nFrames):
    x = segment[:,i] * hamming(nSamples)
    spectrum = fftshift(fft(x,nfft))
    periodogram[i,:] = abs(spectrum[nfft//2-1:])/nSamples
  #calculating mfccs
  fbank = mel_filterbank(nfft, nfiltbank, fs)
  #nfiltbank MFCCs for each frame
  mel_coeff = np.empty((nfiltbank, nFrames))
  for i in range(nfiltbank):
    for k in range(nFrames):
      mel_coeff[i, k] = np.sum((periodogram[k,:]*fbank[:,i]))
  mel_coeff = np.log10(mel_coeff)
  mel_coeff = dct(mel_coeff)
  #exclude 0th order coefficient (much larger than others)
  mel_coeff[0,:]= np.zeros(nFrames)
  return mel_coeff

#LPC starts
def autocorr(x):
  n = len(x)
  variance = np.var(x)
  x = x - np.nanmean(x)
# print(x)
  #n numbers from last index
  r = np.correlate(x, x, mode = 'full')[-n:]
  result = r/(variance*(np.arange(n, 0, -1)))
  return result


def createSymmetricMatrix(acf,p):
  R = np.empty((p,p))
  for i in range(p):
    for j in range(p):
      R[i,j] = acf[np.abs(i-j)]
      R = np.nan_to_num(R)
    return R

def lpc(s,fs,p):
  #divide into segments of 25 ms with overlap of 10ms
  nSamples = np.int32(0.025*fs)
  overlap = np.int32(0.01*fs)
  nFrames = np.int32(np.ceil(len(s)/(nSamples-overlap)))
  #zero padding to make signal length long enough to have nFrames
  padding = ((nSamples-overlap)*nFrames) - len(s)
  if padding > 0:
    signal = np.append(s, np.zeros(padding))
  else:
    signal = s
  segment = np.empty((nSamples, nFrames))
  start = 0
  for i in range(nFrames):
    segment[:,i] = signal[start:start+nSamples]
    start = (nSamples-overlap)*i

 #calculating LPC with Yule-Walker
  lpc_coeffs = np.empty((p, nFrames))
  for i in range(nFrames):
    acf = autocorr(segment[:,i])
    r = -acf[1:p+1].T
    R = createSymmetricMatrix(acf,p)
    R = np.round(R, decimals=1)
    R= np.nan_to_num(R)
    # print(R)
    lpc_coeffs[:,i] = np.dot(np.linalg.pinv(R),r)
    lpc_coeffs[:,i] = lpc_coeffs[:,i]/np.max(np.abs(lpc_coeffs[:,i]))
  return lpc_coeffs

def EUDistance(d,c):
  # np.shape(d)[0] = np.shape(c)[0]
  n = np.shape(d)[1]
  p = np.shape(c)[1]
  distance = np.empty((n,p))

  if n<p:
    for i in range(n):
      copies = np.transpose(np.tile(d[:,i], (p, 1)))
      distance[i,:] = np.sum((copies - c)**2,0)
  else:
    for i in range(p):
      copies = np.transpose(np.tile(c[:,i],(n,1)))
      distance[:,i] = np.transpose(np.sum((d - copies)**2,0))
  distance = np.sqrt(distance)
  return distance

def lbg(features, M):
  eps = 0.01
  codebook = np.mean(features, 1)
  distortion = 1
  nCentroid = 1
  while nCentroid < M:
    #double the size of codebook
    new_codebook = np.empty((len(codebook), nCentroid*2))
    if nCentroid == 1:
      new_codebook[:,0] = codebook*(1+eps)
      new_codebook[:,1] = codebook*(1-eps)
    else:
      for i in range(nCentroid):
        new_codebook[:,2*i] = codebook[:,i] * (1+eps)
        new_codebook[:,2*i+1] = codebook[:,i] * (1-eps)

    codebook = new_codebook
    nCentroid = np.shape(codebook)[1]
    D = EUDistance(features, codebook)

    while np.abs(distortion) > eps:
      #nearest neighbour search
      prev_distance = np.mean(D)
      nearest_codebook = np.argmin(D,axis = 1)

      #cluster vectors and find new centroid
      for i in range(nCentroid):
        #add along 3rd dimension
        codebook[:,i] = np.mean(features[:,np.where(nearest_codebook == i)], 2).T

      #replace all NaN values with 0
      codebook = np.nan_to_num(codebook)
      D = EUDistance(features, codebook)
      distortion = (prev_distance - np.mean(D))/prev_distance
      #print 'distortion' , distortion
  #print 'final codebook', codebook, np.shape(codebook)
  return codebook



def training(nfiltbank, orderLPC):
  nSpeaker = 8
  nCentroid = 16
  codebooks_mfcc = np.empty((nSpeaker,nfiltbank,nCentroid))
  codebooks_lpc = np.empty((nSpeaker, orderLPC, nCentroid))
  directory = r'C:\Users\Dweeja Reddy\Downloads\sm_dataset\test'
  fname = str()
  for i in range(nSpeaker):
    fname = '\s' + str(i+1) + '.wav'
    print('Now speaker ', str(i+1), 'features are being trained')
    (fs,s) = read(directory + fname)
    mel_coeff = mfcc(s, fs, nfiltbank)
    lpc_coeff = lpc(s, fs, orderLPC)
    codebooks_mfcc[i,:,:] = lbg(mel_coeff, nCentroid)
    codebooks_lpc[i,:,:] = lbg(lpc_coeff, nCentroid)
    # plt.figure(i)
    plt.title('Codebook for speaker ' + str(i+1) + ' with ' + str(nCentroid) + ' centroids')
    for j in range(nCentroid):
      plt.subplot(211)
      plt.stem(codebooks_mfcc[i,:,j])
      plt.ylabel('MFCC')
      plt.subplot(212)
      markerline, stemlines, baseline = plt.stem(codebooks_lpc[i,:,j])
      plt.setp(markerline,'markerfacecolor','r')
      plt.setp(baseline,'color', 'k')
      plt.ylabel('LPC')
      plt.axis(ymin = -1, ymax = 1)
      plt.xlabel('Number of features')
  plt.show()
  print ('Training completed')
  #plotting 5th and 6th dimension MFCC features on a 2D plane
  codebooks = np.empty((2, nfiltbank, nCentroid))
  mel_coeff = np.empty((2, nfiltbank, 68))
  for i in range(2):
    fname = '/s' + str(i+1) + '.wav'
    (fs,s) = read(directory + fname)
    mel_coeff[i,:,:] = mfcc(s, fs, nfiltbank)[:,0:68]
    codebooks[i,:,:] = lbg(mel_coeff[i,:,:], nCentroid)
  plt.figure(nSpeaker + 1)
  s1 = plt.scatter(mel_coeff[0,4,:], mel_coeff[0,5,:], s = 100, color = 'r', marker = 'o')
  c1 = plt.scatter(codebooks[0,4,:], codebooks[1,5,:], s = 100, color = 'r', marker = '+')
  s2 = plt.scatter(mel_coeff[1,4,:], mel_coeff[1,5,:], s = 100, color = 'b', marker = 'o')
  c2 = plt.scatter(codebooks[1,4,:], codebooks[1,5,:], s = 100, color = 'b', marker = '+')
  plt.grid()
  plt.legend((s1, s2, c1, c2), ('Sp1','Sp2','Sp1 centroids', 'Sp2 centroids'),scatterpoints=1,
  loc = 'lower right')
  plt.show()

  return (codebooks_mfcc, codebooks_lpc)

nSpeaker = 8
nfiltbank = 12
orderLPC = 15
(codebooks_mfcc, codebooks_lpc) = training(nfiltbank, orderLPC)
directory = r'C:\Users\Dweeja Reddy\Downloads\sm_dataset\train'
fname = str()
nCorrect_MFCC = 0
nCorrect_LPC = 0
def minDistance(features, codebooks):
  speaker = 0
  distmin = np.inf
  for k in range(np.shape(codebooks)[0]):
    D = EUDistance(features, codebooks[k,:,:])
    dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0])
    if dist < distmin:
      distmin = dist
      speaker = k
  return speaker

for i in range(nSpeaker):
  fname = '/s' + str(i+1) + '.wav'
  print('Now speaker ', str(i+1), 'features are being tested')
  (fs,s) = read(directory + fname)
  mel_coefs = mfcc(s,fs,nfiltbank)
  lpc_coefs = lpc(s, fs, orderLPC)
  sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
  sp_lpc = minDistance(lpc_coefs, codebooks_lpc)
  print('Speaker', (i+1), ' in test matches with speaker ', (sp_mfcc+1), 'in train for training with MFCC')
  print ('Speaker', (i+1), ' in test matches with speaker ', (sp_lpc+1), 'in train for training with LPC')
  if i == sp_mfcc:
    nCorrect_MFCC += 1
  if i == sp_lpc:
    nCorrect_LPC += 1

percentageCorrect_MFCC = (nCorrect_MFCC/nSpeaker)*100
print ('Accuracy of result for training with MFCC is ', percentageCorrect_MFCC, '%')
percentageCorrect_LPC = (nCorrect_LPC/nSpeaker)*100
print ('Accuracy of result for training with LPC is ', percentageCorrect_LPC, '%')

