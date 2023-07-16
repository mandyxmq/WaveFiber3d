import numpy as np
import pyfftw

def ForwardFFT(A):
    a = pyfftw.empty_aligned(np.shape(A), dtype='complex128')
    b = pyfftw.empty_aligned(np.shape(A), dtype='complex128')
    fft_object_c = pyfftw.FFTW(a, b, axes=(0,1))

    result = fft_object_c(A)

    return result

def InverseFFT(A):
    a = pyfftw.empty_aligned(np.shape(A), dtype='complex128')
    b = pyfftw.empty_aligned(np.shape(A), dtype='complex128')
    fft_object_c = pyfftw.FFTW(a, b, axes=(0,1), direction='FFTW_BACKWARD')

    result = fft_object_c(A)
 
    return result

def ForwardFFT_shifted(A):
    a = pyfftw.empty_aligned(np.shape(A), dtype='complex128')
    b = pyfftw.empty_aligned(np.shape(A), dtype='complex128')
    fft_object_c = pyfftw.FFTW(a, b, axes=(0,1))

    Asize = np.array(np.shape(A))
    A = np.roll(A, np.floor(Asize/2).astype(int), axis=(0,1))

    result = fft_object_c(A)
    result = np.roll(result, -np.floor(Asize/2).astype(int), axis=(0,1))

    return result

def InverseFFT_shifted(A):
    a = pyfftw.empty_aligned(np.shape(A), dtype='complex128')
    b = pyfftw.empty_aligned(np.shape(A), dtype='complex128')
    fft_object_c = pyfftw.FFTW(a, b, axes=(0,1), direction='FFTW_BACKWARD')

    Asize = np.array(np.shape(A))
    A = np.roll(A, -np.floor(Asize/2).astype(int), axis=(0,1))

    result = fft_object_c(A)
    result = np.roll(result, np.floor(Asize/2).astype(int), axis=(0,1))

    return result

def ACFweightedFT(thetanum, phinum, Fnoiseaveragemat, bandnum, weight):
    weightedFT = np.zeros((thetanum, phinum))
    for i in range(bandnum):
        weightedFT += weight[i]*weight[i]*Fnoiseaveragemat[:,:,i]**2
    acf_weighted_FT = np.real(ForwardFFT_shifted(weightedFT))
    return weightedFT, acf_weighted_FT

def weightedsum(thetanum, phinum, instancenum, noiserawmat, bandnum, weight):
    mean = np.zeros((thetanum, phinum))
    noisemat = np.zeros((thetanum, phinum, instancenum), dtype=float)
    for i in range(instancenum):
        curnoise = np.zeros((thetanum, phinum))
        for index in range(0, bandnum):
            noise = noiserawmat[:,:,i, index]
            curnoise += weight[index]*noise
        noisemat[:,:,i] = curnoise
        mean = mean + curnoise/instancenum
    return noisemat, mean

def ACFfromFT(thetanum, phinum, instancenum, noisemat, mean):
    Fnoiseaverage = np.zeros((thetanum, phinum))
    for i in range(instancenum):
        noise = noisemat[:,:,i]
        Fnoise = np.abs(ForwardFFT_shifted(noise-mean))
        Fnoiseaverage = Fnoiseaverage + Fnoise/instancenum
    acf_from_FT = np.real(ForwardFFT_shifted(Fnoiseaverage**2))
    return Fnoiseaverage, acf_from_FT