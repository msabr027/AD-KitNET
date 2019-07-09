# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:32:47 2019

@author: Mohamed Sabri
"""
import cupy
from scipy.stats import norm
import argparse

def parser_run_model():
    """
    Define parser to parse options from command line, with defaults.
    Refer to this function for definitions of various variables.
    """
    parser = argparse.ArgumentParser(description='Train an FMAEE model and score data in pipeline')

    parser.add_argument('--file', help='where data is stored', type=str, default='./data')
    parser.add_argument('--type', help='type of data to process',
            default='num',
            choices=['num', 'image', 'text'])
    parser.add_argument('--format', help='file format',
            default='num',
            choices=['csv', 'hdf', 'excel','parquet','json'])
    parser.add_argument('--sens', help='cutoff threshold level',
            default='low',
            choices=['low', 'med', 'high'])
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--imgsize1', type=int, default=28)
    parser.add_argument('--imgsize2', type=int, default=28)
    parser.add_argument('--gray', type=str, default='False')
    parser.add_argument('--gpu', type=str, default='False')
    return parser

def pdf(x,mu,sigma): #normal distribution pdf
    x = (x-mu)/sigma
    return cupy.exp(-x**2/2)/(cupy.sqrt(2*cupy.pi)*sigma)

def invLogCDF(x,mu,sigma): #normal distribution cdf
    x = (x - mu) / sigma
    return norm.logcdf(-x) #note: we mutiple by -1 after normalization to better get the 1-cdf

def sigmoid(x):
    return 1. / (1 + cupy.exp(-x))


def dsigmoid(x):
    return x * (1. - x)

def tanh(x):
    return cupy.tanh(x)

def dtanh(x):
    return 1. - x * x

def softmax(x):
    e = cupy.exp(x - cupy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / cupy.sum(e, axis=0)
    else:  
        return e / cupy.array([cupy.sum(e, axis=1)]).T  # ndim = 2


def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

class rollmean:
    def __init__(self,k):
        self.winsize = k
        self.window = cupy.zeros(self.winsize)
        self.pointer = 0

    def apply(self,newval):
        self.window[self.pointer]=newval
        self.pointer = (self.pointer+1) % self.winsize
        return cupy.mean(self.window)

# probability density for the Gaussian dist
# def gaussian(x, mean=0.0, scale=1.0):
#     s = 2 * numpy.power(scale, 2)
#     e = numpy.exp( - numpy.power((x - mean), 2) / s )

#     return e / numpy.square(numpy.pi * s)