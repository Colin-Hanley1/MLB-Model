
import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from scrape import *
warnings.simplefilter(action='ignore', category=FutureWarning)
model = torch.jit.load('mlbmodel.pt')
nrfi = torch.jit.load('nrfimodel.pt')


df = compile('2024','5','8')
def matchup(home, away):
    index1 = df[df['Team'] == home].index[0]
    index2 = df[df['Team'] == away].index[0]
    hRuns = (float(df['HRuns'].iloc[index1])+float(df['ARunsA'].iloc[index2]))/2
    aRuns = (float(df['ARuns'].iloc[index2])+float(df['HRunsA'].iloc[index1]))/2
    hHits = (float(df['HHits'].iloc[index1])+float(df['AHitsA'].iloc[index2]))/2
    aHits = (float(df['AHits'].iloc[index2])+float(df['HHitsA'].iloc[index1]))/2
    hHR = (float(df['HHR'].iloc[index1])+float(df['AHRA'].iloc[index2]))/2
    aHR = (float(df['AHR'].iloc[index2])+float(df['HHRA'].iloc[index1]))/2
    hWalks = (float(df['HWalks'].iloc[index1])+float(df['AWalksA'].iloc[index2]))/2
    aWalks = (float(df['AWalks'].iloc[index2])+float(df['HWalksA'].iloc[index1]))/2
    hK = (float(df['HK'].iloc[index1])+float(df['AKA'].iloc[index2]))/2
    aK = (float(df['AK'].iloc[index2])+float(df['HKA'].iloc[index1]))/2
    hBA = (float(df['HBA'].iloc[index1])+float(df['ABAA'].iloc[index2]))/2
    aBA = (float(df['ABA'].iloc[index2])+float(df['HBAA'].iloc[index1]))/2
    hOPS = (float(df['HOPS'].iloc[index1])+float(df['AOPSA'].iloc[index2]))/2
    aOPS = (float(df['AOPS'].iloc[index2])+float(df['HOPSA'].iloc[index1]))/2
    hBABIP = (float(df['HBABIP'].iloc[index1])+float(df['ABABIPA'].iloc[index2]))/2
    aBABIP = (float(df['ABABIP'].iloc[index2])+float(df['HBABIPA'].iloc[index1]))/2
    hISO = (float(df['HISO'].iloc[index1])+float(df['AISOA'].iloc[index2]))/2
    aISO = (float(df['AISO'].iloc[index2])+float(df['HISOA'].iloc[index1]))/2
    hSecA = (float(df['HSECA'].iloc[index1])+float(df['ASECAA'].iloc[index2]))/2
    aSecA = (float(df['ASECA'].iloc[index2])+float(df['HSECAA'].iloc[index1]))/2
    
    tester = np.array([hRuns-aRuns,hHits-aHits,hHR-aHR,hWalks-aWalks,hK-aK,hBA-aBA,hOPS-aOPS,hBABIP-aBABIP,hISO-aISO,hSecA-aSecA])
    return tester

while True:
    t1 = input('Home Team: ')
    t2 = input('Away Team: ')
    tester = torch.from_numpy(matchup(t1, t2)).float()
    print(float(model(tester)))




