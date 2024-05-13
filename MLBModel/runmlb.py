
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

def inputFormatting(input):
    input = input.lower()
    if "arizona" in input or "diamondbacks" in input:
        return "Arizona"
    if "atlanta" in input or "braves" in input:
        return "Atlanta"
    if "baltimore" in input or "orioles" in input:
        return "Baltimore"
    if "boston" in input or "red sox" in input:
        return "Boston"
    if "cubs" in input:
        return "Chi Cubs"
    if "white sox" in input:
        return "Chi Sox"
    if "cincinnati" in input or "reds" in input:
        return "Cincinnati"
    if "cleveland" in input or "guardians" in input:
        return "Cleveland"
    if "colorado" in input or "rockies" in input:
        return "Colorado"
    if "detroit" in input or "tigers" in input:
        return "Detroit"
    if "houston" in input or "astros" in input:
        return "Houston"
    if "kansas city" in input or "royals" in input:
        return "Kansas City"
    if "angels" in input:
        return "LA Angels"
    if "dodgers" in input:
        return "LA Dodgers"
    if "miami" in input or "marlins" in input:
        return "Miami"
    if "milwaukee" in input or "brewers" in input:
        return "Milwaukee"
    if "minnesota" in input or "brewers" in input:
        return "Minnesota"
    if "yankees" in input:
        return "NY Yankees"
    if "mets" in input:
        return "NY Mets"
    if "oaklamd" in input or "athletics" in input or "a's" in input or "as" in input:
        return "Oakland"
    if "philadelphia" in input or "phillies" in input:
        return "Philadelphia"
    if "pittsburgh" in input or "pirates" in input:
        return "Pittsburgh"
    if "san diego" in input or "padres" in input:
        return "San Diego"
    if "seattle" in input or "mariners" in input:
        return "Seattle"
    if "giants" in input:
        return "SF Giants"
    if "st. louis" in input or "cardinals" in input:
        return "St. Louis"
    if "tampa bay" in input or "rays" in input:
        return "Tampa Bay"
    if "texas" in input or "rangers" in input:
        return "Texas"
    if "toronto" in input or "blue jays" in input:
        return "Toronto"



df = compile('2024','5','12')
def matchup(home, away):
    teams = ("Arizona", "Atlanta", "Baltimore", "Chi Cubs", "Chi Sox", "Cincinnati", "Cleveland", "Colorado", "Detroit", "Houston", "Kansas City", "LA Dodgers", "LA Angels", "Miami", "Milwaukee", "Minnesota", "NY Mets", "NY Yankees", "Oakland", "Philadelphia", "Pittsburgh", "San Diego", "Seattle", "SF Giants", "St. Louis", "Tampa Bay", "Texas", "Toronto")
    if home not in teams and away not in teams:
        return("Input incorrect, check spelling")  
    
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
    t1 = inputFormatting(input('Home Team: '))
    t2 = inputFormatting(input('Away Team: '))
    tester = matchup(t1, t2)
    if (type(tester) == str):
        print(tester)
    else:
        tester = torch.from_numpy(tester).float()
        print(float(model(tester)))





