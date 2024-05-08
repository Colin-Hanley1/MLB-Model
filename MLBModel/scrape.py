from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np

def getHTMLdocument(url):
    response = requests.get(url)
    
    return response.text

url_to_scrape = "https://www.teamrankings.com/mlb/stat/runs-per-game"

def scrape(url):
    html_document = getHTMLdocument(url)
    soup = BeautifulSoup(html_document, 'html.parser')
    table = soup.find('table', class_='datatable')
    df = pd.DataFrame(columns =['Team','2024', 'Last 3', 'Last 1', 'Home', 'Away', '2023'])

    for row in table.tbody.find_all('tr'):
        columns = row.find_all('td')
    
        if (columns!=[]):
            team = columns[1].text.strip()
            current = columns[2].text.strip()
            lastthree = columns[3].text.strip()
            lastone = columns[4].text.strip()
            home = columns[5].text.strip()
            away = columns[6].text.strip()
            prev = columns[7].text.strip()
        df = df._append({'Team': team,'2024': current, 'Last 3': lastthree, 'Last 1': lastone, 'Home': home, 'Away': away, '2023': prev}, ignore_index=True)   

        
    index = df[df['Team'] == 'Atlanta'].index[0]
    return df


def compile(year,month,day):
    runsdf = scrape("https://www.teamrankings.com/mlb/stat/runs-per-game"+'?date='+year+'-'+month+'-'+day)
    hitsdf = scrape("https://www.teamrankings.com/mlb/stat/hits-per-game"+'?date='+year+'-'+month+'-'+day)
    hrdf = scrape("https://www.teamrankings.com/mlb/stat/home-runs-per-game"+'?date='+year+'-'+month+'-'+day)
    walksdf = scrape("https://www.teamrankings.com/mlb/stat/walks-per-game"+'?date='+year+'-'+month+'-'+day)
    kdf = scrape("https://www.teamrankings.com/mlb/stat/strikeouts-per-game"+'?date='+year+'-'+month+'-'+day)
    badf = scrape("https://www.teamrankings.com/mlb/stat/batting-average"+'?date='+year+'-'+month+'-'+day)
    opsdf = scrape("https://www.teamrankings.com/mlb/stat/on-base-plus-slugging-pct"+'?date='+year+'-'+month+'-'+day)
    babipdf = scrape("https://www.teamrankings.com/mlb/stat/batting-average-on-balls-in-play"+'?date='+year+'-'+month+'-'+day)
    isodf = scrape("https://www.teamrankings.com/mlb/stat/isolated-power"+'?date='+year+'-'+month+'-'+day)
    secadf = scrape("https://www.teamrankings.com/mlb/stat/secondary-average"+'?date='+year+'-'+month+'-'+day)
    orunsdf = scrape("https://www.teamrankings.com/mlb/stat/opponent-runs-per-game"+'?date='+year+'-'+month+'-'+day)
    ohitsdf = scrape("https://www.teamrankings.com/mlb/stat/opponent-hits-per-game"+'?date='+year+'-'+month+'-'+day)
    ohrdf = scrape("https://www.teamrankings.com/mlb/stat/opponent-home-runs-per-game"+'?date='+year+'-'+month+'-'+day)
    owalksdf = scrape("https://www.teamrankings.com/mlb/stat/opponent-walks-per-game"+'?date='+year+'-'+month+'-'+day)
    okdf = scrape("https://www.teamrankings.com/mlb/stat/opponent-strikeouts-per-game"+'?date='+year+'-'+month+'-'+day)
    obadf = scrape("https://www.teamrankings.com/mlb/stat/opponent-batting-average"+'?date='+year+'-'+month+'-'+day)
    oopsdf = scrape("https://www.teamrankings.com/mlb/stat/opponent-on-base-plus-slugging-pct"+'?date='+year+'-'+month+'-'+day)
    obabipdf = scrape("https://www.teamrankings.com/mlb/stat/opponent-batting-average-on-balls-in-play"+'?date='+year+'-'+month+'-'+day)
    oisodf = scrape("https://www.teamrankings.com/mlb/stat/opponent-isolated-power"+'?date='+year+'-'+month+'-'+day)
    osecadf = scrape("https://www.teamrankings.com/mlb/stat/opponent-secondary-average"+'?date='+year+'-'+month+'-'+day)
    
    df = pd.DataFrame(columns=['Team','HRuns', 'ARuns', 'HRunsA', 'ARunsA','HHits', 'AHits', 'HHitsA', 'AHitsA', 'HHR', 'AHR', 'HHRA', 'AHRA','HWalks', 'AWalks', 'HWalksA', 'AWalksA','HK', 'AK', 'HKA', 'AKA','HBA', 'ABA', 'HBAA', 'ABAA','HOPS', 'AOPS', 'HOPSA', 'AOPSA','HBABIP', 'ABABIP', 'HBABIPA', 'ABABIPA','HISO', 'AISO', 'HISOA', 'AISOA', 'HSECA', 'ASECA', 'HSECAA', 'ASECAA'])
    
    for x in range(30):
        team = runsdf['Team'].iloc[x]
        df = df._append({'Team': runsdf['Team'].iloc[runsdf[runsdf['Team'] == team].index[0]], 'HRuns': runsdf['Home'].iloc[runsdf[runsdf['Team'] == team].index[0]],'ARuns': runsdf['Away'].iloc[runsdf[runsdf['Team'] == team].index[0]],  'HRunsA': orunsdf['Home'].iloc[orunsdf[orunsdf['Team'] == team].index[0]],'ARunsA': orunsdf['Away'].iloc[orunsdf[orunsdf['Team'] == team].index[0]],'HHits': hitsdf['Home'].iloc[hitsdf[hitsdf['Team'] == team].index[0]],'AHits': hitsdf['Away'].iloc[hitsdf[hitsdf['Team'] == team].index[0]],  'HHitsA': ohitsdf['Home'].iloc[ohitsdf[ohitsdf['Team'] == team].index[0]],'AHitsA': ohitsdf['Away'].iloc[ohitsdf[ohitsdf['Team'] == team].index[0]],'HHR': hrdf['Home'].iloc[hrdf[hrdf['Team'] == team].index[0]],'AHR': hrdf['Away'].iloc[hrdf[hrdf['Team'] == team].index[0]],  'HHRA': ohrdf['Home'].iloc[ohrdf[ohrdf['Team'] == team].index[0]],'AHRA': ohrdf['Away'].iloc[ohrdf[ohrdf['Team'] == team].index[0]],'HWalks': walksdf['Home'].iloc[walksdf[walksdf['Team'] == team].index[0]],'AWalks': walksdf['Away'].iloc[walksdf[walksdf['Team'] == team].index[0]],  'HWalksA': owalksdf['Home'].iloc[owalksdf[owalksdf['Team'] == team].index[0]],'AWalksA': owalksdf['Away'].iloc[owalksdf[owalksdf['Team'] == team].index[0]],'HK': kdf['Home'].iloc[kdf[kdf['Team'] == team].index[0]],'AK': kdf['Away'].iloc[kdf[kdf['Team'] == team].index[0]],  'HKA': okdf['Home'].iloc[okdf[okdf['Team'] == team].index[0]],'AKA': okdf['Away'].iloc[okdf[okdf['Team'] == team].index[0]],'HBA': badf['Home'].iloc[badf[badf['Team'] == team].index[0]],'ABA': badf['Away'].iloc[badf[badf['Team'] == team].index[0]],  'HBAA': obadf['Home'].iloc[obadf[obadf['Team'] == team].index[0]],'ABAA': obadf['Away'].iloc[obadf[obadf['Team'] == team].index[0]],'HOPS': opsdf['Home'].iloc[opsdf[opsdf['Team'] == team].index[0]],'AOPS': opsdf['Away'].iloc[opsdf[opsdf['Team'] == team].index[0]],  'HOPSA': oopsdf['Home'].iloc[oopsdf[oopsdf['Team'] == team].index[0]],'AOPSA': oopsdf['Away'].iloc[oopsdf[oopsdf['Team'] == team].index[0]],'HBABIP': babipdf['Home'].iloc[babipdf[babipdf['Team'] == team].index[0]],'ABABIP': babipdf['Away'].iloc[babipdf[babipdf['Team'] == team].index[0]],  'HBABIPA': obabipdf['Home'].iloc[obabipdf[obabipdf['Team'] == team].index[0]],'ABABIPA': obabipdf['Away'].iloc[obabipdf[obabipdf['Team'] == team].index[0]],'HISO': isodf['Home'].iloc[isodf[isodf['Team'] == team].index[0]],'AISO': isodf['Away'].iloc[isodf[isodf['Team'] == team].index[0]],  'HISOA': oisodf['Home'].iloc[oisodf[oisodf['Team'] == team].index[0]],'AISOA': oisodf['Away'].iloc[oisodf[oisodf['Team'] == team].index[0]],'HSECA': osecadf['Home'].iloc[secadf[secadf['Team'] == team].index[0]],'ASECA': secadf['Away'].iloc[secadf[secadf['Team'] == team].index[0]],'HSECAA': osecadf['Home'].iloc[osecadf[osecadf['Team'] == team].index[0]],'ASECAA': osecadf['Away'].iloc[osecadf[osecadf['Team'] == team].index[0]]}, ignore_index=True)
    return df

compile('2024','4','24')

