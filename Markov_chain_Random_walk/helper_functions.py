from bs4 import BeautifulSoup
import requests
from selenium.webdriver import ActionChains
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from scipy.stats import chisquare
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

def get_symbols(csv_output = True):
    url_list=["https://ca.finance.yahoo.com/watchlists/category/section-gainers"]
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=options,
                              executable_path='chromedriver_win32/chromedriver')

    watchlist_list = []
    for url_i in url_list:
        driver.get(url_i)
        soup_url= BeautifulSoup(driver.page_source,'lxml')

        for i in soup_url.find_all(class_ = "data-col0 Ta(start) Pstart(6px) Pend(15px)"):#.find('a'):
            watchlist_list.append("https://ca.finance.yahoo.com/"+i.find('a')['href'])

    stock_list = pd.DataFrame(columns=['Symbol','Name'])
    for url_i in watchlist_list:
        driver.get(url_i)
        soup_url= BeautifulSoup(driver.page_source,'lxml')

        for i in soup_url.find(class_ = "cwl-symbols W(100%)").find('tbody'):
            data_i = []
            for j in i:
                data_i.append(j.text)
            stock_list = stock_list.append({'Symbol' :              data_i[0],
                                            'Name'   :              data_i[1],
                                           },
                                            ignore_index=True)
    stock_list = stock_list.drop_duplicates(ignore_index = True)

    if csv_output == True:
    	stock_list.to_csv('stock_list.csv',index=False)

    return stock_list.drop_duplicates(ignore_index = True)

def create_df(symbol_data, csv_output = True):

	watch_df = yf.download(list(symbol_data.Symbol.unique()),
                       start='2017-01-02',
                       end='2022-01-01',
                       progress=False,
                       )['Close']

	watch_df = watch_df.dropna(subset=['AAPL']).reset_index(drop = True).pct_change()
	watch_df = watch_df.dropna(axis=1, how='all').reset_index(drop = True)
	watch_df = watch_df.dropna(axis=0, how='all').reset_index(drop = True)

	if csv_output:
		watch_df.to_csv('watch_list_all.csv',index=False)

	return watch_df

def transition_matrix_binary(watch_data, stock_name):
    
    x=watch_data.loc[:,[stock_name]].dropna()
    
    shape_matrix = 2
    
    conditions  = [x >= 0, x < 0]


    choices     = [ 'Positive', 'Negative']

    watch_df_2 = x.copy()
    watch_df_2[stock_name] = np.select(conditions, choices)

    new_series = watch_df_2[stock_name][1:].values
    watch_df_2 = watch_df_2.iloc[:-1,:].copy()

    watch_df_2[f'{stock_name}_change'] = new_series
    watch_df_2['from_to'] = watch_df_2[stock_name] + ' ' + watch_df_2[f'{stock_name}_change']
    
    matrix_ = watch_df_2['from_to'].value_counts().sort_index().values.reshape(shape_matrix, shape_matrix)
    matrix_ = matrix_ / np.sum(matrix_,axis = 0).reshape(-1,1)
    
    sns.set(rc={'figure.figsize':(6,5)})
    sns.set(font_scale = 1.3)

    labels = watch_df_2[stock_name].value_counts().sort_index().index

    heat_map = sns.heatmap(matrix_, annot=True, vmin=0.0, vmax=1.0,
                           xticklabels = labels,
                           yticklabels = labels,
                           cmap = sns.cubehelix_palette(light=1, as_cmap=True),
                           fmt='.2f', cbar_kws={"shrink": 0.82},
                           linewidths=0.1,
                           linecolor='gray')

    plt.title(f'Transition matrix of {stock_name}')
    plt.savefig('tr_matrix_binary.png')
    plt.show()

def transition_matrix_median(watch_data, stock_name):
    
    x=watch_data.loc[:,[stock_name]].dropna()

    quantiles = watch_data[stock_name].quantile([0, 0.5, 1]).values
    
    shape_matrix = len(quantiles) - 1
    
    conditions  = [(x >= quantiles[0]) & (x < quantiles[1]),
                   (x >= quantiles[1]) & (x <= quantiles[2])]


    choices     = [ '<= Med', '> Med']

    watch_df_2 = x.copy()
    watch_df_2[stock_name] = np.select(conditions, choices)

    new_series = watch_df_2[stock_name][1:].values
    watch_df_2 = watch_df_2.iloc[:-1,:].copy()

    watch_df_2[f'{stock_name}_change'] = new_series
    watch_df_2['from_to'] = watch_df_2[stock_name] + ' ' + watch_df_2[f'{stock_name}_change']

    shape_matrix = len(quantiles) - 1
    
    matrix_ = watch_df_2['from_to'].value_counts().sort_index().values.reshape(shape_matrix, shape_matrix)
    matrix_ = matrix_ / np.sum(matrix_,axis = 0).reshape(-1,1)
    
    sns.set(rc={'figure.figsize':(6,5)})
    sns.set(font_scale = 1.3)

    labels = watch_df_2[stock_name].value_counts().sort_index().index

    heat_map = sns.heatmap(matrix_, annot=True, vmin=0.0, vmax=1.0,
                           xticklabels = labels,
                           yticklabels = labels,
                           cmap = sns.cubehelix_palette(light=1, as_cmap=True),
                           fmt='.2f', cbar_kws={"shrink": 0.82},
                           linewidths=0.1,
                           linecolor='gray')

    plt.title(f'Transition matrix of {stock_name}')
    plt.savefig('tr_matrix_median.png')
    plt.show()

def transition_matrix_quartile(watch_data, stock_name):
    
    x=watch_data.loc[:,[stock_name]].dropna()

    quantiles = watch_data[stock_name].quantile([0, 0.25, 0.5, 0.75, 1]).values
    
    shape_matrix = len(quantiles) - 1
    
    conditions  = [(x >= quantiles[0]) & (x < quantiles[1]),
                   (x >= quantiles[1]) & (x < quantiles[2]),
                   (x >= quantiles[2]) & (x < quantiles[3]),
                   (x >= quantiles[3]) & (x <= quantiles[4])]


    choices     = [ 'Q1', 'Q2','Q3', 'Q4']

    watch_df_2 = x.copy()
    watch_df_2[stock_name] = np.select(conditions, choices)

    new_series = watch_df_2[stock_name][1:].values
    watch_df_2 = watch_df_2.iloc[:-1,:].copy()

    watch_df_2[f'{stock_name}_change'] = new_series
    watch_df_2['from_to'] = watch_df_2[stock_name] + ' ' + watch_df_2[f'{stock_name}_change']

    shape_matrix = len(quantiles) - 1
    
    matrix_ = watch_df_2['from_to'].value_counts().sort_index().values.reshape(shape_matrix, shape_matrix)
    matrix_ = matrix_ / np.sum(matrix_,axis = 0).reshape(-1,1)
    
    sns.set(rc={'figure.figsize':(6,5)})
    sns.set(font_scale = 1.3)

    labels = watch_df_2[stock_name].value_counts().sort_index().index

    heat_map = sns.heatmap(matrix_, annot=True, vmin=0.0, vmax=1.0,
                           xticklabels = labels,
                           yticklabels = labels,
                           cmap = sns.cubehelix_palette(light=1, as_cmap=True),
                           fmt='.2f', cbar_kws={"shrink": 0.82},
                           linewidths=0.1,
                           linecolor='gray')

    plt.title(f'Transition matrix of {stock_name}')
    plt.savefig('tr_matrix_quartile.png')
    plt.show()

def transition_matrix_percentile(watch_data, stock_name):
    
    x=watch_data.loc[:,[stock_name]].dropna()
    
    percentile_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    quantiles = watch_data[stock_name].quantile(percentile_list).values
    
    shape_matrix = len(quantiles) - 1
    
    conditions  = [(x >= quantiles[0]) & (x < quantiles[1]),
                   (x >= quantiles[1]) & (x < quantiles[2]),
                   (x >= quantiles[2]) & (x < quantiles[3]),
                   (x >= quantiles[3]) & (x < quantiles[4]),
                   (x >= quantiles[4]) & (x < quantiles[5]),
                   (x >= quantiles[5]) & (x < quantiles[6]),
                   (x >= quantiles[6]) & (x < quantiles[7]),
                   (x >= quantiles[7]) & (x < quantiles[8]),
                   (x >= quantiles[8]) & (x < quantiles[9]),
                   (x >= quantiles[9]) & (x <= quantiles[10])]


    choices     = ['0.1', '0.2','0.3', '0.4', '0.5',
                   '0.6', '0.7','0.8', '0.9', '1.0']

    watch_df_2 = x.copy()
    watch_df_2[stock_name] = np.select(conditions, choices)

    new_series = watch_df_2[stock_name][1:].values
    watch_df_2 = watch_df_2.iloc[:-1,:].copy()

    watch_df_2[f'{stock_name}_change'] = new_series
    watch_df_2['from_to'] = watch_df_2[stock_name] + ' ' + watch_df_2[f'{stock_name}_change']

    shape_matrix = len(quantiles) - 1
    
    matrix_ = watch_df_2['from_to'].value_counts().sort_index().values.reshape(shape_matrix, shape_matrix)
    matrix_ = matrix_ / np.sum(matrix_,axis = 0).reshape(-1,1)
    
    sns.set(rc={'figure.figsize':(10,7)})
    sns.set(font_scale = 1.3)

    labels = watch_df_2[stock_name].value_counts().sort_index().index

    heat_map = sns.heatmap(matrix_, annot=True, vmin=0.0, vmax=1.0,
                           xticklabels = labels,
                           yticklabels = labels,
                           cmap = sns.cubehelix_palette(light=1, as_cmap=True),
                           fmt='.2f', cbar_kws={"shrink": 0.82},
                           linewidths=0.1,
                           linecolor='gray')

    plt.title(f'Transition matrix of {stock_name}')
    plt.savefig('tr_matrix_percentile.png')
    plt.show()

def transition_matrix_data(watch_df, stocks_list):
    
    q1_1, q1_2, q1_3, q1_4 = [], [], [], []
    q2_1, q2_2, q2_3, q2_4 = [], [], [], []
    q3_1, q3_2, q3_3, q3_4 = [], [], [], []
    q4_1, q4_2, q4_3, q4_4 = [], [], [], []
    
    values_list = [
        [q1_1, q1_2, q1_3, q1_4],
        [q2_1, q2_2, q2_3, q2_4],
        [q3_1, q3_2, q3_3, q3_4],
        [q4_1, q4_2, q4_3, q4_4]
    ]
    
    matrix_total = np.zeros((4,4))
    chisquare_pvalue = []
    chisquare_stat = []

    for i in stocks_list:
        try:
            x=watch_df.loc[:,[i]].dropna()

            quantiles = watch_df[i].quantile([0, 0.25, 0.5, 0.75, 1]).values

            shape_matrix = len(quantiles) - 1

            conditions  = [(x >= quantiles[0]) & (x < quantiles[1]),
                           (x >= quantiles[1]) & (x < quantiles[2]),
                           (x >= quantiles[2]) & (x < quantiles[3]),
                           (x >= quantiles[3]) & (x <= quantiles[4])]


            choices     = [ 'Q1', 'Q2','Q3', 'Q4']

            watch_df_2 = x.copy()
            watch_df_2[i] = np.select(conditions, choices)

            new_series = watch_df_2[i][1:].values
            watch_df_2 = watch_df_2.iloc[:-1,:].copy()

            watch_df_2[f'{i}_change'] = new_series
            watch_df_2['from_to'] = watch_df_2[i] + ' ' + watch_df_2[f'{i}_change']

            shape_matrix = len(quantiles) - 1

            matrix_ = watch_df_2['from_to'].value_counts().sort_index().values.reshape(shape_matrix, shape_matrix)

            matrix_total += matrix_

            matrix_ = matrix_ / np.sum(matrix_, axis = 0).reshape(-1,1)

            chisquare_pvalue.append(chisquare(matrix_, axis = None)[1])
            chisquare_stat.append(chisquare(matrix_, axis = None)[0])

            for i in range(4):
                for j in range(4):
                    values_list[i][j].append(matrix_[i][j])
        except:
            continue

    matrix_total = matrix_total / np.sum(matrix_total,axis = 0).reshape(-1,1)
    
   
    
    return matrix_total, values_list, chisquare_stat, chisquare_pvalue

def trainsition_matrix_all(change_matrix):
    
    sns.set(rc={'figure.figsize':(6,5)})
    sns.set(font_scale = 1.3)

    labels = [ 'Q1', 'Q2','Q3', 'Q4']

    heat_map = sns.heatmap(change_matrix, annot=True, vmin=0.0, vmax=1.0,
                           xticklabels = labels,
                           yticklabels = labels,
                           cmap = sns.cubehelix_palette(light=1, as_cmap=True),
                           fmt='.2f', cbar_kws={"shrink": 0.82},
                           linewidths=0.1,
                           linecolor='gray')

    plt.title(f'Transition matrix of all stocks')
    plt.show()

def changes_distributions(change_series):
    fig, axes = plt.subplots(4, 4,
                             sharey = True,
                             figsize=(16,16))
    fig.suptitle('Transition matrix with distributions',fontsize = 25)

    choise = [ 'Q1', 'Q2','Q3', 'Q4']
    sns.set(font_scale = 1.0)
    for i in range(4):
        for j in range(4):
            sns.distplot(ax=axes[i,j], x=change_series[i][j])
            axes[i,j].set_title(f'From {choise[i]} to {choise[j]}')
            axes[i,j].set_xticks((np.arange(0,1.25,0.25)))
            axes[i,j].set_ylabel(None)
    plt.savefig('tr_matrix_dist.png')
    plt.show()

def plot_p_values(pvalue_list):
	sns.distplot(x=pvalue_list)
	plt.xticks((np.arange(0.95,1.05,0.05)))
	plt.xlabel('')
	plt.ylabel('')
	plt.title('P values from Chi Square')
	plt.show()


if __name__=='__main__':
	pass