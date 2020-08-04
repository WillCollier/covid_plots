import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#-------------------FUNCTIONS-----------------#
def pullout_dcids(i, len_arr_out):
    dcideaths = i['Daily change in deaths'].values
    dci_deaths = np.pad(dcideaths, (len_arr_out - len(dcideaths), 0))
    dci_deaths[np.where(np.isnan(dci_deaths) == True)] = 0
    return dci_deaths

def pullout_csums(i, len_arr_out):
    cumdeaths = i['Cumulative deaths'].values
    cum_deaths = np.pad(cumdeaths, (len(reporting_dates)-len(cumdeaths), 0))
    cum_deaths[np.where(np.isnan(cum_deaths)==True)] = 0
    return cum_deaths

def moving_avg(array_in, nOsteps):
    # create a 7-day moving average for daily cases
    new_array = np.zeros(len(array_in))
    for j in range(len(new_array)):
        if j == 0:
            new_array[0] += array_in[0]
        elif 0 < j < nOsteps:
            new_array[j] += np.mean(array_in[:j])
        else:
            new_array[j] += np.mean(array_in[j-nOsteps:j])
    return new_array


#-------------------Load dataset-----------------#
#take dataset from
# cv_deaths = pd.read_csv('../PhD/coronavirus-deaths_latest.csv')
cv_deaths = pd.read_csv('../PhD/covid.csv')


#Seperate out the different countries
#Reorder for cumulative in sensible direction
England = cv_deaths.loc[cv_deaths['Area name'] == 'England'].iloc[::-1]
Wales = cv_deaths.loc[cv_deaths['Area name'] == 'Wales'].iloc[::-1]
Scotland = cv_deaths.loc[cv_deaths['Area name'] == 'Scotland'].iloc[::-1]
Northern_Ireland = cv_deaths.loc[cv_deaths['Area name'] == 'Northern Ireland'].iloc[::-1]
UK = cv_deaths.loc[cv_deaths['Area name'] == 'United Kingdom'].iloc[::-1]


#combine into a single dataframe
#create numpy arrays
reporting_dates = UK['Reporting date'].values
len_array = reporting_dates.shape[0]
for i in [England, Wales, Scotland, Northern_Ireland, UK]:

    dci_deaths = pullout_dcids(i, len_array)
    rolling_avg = moving_avg(dci_deaths, 7)
    cum_deaths = pullout_csums(i, len_array)

    reporting_dates = np.c_[reporting_dates, dci_deaths]
    reporting_dates = np.c_[reporting_dates, rolling_avg]
    reporting_dates = np.c_[reporting_dates, cum_deaths]



Data = {'Date': reporting_dates[:,0], 'England_dcids': reporting_dates[:,1],
        'England_rollavg': reporting_dates[:,2],
        'England_cum': reporting_dates[:,3], 'Wales_dcids': reporting_dates[:,4],
        'Wales_rollavg':reporting_dates[:,5],
        'Wales_cum': reporting_dates[:,6], 'Scotland_dcids': reporting_dates[:,7],
        'Scotland_rollavg':reporting_dates[:,8],
        'Scotland_cum': reporting_dates[:,9], 'Northern_Ireland_dcids': reporting_dates[:,10],
        'Northern_Ireland_rollavg':reporting_dates[:,11],
        'Northern_Ireland_cum': reporting_dates[:,12], 'UK_dcids': reporting_dates[:,13],
        'UK_rollavg':reporting_dates[:,14], 'UK_cum': reporting_dates[:,15]}

combined_df = pd.DataFrame(Data, columns=['Date', 'England_dcids', 'England_rollavg',
                                    'England_cum', 'Wales_dcids', 'Wales_rollavg',
                                    'Wales_cum', 'Scotland_dcids', 'Scotland_rollavg',
                                    'Scotland_cum', 'Northern_Ireland_dcids',
                                    'Northern_Ireland_rollavg', 'Northern_Ireland_cum',
                                    'UK_dcids', 'UK_rollavg', 'UK_cum'])

plot_col_dcids = ['England_dcids','Scotland_dcids','Wales_dcids','Northern_Ireland_dcids','UK_dcids']
plot_col_roll_avgs = ['England_rollavg','Scotland_rollavg','Wales_rollavg','Northern_Ireland_rollavg','UK_rollavg']
cols = sns.color_palette("Set2")[:len(plot_col_dcids)]

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
cols = sns.xkcd_palette(colors)

fig, ax = plt.subplots()

combined_df.plot(x = 'Date', y = plot_col_dcids, ax=ax, grid=False, rot=90, alpha=0.5,
                 color=cols, figsize=(12,8))
combined_df.plot(x = 'Date', y = plot_col_roll_avgs, ax=ax, sharex=True, grid=False, rot=90,
                 lw=3, color=cols)
plt.tight_layout()

fig, ax = plt.subplots()
plot_col_cum = ['England_cum','Scotland_cum','Wales_cum','Northern_Ireland_cum','UK_cum']
combined_df.plot(x = 'Date', y = plot_col_cum, ax=ax, grid=False, rot=90, figsize=(12,8), color=cols)
plt.tight_layout()





















