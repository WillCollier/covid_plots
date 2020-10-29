import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tkinter import *

from requests import get

# dchoose = 'region'
# #---------Use government API to access data----------#
# #--https://coronavirus.data.gov.uk/developers-guide--#
# def get_data(url):
#     response = get(endpoint, timeout=10)
#
#     if response.status_code >= 400:
#         raise RuntimeError(f'Request failed: {response.text}')
#
#     return response.json()
#
#
# if dchoose == 'nation':
#     if __name__ == '__main__':
#         endpoint = (
#             'https://api.coronavirus.data.gov.uk/v1/data?'
#             'filters=areaType=nation&'
#             'structure={"date":"date", "newDeaths":"newDeaths28DaysByPublishDate", "cumDeaths":"cumDeaths28DaysByPublishDate","nation":"areaName"}'
#         )
#         data = get_data(endpoint)
#         # print(data)
#
# elif dchoose == 'region':
#     if __name__ == '__main__':
#         endpoint = (
#             'https://api.coronavirus.data.gov.uk/v1/data?'
#             'filters=areaType=region&'
#             'structure={"date":"date", "newDeaths":"newDeaths28DaysByPublishDate", "cumDeaths":"cumDeaths28DaysByPublishDate","nation":"areaName"}'
#         )
#         data = get_data(endpoint)
#         print(data)



from requests import get
from json import dumps

ENDPOINT = "https://api.coronavirus.data.gov.uk/v1/data"
AREA_TYPE = "nation"
# AREA_TYPE = "region"
# AREA_NAME = "england"

filters = [
    f"areaType={ AREA_TYPE }"#,
    #f"areaName={ AREA_NAME }"
]

if AREA_TYPE == "nation":
    structure = {
        "date": "date",
        "nation": "areaName",
        "code": "areaCode",
        # "cases": {
        "newCases": "newCasesByPublishDate",
        "cumCases": "cumCasesByPublishDate",
        # },
        # "deaths": {
        "newDeaths": "newDeaths28DaysByPublishDate",
        "cumDeaths": "cumDeaths28DaysByPublishDate"
        # }
    }
if AREA_TYPE == "region":
    structure = {
        "date": "date",
        "nation": "areaName",
        "code": "areaCode",
        # "cases": {
        "newCases": "newCasesBySpecimenDate",
        "cumCases": "cumCasesBySpecimenDate",
        # },
        # "deaths": {
        "newDeaths": "newDeaths28DaysByPublishDate",
        "cumDeaths": "cumDeaths28DaysByPublishDate"
        # }
    }




api_params = {
    "filters": str.join(";", filters),
    "structure": dumps(structure, separators=(",", ":"))
}

response = get(ENDPOINT, params=api_params, timeout=10)

if response.status_code >= 400:
    raise RuntimeError(f'Request failed: { response.text }')

print(response.url)
print(response.json())
data = response.json()

def mergeDict(dict1, dict2):
   ''' Merge dictionaries and keep values of common keys in list'''
   dict3 = {**dict1, **dict2}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
            # print(key, dict1[key], dict2[key], value)
            if (type(value) == str )| (type(value) == int) | (type(value) == float):
                dict3[key] = [value , dict1[key]]
                # print("first", dict3[key])
            elif type(value) == list:
                value.append(dict1[key])
                dict3[key] = value
                # print("rest", dict3[key])
   return dict3

# Merge dictionaries and add values of common keys in a list
for i in range(data['length']-1):
    if i == 0:
        dict = mergeDict(data["data"][i], data["data"][i+1])
    else:
        dict = mergeDict(data["data"][i+1], dict)

#-------------------Load dataset-----------------#
cv_latest = pd.DataFrame(dict)
#-------------------Find uninque nations-----------------#
nations = np.unique(cv_latest['nation'])
reporting_dates = np.unique(cv_latest['date'])

#-------------Make list of dictionary keys-----------#
keys = ['date']
for i in nations:
    keys.append(i + '_dcids')
    keys.append(i + '_rollavg')
    keys.append(i + '_rollavg_std')
    keys.append(i + '_cum')
keys.append('UK_dcids')
keys.append('UK_rollavg')
keys.append('UK_rollavg_std')
keys.append('UK_cum')
keys = np.array(keys)
# print(keys)


#-------------------FUNCTIONS-----------------#
# def pullout_dcids(i, len_arr_out):
#     dcideaths = i['newCases'].values
#     dci_deaths = np.pad(dcideaths, (len_arr_out - len(dcideaths), 0))
#     dci_deaths[np.where(np.isnan(dci_deaths) == True)] = 0
#     return dci_deaths
#
# def pullout_csums(i, len_arr_out):
#     cumdeaths = i['cumCases'].values
#     cum_deaths = np.pad(cumdeaths, (len(reporting_dates)-len(cumdeaths), 0))
#     cum_deaths[np.where(np.isnan(cum_deaths)==True)] = 0
#     return cum_deaths


def pullout_dcids(i, len_arr_out):
    dcideaths = i['newDeaths'].values
    dci_deaths = np.pad(dcideaths, (len_arr_out - len(dcideaths), 0))
    dci_deaths[np.where(np.isnan(dci_deaths) == True)] = 0
    return dci_deaths

def pullout_csums(i, len_arr_out):
    cumdeaths = i['cumDeaths'].values
    cum_deaths = np.pad(cumdeaths, (len(reporting_dates)-len(cumdeaths), 0))
    cum_deaths[np.where(np.isnan(cum_deaths)==True)] = 0
    return cum_deaths

def moving_avg(array_in, nOsteps):
    # create a 7-day moving average for daily cases
    # and a standard deviation
    new_array = np.zeros(len(array_in))
    new_array_std = np.zeros(len(array_in))
    for j in range(len(new_array)):
        if j == 0:
            new_array[0] += array_in[0]
            if array_in[0] != 0:
                new_array_std[0] += np.sqrt(array_in[0])
            else:
                new_array_std[0] += 0
        elif 0 < j < nOsteps:
            new_array[j] += np.mean(array_in[:j])
            new_array_std[j] += np.std(array_in[:j])
        else:
            new_array[j] += np.mean(array_in[j-nOsteps:j])
            new_array_std[j] += np.std(array_in[j - nOsteps:j])

    return new_array, new_array_std

#-------------------FUNCTIONS-----------------#

#combine into a single dataframe with columns per country
#create numpy arrays
len_array = reporting_dates.shape[0]
sum_dci_deaths, sum_rolling_avg = np.zeros(len_array), np.zeros(len_array)
sum_cum_deaths, sum_rolling_avg_std = np.zeros(len_array), np.zeros(len_array)
for i in nations:
    country_data = cv_latest.loc[cv_latest['nation'] == i].iloc[::-1]

    dci_deaths = pullout_dcids(country_data, len_array)
    rolling_avg, rolling_avg_std = moving_avg(dci_deaths, 7)
    cum_deaths = pullout_csums(country_data, len_array)

    # rolling_avg_std += 1e-6

    sum_dci_deaths += dci_deaths
    # sum_rolling_avg += rolling_avg
    # sum_rolling_avg_std += np.sqrt(sum_rolling_avg_std**2 + rolling_avg**2)
    sum_cum_deaths += cum_deaths

    reporting_dates = np.c_[reporting_dates, dci_deaths]
    reporting_dates = np.c_[reporting_dates, rolling_avg]
    reporting_dates = np.c_[reporting_dates, rolling_avg_std]
    reporting_dates = np.c_[reporting_dates, cum_deaths]

sum_rolling_avg, sum_rolling_avg_std = moving_avg(sum_dci_deaths, 7)

Data = {}
for i in range(len(keys)):
    if i >= len(keys)-4:
        if i == len(keys)-4:
            Data[keys[i]] = sum_dci_deaths
        elif i == len(keys)-3:
            Data[keys[i]] = sum_rolling_avg
        elif i == len(keys)-2:
            Data[keys[i]] = sum_rolling_avg_std
        elif i == len(keys)-1:
            Data[keys[i]] = sum_cum_deaths
    else:
        Data[keys[i]] = reporting_dates[:, i]

combined_df = pd.DataFrame(Data, columns=keys)

for i in range(1,len(keys)):
    combined_df[keys[i]] = combined_df[keys[i]].apply(pd.to_numeric, downcast='float', errors='coerce')

plot_col_dcids = keys[np.arange(1, len(keys), 4)]
plot_col_roll_avgs = keys[np.arange(2, len(keys), 4)]
plot_col_roll_avgs_std = keys[np.arange(3, len(keys), 4)]
cols = sns.color_palette("Paired")[:len(plot_col_dcids)]

# colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "olive", "maroon", "sea green"]
# cols = sns.xkcd_palette(colors)

fig, ax = plt.subplots()

combined_df.plot(x='date', y=plot_col_dcids, ax=ax, grid=False, rot=90, alpha=0.5,
                 color=cols, figsize=(12, 8))

# combined_df.plot(x='date', y=plot_col_roll_avgs, ax=ax, sharex=True, grid=False,
#                  rot=90, lw=3, color=cols)

for i in range(len(plot_col_roll_avgs)):
    upper_bounds = combined_df[plot_col_roll_avgs[i]].to_numpy() - combined_df[plot_col_roll_avgs_std[i]].to_numpy()
    lower_bounds = combined_df[plot_col_roll_avgs[i]].to_numpy() + combined_df[plot_col_roll_avgs_std[i]].to_numpy()
    ax.fill_between(combined_df['date'], lower_bounds, upper_bounds,
                    alpha=0.3, facecolor=cols[i])

for i in range(len(plot_col_roll_avgs)):
    combined_df.plot(x='date', y=plot_col_roll_avgs[i], ax=ax, sharex=True, grid=False, rot=90,
                     lw=3, color=cols[i])

plt.tight_layout()

fig, ax = plt.subplots()
plot_col_cum = keys[np.arange(4, len(keys), 4)]
combined_df.plot(x='date', y=plot_col_cum, ax=ax, grid=False, rot=90, figsize=(12,8), color=cols)
plt.tight_layout()



#
# def open_window():
#
#
#     master = Tk()
#     master.title("Display Covid Plot")
#     master.geometry('600x400')
#
#     master_OPTIONS = ["Include", "Remove"]
#
#     master_variables = StringVar(master)
#     master_variables.set(master_OPTIONS[0]) # default value
#
#     w = OptionMenu(master, master_variables, *master_OPTIONS)
#     w.pack()
#
#     entry_master_window = Entry(master, width=40)
#     entry_master_window.insert(0, "WillData")
#     entry_master_window.pack()
#
#
#     button_master_window = Button(master, text="OK", command=ok)
#     button_master_window.pack()
#
#     master.mainloop()
#
#
# open_window()





#
# class plot_window_select:
#
#
#
#     def __init__(self, df):
#         self.df = df
#         keys = []
#         for i in df.columns:
#             keys.append(i)
#         self.keys = keys
#         self.names_flags = np.ones(len(keys) - 1)
#
#     def generate(self):
#
#         for i in range(len(self.select_options)):
#             if self.select_options[i].get() == "Include":
#                 self.names_flags[i] = 1
#             else:
#                 self.names_flags[i] = 0
#
#         print(self.names_flags)
#         self.plot()
#
#
#     def make_window(self):
#         self.master = Tk()
#         self.master.title("Display Covid Plot")
#         self.master.geometry('1500x800')
#
#         master_OPTIONS = ["Include", "Remove"]
#
#
#         labels = []
#         for i in range(len(keys)-1):
#             label = Label(self.master, text=keys[i+1])
#             label.grid(column=0, row=i)
#             labels.append(label)
#
#         self.select_options = []
#         for i in range(len(keys)-1):
#
#             plot_tag_variables = StringVar(self.master)
#             plot_tag_variables.set(master_OPTIONS[0])
#             tags = OptionMenu(self.master, plot_tag_variables, *master_OPTIONS)
#
#             tags.grid(column=1, row=i)
#             self.select_options.append(plot_tag_variables)
#
#
#         button_plot_window = Button(self.master, text="generate", command=self.generate)
#         button_plot_window.grid(column=0, row=len(keys)+1)
#
#
#         self.master.mainloop()
#
#
#
#     def plot(self):
#         from matplotlib.figure import Figure
#         from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
#                                                        NavigationToolbar2Tk)
#
#         for i in range(1, len(self.keys)):
#             combined_df[self.keys[i]] = combined_df[self.keys[i]].apply(pd.to_numeric, downcast='float', errors='coerce')
#
#
#         plot_cols = []
#         for i in range(len(self.names_flags)):
#             if self.names_flags[i]==1:
#                 plot_cols.append(self.keys[i+1])
#
#         cols = sns.color_palette("Set2")[:len(plot_cols)]
#
#         fig = Figure(figsize=(7, 7),
#                      dpi=100)
#
#         # adding the subplot
#         ax = fig.add_subplot(111)
#
#
#         combined_df.plot(x=self.keys[0], y=plot_cols, ax=ax, grid=False, rot=90, alpha=0.5,
#                          color=cols, figsize=(12, 8))
#
#         canvas = FigureCanvasTkAgg(fig, master=self.master)
#         canvas.draw()
#
#         # placing the canvas on the Tkinter window
#         canvas.get_tk_widget().grid(column=3, row=0, columnspan=len(self.keys)*5, rowspan=len(self.keys)*5)
#
#
#
# p = plot_window_select(combined_df)
# p.make_window()