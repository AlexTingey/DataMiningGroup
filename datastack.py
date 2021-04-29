import numpy as np
import glob as glob
import pandas as pd
from datetime import datetime
file_list = glob.glob('./dataFiles/*.csv')

data = []

file_list.remove('./dataFiles\\stackedData.csv')
print(file_list)
file_list.sort()
seen_dates = set()
date_tuples = []
last_date = None
current_start_date_in_tuple = None
for idx in range(len(file_list)):
    path = file_list[idx]
    try:
        appendData = []
        dataGen = np.genfromtxt(path, delimiter=",", skip_header=1)
        numRows = dataGen.shape[0]
        for dataIdx in range(numRows):
            date = dataGen[dataIdx][0]

            if date in seen_dates:
                print(f"skipping date {date} because we have seen it before")
            else:
                seen_dates.add(date)
                appendData.append(dataGen[dataIdx])
                if last_date == None and current_start_date_in_tuple == None:
                    last_date = date
                    current_start_date_in_tuple = date
                elif last_date != None:
                    real_curr_date = datetime.utcfromtimestamp(date)
                    last_date_real = datetime.utcfromtimestamp(last_date)
                    if abs((real_curr_date - last_date_real).days > 1):
                        real_current_start_date = datetime.utcfromtimestamp(current_start_date_in_tuple)

                        start_date_format = real_current_start_date.strftime('%Y-%m-%d')
                        last_date_format = last_date_real.strftime('%Y-%m-%d')
                        date_tuples.append((start_date_format, last_date_format))

                        current_start_date_in_tuple = date
                    last_date = date
        if(len(appendData) == 0):
            continue
        else:
            data.append(appendData)
    except Exception as e:
        print(path + " Error! " + e)
real_current_start_date = datetime.utcfromtimestamp(current_start_date_in_tuple)

start_date_format = real_current_start_date.strftime('%Y-%m-%d')
last_date_format = last_date_real.strftime('%Y-%m-%d')
date_tuples.append((start_date_format, last_date_format))
full_data = np.vstack(data)
final_data = pd.DataFrame.from_records(full_data)

print(final_data)

final_data.to_csv('./dataFiles/stackedData.csv')
print(date_tuples)