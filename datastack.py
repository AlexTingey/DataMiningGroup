import numpy as np
import glob as glob
import pandas as pd

file_list = glob.glob('./dataFiles/*.csv')

data = []

file_list.remove('./dataFiles\\stackedData.csv')
print(file_list)
file_list.sort()
seen_dates = set()

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
        
        data.append(appendData)
    except Exception as e:
        print(path + " Error! " + e)

full_data = np.vstack(data)
final_data = pd.DataFrame.from_records(full_data)

print(final_data)

final_data.to_csv('./dataFiles/stackedData.csv')