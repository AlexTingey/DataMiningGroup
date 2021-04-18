import numpy as np
import glob as glob
import pandas as pd

file_list = glob.glob('./dataFiles/*.csv')

data = []

file_list.remove('./dataFiles\\stackedData.csv')
print(file_list)
file_list.sort()
for path in file_list:
    try:
        dataGen = np.genfromtxt(path, delimiter=",", skip_header=1)
        data.append(dataGen.tolist())
    except:
        print(path + " Error!")

full_data = np.vstack(data)
final_data = pd.DataFrame.from_records(full_data)

print(final_data)

final_data.to_csv('./dataFiles/stackedData.csv')