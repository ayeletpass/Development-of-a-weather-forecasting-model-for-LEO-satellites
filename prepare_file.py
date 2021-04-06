import pandas as pd
import numpy as np
import csv
def removeEmptyRow(path, output_path):
   with open(path) as fin:
       rows = pd.read_csv(path, header=None)
       cols = next(csv.reader(fin))
       df = pd.DataFrame(columns=cols)

       nan_row = np.array(np.where(pd.isnull(rows)))


       print(zip(nan_row, range(len(cols))))

       print(len(rows))
       for i,j in zip(nan_row, range(len(cols))):
           print(i,j)
       r = [df.iloc[i, j] for i, j in zip(nan_row, range(len(cols)))]
       print(r)
       # for row in rows :
       #     if not row.empty:
       #          df = df.append(row)
       # df.to_csv(output_path)

if __name__ == "__main__":
    path = 'simulator.ver3.csv'
    output_path = "simulator3.csv"
    removeEmptyRow(path, output_path)