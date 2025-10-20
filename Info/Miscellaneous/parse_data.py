import pandas as pd

dataframe = pd.read_csv("sensor_variable_range.csv", sep = "\t")
file = open("new_file.txt", "w", encoding = "utf-8")
for row in range(len(dataframe.index)):
    range = str(dataframe.iat[row, 3]).replace("nan", "Relative units")
    file.write(f"{dataframe.iat[row, 0]} - {dataframe.iat[row, 2]} | ({range})<br>\n")

file.close()

