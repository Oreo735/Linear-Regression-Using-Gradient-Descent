import pandas as pd
import numpy as np

Houses_data_set = pd.read_pickle("housing.pkl")

housing.head()
housing.tail(25)
housing.info()

housing["ocean_proximity"].value_counts()
housing.describe()

housing.hist(bins=50, figsize=(12, 8))
plt.show()
