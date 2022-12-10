import pandas as pd
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, index=[1,2])
print(df[0].values)
