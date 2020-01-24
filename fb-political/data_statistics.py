import sys
import pandas as pd
import numpy as np 


if __name__ == "__main__":

    df = pd.read_csv(sys.argv[1])

    x1 = df['is_phd'].tolist()
    x2 = df['hindu'].tolist()
    x3 = df['been_abroad'].tolist()
    x4 = df['hindi'].tolist()
    x5 = df['north'].tolist()
    y = df['is_right'].tolist()

    print(df.shape, df.columns)


    print("num x1:", np.sum(np.array(x1)))
    print("num x2:", np.sum(np.array(x2)))
    print("num x3:", np.sum(np.array(x3)))
    print("num x4:", np.sum(np.array(x4)))
    print("num x5:", np.sum(np.array(x5)))
    print("num y:", np.sum(np.array(y)))
