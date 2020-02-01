import numpy as np
import pandas as pd

"""==========================================================================
This program create exam marks for 100 students in six subjects. The marks are 
gievn randomly between 0 and 100. In every subject pass marks are 33 and if
a student fails in two or more subjects then he or she is considered fail.
Apart from binary output - pass (1) and fail (0) the students are also given 
divisions - 1,2 and 3 if total marks are more than 60 %, between 60 and 45 %
and between 45 and 33 %. If the total marks are less that 33 % the diviion is 0.
This data set can be used for binary as well as multi-class classification. 
At present all the class values are exact, however, we can make the 
problem challenging by adding some noise (putting wrong labels for some 
data point) also. 

     - Jayanti Prasad [prasad.jayanti@gmail.com]
       Feb 01, 2020 
 

========================================================================="""

def get_exam_data ():
    # create marks for six subjects in the range 0-50 

    subjects = ["Hindi","English","Science","Maths","History","Geograpgy"]

    np.random.seed(seed=167) 

    # set the marks randomly 
    marks = np.random.randint(100, size=(1000,6))
    df = pd.DataFrame(data=marks, columns=subjects)

    
    df['Total'] =  df[subjects].sum(axis=1)
    
    df1 = df.copy()

    # set the result fail is marks in a subject < 17  
    for s in subjects:
        df1[s] =[1 if i > 33 else 0 for i in  df[s].tolist()]

    df1['Results'] =  df1[subjects].sum(axis=1)

    # declare the result fail (0) if fail in more than 2 subjects 
    df['Results'] = df1['Results'].apply(lambda x : 0 if x <=4 else 1)

    # get the divison :  1  for > 60 %, 2 for > 45 % and < 60 %, 3 for  < 45 % and > 33 %,
    # 0 for < 33 %

    df['Div'] = [1 if i > 360  else 2 if i < 360 and i > 270
       else 3 if i < 270 and i > 198 else 0 for i in df['Total'].tolist() ]

    # return the result data frame 

    return df 


if __name__ == "__main__":

    df = get_exam_data ()

    df.to_csv("results.csv") 
