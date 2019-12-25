import pandas as pd
import numpy as np
import time


print('开始')
s=time.time()
path='blendsub/'

du1=pd.read_csv(path+'1.csv',header=-1)
du2=pd.read_csv(path+'2.csv',header=-1)
du3=pd.read_csv(path+'3.csv',header=-1)
du4=pd.read_csv(path+'4.csv',header=-1)
du5=pd.read_csv(path+'1.csv',header=-1)
du6=pd.read_csv(path+'2.csv',header=-1)
du7=pd.read_csv(path+'3.csv',header=-1)
du8=pd.read_csv(path+'9.csv',header=-1)
du9=pd.read_csv(path+'9.csv',header=-1)
du10=pd.read_csv(path+'10.csv',header=-1)
du12=pd.read_csv(path+'12.csv',header=-1)

print('读取完成')

du1.columns=['1','2']
du2.columns=['1','2']
du3.columns=['1','2']
du4.columns=['1','2']
du5.columns=['1','2']
du6.columns=['1','2']
du7.columns=['1','2']
du8.columns=['1','2']
du9.columns=['1','2']
du10.columns=['1','2']
du11.columns=['1','2']
du12.columns=['1','2']




#最终权重
a1=0.3
a2= 0.15
a3= 0.125
a4= 0.1
a5=0.2
a6=0.325
a7=0.3
a8=0.375
a9=0.275
a10=0.225
a11=0.15
a12=0.3



du1['2']=du1['2']*a1+du2['2']*a2+du3['2']*a3+du4['2']*a4+du5['2']*a5+du6['2']*a6+du7['2']*a7+du8['2']*a8+du9['2']*a9+du10['2']*a10+du11['2']*a11+du12['2']*a12
du1[['1','2']].to_csv(path+'submit.csv',columns=None,header=None,index=None,sep=',')
print('完成时间',time.time()-s)