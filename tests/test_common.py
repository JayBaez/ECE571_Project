from ece571.common import engineer_features, chronological_split
import pandas as pd

def test_features_and_split():
 d=pd.DataFrame({'Year':[2020]*4,'Month':[1]*4,'Day':[1]*4,'Hour':[10,10,11,11],'Minute':[0,30,0,30],'GHI':[100,200,300,400],'Clearsky GHI':[500]*4,'Wind Direction':[0,90,180,270],'Output Power':[1,2,3,4]})
 d['Timestamp']=pd.to_datetime(d[['Year','Month','Day','Hour','Minute']]); x=engineer_features(d); assert 'ClearSkyIndex' in x and 'HourSin' in x
 a,b=chronological_split(x,.75); assert len(a)==3 and len(b)==1
