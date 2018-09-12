    # Importing requisite libraries

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from functools import reduce
    
    
    #IMPORTING DATASET for years 2017 and 2016
    
    dataset1 = pd.read_csv('FinalData2017.csv')
    dataset2 = pd.read_csv('FinalData2016.csv')
    
    wdata1 = pd.read_csv('climate2017.csv') 
    wdata2 = pd.read_csv('climate2016.csv') 
    dataset = pd.read_csv('FinalData.csv') 
    
    #Exploratory Data Analysis
    
    #trip average 
    #trip count per hr (busiest hr)
    #busiest stations
    #busiest month / day
    
    #Annual Busiest Hour
    
     
    x1 = [d.split(' ')[0] for d in datasetOld['Start Time']]
    y = [d.split(' ')[0] for d in dataset['Stop Time']]
    counts1 =  datasetOld.groupby([x1,'HourOfIssue','Start Station ID']).count()
    counts =  dataset.groupby([x,'HourOfIssue','Start Station ID']).count()
    counte =  dataset.groupby([y, 'HourOfWithdraw','End Station ID']).count()
   
    
    height = tripcnt
    bars = ('0', '1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23')    
    plt.bar(bars, height, color=(0.2, 0.4, 0.6, 0.6))
    s=max(tripcnt)
    plt.text(0, 30000,'Busiest Hr(2017) is [8:00 - 9:00] am', fontsize=10, bbox=dict(facecolor='green', alpha=0.5))       
    plt.text(5, 28000,s, fontsize=10)       
    plt.xlabel('Hour Wise Trip Count Over a Year', fontweight='bold',color = 'black', fontsize='12', horizontalalignment='center')    
    
    #Busiest stations per hour
    tripPerSt = ndarray((55,),int)
    for i in range(55): tripPerSt[i]=0 
   
    cnt=0
    cnt = dataset.groupby(['Start Station ID']).count()
    i=0
    for i in range(55):
        tripPerSt[i]= cnt['Trip Duration'][i] 
    
    tripPerSt=cnt['Trip Duration']
    m=tripPerSt.index.values
    df = pd.DataFrame({'Trips':tripPerSt[:,],'Station ID':m[:,]})

    x = ndarray((55,),int)
    for i in range(55): x[i]=0 
    x=df['Station ID'] 
    y=df['Trips']
    bar=('3183','3184','3185','3186','3187','3188','3189','3190','3191','3192','3193','3194','3195','3196','3197','3198','3199','3200','3201','3202','3203','3205','3206','3207','3209','3210','3211','3212','3213','3214','3215','3216','3217','3220','3225','3267','3268','3269','3270','3271','3272','3273','3274','3275','3276','3277','3278','3279','3280','3281','3426','3481','3638','3639','3640')
    plt.barh(bar, y, color=(0.2, 0.4, 0.6, 0.6))
    s=max(y)
    plt.text(20, 3500,'Busiest Dock Station', fontsize=10, bbox=dict(facecolor='green', alpha=0.5))       
    plt.text(21, 3500,s, fontsize=10)       
    plt.xlabel('Trips taken from each Start Station in a Year', fontweight='bold',color = 'black', fontsize='10', horizontalalignment='center')    
    plt.ylabel('Start Station ID', fontweight='bold',color = 'black', fontsize='12', horizontalalignment='center')    
    
   # Busiest Month
        
    value1=df['Station ID']
    value2=df['Trips']
    plt.stem(mcnt, markerfmt=' ')
    bar2=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
    (markers, stemlines, baseline) = plt.stem(mcnt)
    plt.setp(markers, marker='D', markersize=10, markeredgecolor="black", markeredgewidth=2,color="olive")
    plt.xlabel('Month of the Year', fontweight='bold',color = 'Black', fontsize='15', horizontalalignment='center')    
    plt.ylabel('No of Trips for the Month', fontweight='bold',color = 'Black', fontsize='15', horizontalalignment='center')    
    plt.text(3, 35000,'Busiest Month "AUGUST"',bbox=dict(facecolor='green', alpha=0.5))
    
    plt.show()
    
    #Holiday Trips 
    
    tpcnt[0]/351
    tpcnt[1]/14
    holcnt = ndarray((2,),int)
    for i in range(2): holcnt[i]=0  
    holcount=dataset.groupby(['Holidays',x]).count()
       
    
    # MAIN CODE
    i=0
    for i in range(247584):
        dataset['HourOfIssue'][i]=int(dataset['HourOfIssue'][i])
            
    
    del counte
    
    dataset.to_csv("xtraData.csv", encoding='utf-8',index=False)
    
    
    dataset = pd.read_csv('extraData.csv')
    list(dataset)
    dataset.HourOfIssue.unique()
    counts =  dataset.groupby([x,'HourOfIssue','Start Station ID']).count()
    #counts = dataset.groupby(['Start Station ID','HourOfIssue']).count()
    countback = dataset.groupby(['End Station ID',])
    
    
    
    
    datewise = dataset.groupby(['Stop Time']).count()
    
    bikecount = dataset['Bike ID'].count()
    
    uniqueBikes = dataset['Bike ID'].unique() #number of unique bikes 
    uniqueStop = dataset['End Station ID'].unique()
    uniqueStart = dataset['Start Station ID'].unique()
    
    z=  counts.index.values
    
    
    dataset.HourOfWithdraw.unique()
    y = [d.split(' ')[0] for d in dataset['Stop Time']]
    counte =  dataset.groupby([y, 'HourOfWithdraw','End Station ID']).count()
 #   counte = dataset.groupby(['End Station ID','HourOfWithdraw']).count()
    
    
    del uniqueStations
    
    x= np.unique(dataset['Bike ID'])
    
    df_list = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10, dataset11, dataset12]
    df_final = reduce(lambda left,right: pd.merge(left,right), dfs)
    
    filenames = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset9', 'dataset10', 'dataset11', 'dataset12']
    dfs = [pd.read_csv(filename, index_col=index_col) for filename in filenames)]
    dfs[0].join(dfs[1:])
    
    from math import sin, cos, sqrt, atan2, radians
    
    filenames = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset9', 'dataset10', 'dataset11', 'dataset12']
    dfs = [pd.read_csv(filename, index_col=index_col) for filename in filenames)]
    dfs[0].join(dfs[1:])
    #START TIME
    datasetOld = dataset
    
    dataset['StartDay']=[d.split('/')[0] for d in dataset['Start Time']]
    dataset['StartMonth']=[d.split('/')[1] for d in dataset['Start Time']]
    x=[d.split('/')[2] for d in dataset['Start Time']]
    dataset['StartYear']=[d.split(' ')[0] for d in x]
    
    #STOP TIME
    dataset['StopDay']=[d.split('/')[0] for d in dataset['Stop Time']]
    dataset['StopMonth']=[d.split('/')[1] for d in dataset['Stop Time']]
    y=[d.split('/')[2] for d in dataset['Stop Time']]
    dataset['StopYear']=[d.split(' ')[0] for d in x]
    
    
    dataset['Start Hour']=[d.split(' ')[1] for d in dataset['Start Time']]
    dataset['HourOfIssue']=[d.split(':')[0] for d in dataset['Start Hour']]
    
    dataset['End Hour']=[d.split(' ')[1] for d in dataset['Stop Time']]
    
    dataset['HourOfWithdraw']=[d.split(':')[0] for d in dataset['Start Hour']]
    
    
    
    #Target Variable
    startIndex = ndarray((125279,),int)
    i=0
    a=z
        
    del startIndex
    
    z.insert(loc=0, column='Index', value=0, allow_duplicates=True)
    
    dataset.to_csv("Bikes_new.csv", encoding='utf-8',index=False)
    
    #Data after manipulation
    
    
    
       int(x)
    i=j=0
    for i in range(294928):
        for j in range(294928):
            if int(dataset['StartDay'][i]) == 2 and int(dataset['StartMonth'][j]) == 1:
                print(i)
        
    if dataset['StartDay'][i]
    #quarter based Divisione
    
    for i in range(294928):
        pd.to_datetime(dataset['Stop Time'][i])
    
    x=dataset['Stop Time'][i]
   
    dataset['Start Time'][0].isoformat(timespec='auto')
    del df['StartTime']
    del dataset['Demand']
    
    dataset.insert(loc=21, column='Q1', value=0, allow_duplicates=True)
    dataset.insert(loc=22, column='Q2', value=0, allow_duplicates=True)
    dataset.insert(loc=23, column='Q3', value=0, allow_duplicates=True)
    dataset.insert(loc=24, column='Q4', value=0, allow_duplicates=True)
    
    from numpy import ndarray

    a = ndarray((125279,),int)
    
    for i in range(125279): a[i]=0
    
    dataset.insert(loc=22, column='HourOfIssue', value=0, allow_duplicates=True)
    
    for i in range(294928):
        pd.to_datetime(dataset['Start Time'][i]).time()
        
        a[i]=HR.hour
        
        dataset['Start Hour'] = 
        
    #    a[0]=HR.hour
    
    
    from numpy import ndarray
    tripcnt = ndarray((24,),int)
    for i in range(24): tripcnt[i]=0 
    
    from numpy import ndarray
    trip = ndarray((24,),int)
    for i in range(24): trip[i]=0 
    
    
    from numpy import ndarray
    tripavg = ndarray((24,),int)
    for i in range(24): tripavg[i]=0 
    
    #Holiday addition
    i=0
    for i in range(294928):  
        if dataset['StartDay'][i] == '02' & dataset['StartMonth'][i] == '01':
            dataset['Holiday']=1
    
    x=dataset['StartDay'][0]
    
    import ast
    ast.literal_eval(x)
    
    
    # trips avg & busiest hrs of the year 
    
    trip = ndarray((24,),int)
    for i in range(24): trip[i]=0 
    tripcnt = ndarray((24,),int)
    for i in range(24): tripcnt[i]=0 
    
    i=0    
    
    for i in range(294928):
        if dataset['HourOfIssue'][i]==0:        
            trip[0]+=dataset['Trip Duration'][i]
            tripcnt[0]+=1            
        elif dataset['HourOfIssue'][i]==1:
            trip[1]+=dataset['Trip Duration'][i]
            tripcnt[1]+=1
        elif dataset['HourOfIssue'][i]==2:
            trip[2]+=dataset['Trip Duration'][i]
            tripcnt[2]+=1
        elif dataset['HourOfIssue'][i]==3:
            trip[3]+=dataset['Trip Duration'][i]
            tripcnt[3]+=1
        elif dataset['HourOfIssue'][i]==4:
            trip[4]+=dataset['Trip Duration'][i]
            tripcnt[4]+=1
        elif dataset['HourOfIssue'][i]==5:
            trip[5]+=dataset['Trip Duration'][i]
            tripcnt[5]+=1                        
        elif dataset['HourOfIssue'][i]==6:
            trip[6]+=dataset['Trip Duration'][i]
            tripcnt[6]+=1                        
        elif dataset['HourOfIssue'][i]==7:
            trip[7]+=dataset['Trip Duration'][i]
            tripcnt[7]+=1                        
        elif dataset['HourOfIssue'][i]==8:
            trip[8]+=dataset['Trip Duration'][i]
            tripcnt[8]+=1                        
        elif dataset['HourOfIssue'][i]==9:
            trip[9]+=dataset['Trip Duration'][i]
            tripcnt[9]+=1                        
        elif dataset['HourOfIssue'][i]==10:
            trip[10]+=dataset['Trip Duration'][i]
            tripcnt[10]+=1                        
        elif dataset['HourOfIssue'][i]==11:
            trip[11]+=dataset['Trip Duration'][i]
            tripcnt[11]+=1                        
        elif dataset['HourOfIssue'][i]==12:
            trip[12]+=dataset['Trip Duration'][i]
            tripcnt[12]+=1                        
        elif dataset['HourOfIssue'][i]==13:
            trip[13]+=dataset['Trip Duration'][i]
            tripcnt[13]+=1                        
        elif dataset['HourOfIssue'][i]==14:
            trip[14]+=dataset['Trip Duration'][i]
            tripcnt[14]+=1                        
        elif dataset['HourOfIssue'][i]==15:
            trip[15]+=dataset['Trip Duration'][i]
            tripcnt[15]+=1                   
        elif dataset['HourOfIssue'][i]==16:
            trip[16]+=dataset['Trip Duration'][i]
            tripcnt[16]+=1                        
        elif dataset['HourOfIssue'][i]==17:
            trip[17]+=dataset['Trip Duration'][i]
            tripcnt[17]+=1                        
        elif dataset['HourOfIssue'][i]==18:
            trip[18]+=dataset['Trip Duration'][i]
            tripcnt[18]+=1                        
        elif dataset['HourOfIssue'][i]==19:
            trip[19]+=dataset['Trip Duration'][i]
            tripcnt[19]+=1                   
        elif dataset['HourOfIssue'][i]==20:
            trip[20]+=dataset['Trip Duration'][i]
            tripcnt[20]+=1                        
        elif dataset['HourOfIssue'][i]==21:
            trip[21]+=dataset['Trip Duration'][i]
            tripcnt[21]+=1                        
        elif dataset['HourOfIssue'][i]==22:
            trip[22]+=dataset['Trip Duration'][i]
            tripcnt[22]+=1                        
        elif dataset['HourOfIssue'][i]==23:
            trip[23]+=dataset['Trip Duration'][i]
            tripcnt[23]+=1
                                  
    from numpy import ndarray
    bph = ndarray((31,),int)
    for i in range(31): bph[i]=0 
    
    #trip average
    for i in range(24):
        tripavg[i]=trip[i]/tripcnt[i]
    
    
   # ctr0=ctr1=ctr2=ctr3=ctr4=ctr5=ctr6=ctr7=ctr8=ctr9=ctr10=   ctr11=ctr12=ctr13=ctr14=ctr15=ctr16=ctr17=ctr18=ctr19=ctr20=ctr21=ctr22=ctr23=0
    
    df = pd.read_csv('time.csv')
    df['Bikes Issued']=ctr
    dataset.insert(loc=24, column='TripAvg/Hr', value=0, allow_duplicates=True)
    df['TripAvg/Hr']=tripavg
    
    data = pd.read_csv('dataFresh.csv')
    
    data[[]].groupby(['Start Hour'])
    bull
    
    df.groupby(['Start Station ID','Start Hour']).size().reset_index().groupby('col2')[[0]].max()
    
   i=0     
    for i in range(12201):
        j=1
        for j in range(31):
            if data['StartDay'][i]==j:
                bph[j]+=1
                j+=1
                
                
   #dock station wise classification 
    # approximate radius of earth in km
    R = 6373.0
    
    lat1 = radians(52.2296756)
    lon1 = radians(21.0122287)
    lat2 = radians(52.406374)
    lon2 = radians(16.9251681)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    haversine.distance(sqrt(bph*error-const(dlat)))
    
    distance = R * c
    
    print("Result:", distance)
    
    # Should be 278.546
    from . import mpu
    
    lat1 = 52.2296756
    lon1 = 21.0122287
    
    # Point two
    lat2 = 52.406374
    lon2 = 16.9251681
    
    # What you were looking for
    dist = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
    print(dist)
    
    # Another Method
    tpcnt = ndarray((2,),int)
    for i in range(2): tpcnt[i]=0 
    
    i=0
    for i in range(12): mcnt[i]=0
    
    for i in range(294928):
        if int(dataset['StartMonth'][i])==1:        
            mcnt[0]+=1
        elif int(dataset['StartMonth'][i])==2:        
            mcnt[1]+=1
        elif int(dataset['StartMonth'][i])==3:        
            mcnt[2]+=1
        elif int(dataset['StartMonth'][i])==4:        
            mcnt[3]+=1
        elif int(dataset['StartMonth'][i])==5:        
            mcnt[4]+=1
        elif int(dataset['StartMonth'][i])==6:        
            mcnt[5]+=1
        elif int(dataset['StartMonth'][i])==7:        
            mcnt[6]+=1
        elif int(dataset['StartMonth'][i])==8:        
            mcnt[7]+=1
        elif int(dataset['StartMonth'][i])==9:        
            mcnt[8]+=1
        elif int(dataset['StartMonth'][i])==10:        
            mcnt[9]+=1
        elif int(dataset['StartMonth'][i])==11:        
            mcnt[10]+=1
        elif int(dataset['StartMonth'][i])==12:        
            mcnt[11]+=1
    i=0
    
    tpcnt = ndarray((2,),int)
    for i in range(2): tpcnt[i]=0 
   
    for i in range(294928):
        if dataset['Holidays'][i] == 0:
            tpcnt[0]+= 1 
        else:
            tpcnt[1]+=1
            
            
    finalData = pd.read_csv('Finale.csv')
    wdata
    
    finalData.insert(loc=3, column='Temp', value=0, allow_duplicates=True) 
    finalData.insert(loc=4, column='Humidity', value=0, allow_duplicates=True) 
    finalData.insert(loc=5, column='Precipitation', value=0, allow_duplicates=True) 
    finalData.insert(loc=6, column='Visibility', value=0, allow_duplicates=True) 
    finalData.insert(loc=7, column='WindSpeed', value=0, allow_duplicates=True) 
              
    demo= finalData
    demo['Date']
    
    demo.columns=['tripDate','Hour','Station ID','Temp','Net Bike Count','Humidity', 'Rain/Snow', 'Visibility', 'WindSpeed', 'Holidays']
    demo['tripDate'][0]
    
    i=0
    j=0
    for i in range(177399):
        for j in range(365):
            if demo['tripDate'][i] == wdata['Date'][j]:
                demo['Temp'][i] = wdata['Temp'][j]
                demo['Humidity'][i] = wdata['Temp'][j]
                demo['Precipitation'][i] = wdata['Rain/Snow'][j]
                demo['WindSpeed'][i] = wdata['Wind Speed'][j]
                demo['Visibility'][i] = wdata['Visibility'][j]
            
            
    finalData = Trial 
    
    Trial.to_csv("Trial.csv", encoding='utf-8',index=False)
    
    #MODEL FITTING
    
    
    bikeData = pd.read_csv('demo.csv')
    
    Trial = bikeData
    
    import numpy as np
    from sklearn.model_selection import train_test_split
       
    
    train, test = train_test_split(bikeData, test_size=0.2)
    
    Data1.to_csv("Bull.csv", encoding='utf-8',index=False)
    Data2.to_csv("2017bkup.csv", encoding='utf-8',index=False)
   
    import datetime as dt
    Try['tripDate'] = pd.to_datetime(Try['tripDate'])
    Trial['tripDate'] = Trial['tripDate'].map(dt.datetime.toordinal)
    
    Trial['tripDate'] = Trial['tripDate'].map(dt.datetime.fromordinal)
    
    Try=Trial
    
    x = [d.split(' ')[0] for Try['tripDate']]
    x=[d.split(' ')[0] for Try]
    
    y=[d.split(' ')[0] for d in x]
    
    #  the stations
    demo1=Data16
    demo2=Data17
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    del Data16
    Data1 = pd.read_csv('Oldyear.csv')
    Data2 = pd.read_csv('Newyear.csv')
    
    #HOUR ENCODING
    L1 = LabelEncoder()
    O1 = OneHotEncoder()
    demo1['Hour_Encoded'] = L1.fit_transform(demo1['Hour'])
    XM1 = O.fit_transform(demo1['Hour_Encoded'].values.reshape(-1,1)).toarray()
    XM1
    OneHot1 = pd.DataFrame(XM1, columns = ["Hour_"+str(int(i)) for i in range(XM1.shape[1])])
    demo1 = pd.concat([demo1, OneHot1], axis=1)
    
    L2 = LabelEncoder()
    O2 = OneHotEncoder()
    demo2['Hour_Encoded'] = L2.fit_transform(demo2['Hour'])
    XM2 = O.fit_transform(demo2['Hour_Encoded'].values.reshape(-1,1)).toarray()
    XM2
    OneHot2 = pd.DataFrame(XM2, columns = ["Hour_"+str(int(i)) for i in range(XM2.shape[1])])
    demo2 = pd.concat([demo2, OneHot2], axis=1)
    
    #STATION ENCODING
    L11 = LabelEncoder()
    O11 = OneHotEncoder()
    demo1['Station_Encoded'] = L11.fit_transform(demo1['Station ID'])
    XN11 = O11.fit_transform(demo1['Station_Encoded'].values.reshape(-1,1)).toarray()
    XN11
    OneHot11 = pd.DataFrame(XN11, columns = ["Station_"+str(int(i)) for i in range(XN11.shape[1])])
    demo1 = pd.concat([demo1, OneHot11], axis=1)
    
    L22 = LabelEncoder()
    O22 = OneHotEncoder()
    demo2['Station_Encoded'] = L22.fit_transform(demo2['Station ID'])
    XN22 = O22.fit_transform(demo2['Station_Encoded'].values.reshape(-1,1)).toarray()
    XN22
    OneHot22 = pd.DataFrame(XN22, columns = ["Station_"+str(int(i)) for i in range(XN22.shape[1])])
    demo2 = pd.concat([demo2, OneHot22], axis=1)
    
    
    list(demo1)
    
    
    #MODEL LINEAR
    del dataFeed 
    dataFeed = pd.read_csv('dataFeed.csv') 
    from sklearn import datasets, linear_model
    
    del Y
    X = Data2.iloc[:,1:173]
    Y = Data2.iloc[:,0]

    X1 = Data1.iloc[:,1:144]
    Y1 = Data1.iloc[:,0]

    import pandas as pd
    pd.to_numeric(dataFeed['Rain/Snow'])

    df = demo2['tripDate']
    df = df.set_index('date', append=False)
    df = df.index.to_julian_date()
    
    del X_train, X_test, Y_train, Y_test    
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size = 0.3, random_state = 0)

    import datetime as dt
    X['tripDate'] = pd.to_datetime(X['tripDate'])
        
    import datetime as dt
    X1['tripDate'] = pd.to_datetime(X1['tripDate'])
    X1['tripDate'] = X1['tripDate'].map(dt.datetime.toordinal)
    
    regressor
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X,Y)
    X_test.dtypes
    
    Y = Y.values.reshape(-1,1) 
    
    del Y_pred
    Y_pred = Y_pred.reshape(-1,1) 
    
    Y_pred = regressor.predict(X_test)    
    lin_acc = regressor.score(Y_pred,Y_test)
    
    Y_test = Y_test.astype(float) 
    
    list(X_test)
    
    #MODEL LOGISTIC
    dataFeed['Rain/Snow']=dataFeed['Rain/Snow'].astype(float)
    
    del X_train Y_train
    
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)

    Y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set:'.format(logreg.score(X_test, Y_test)))
    
    dataFeed.dtypes
    
    #MODEL SVR
    
    pd.to_numeric(dataFeed)
    from sklearn import svm
    SVM_model = svm.svc() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.

    SVM_model.fit(X_train, Y_train)
    model.score(X_train, Y_train)
    #Predict Output
    Y_pred= model.predict(X_test)
    
    #MODEL DECISION TREE
    
    from sklearn.tree import DecisionTreeRegressor
    regression_tree = tree.DecisionTreeRegressor()
    regression_tree.fit(X_train,Y_train)
    Y_pred = regression_tree. 
    
    #MODEL Random Forest
    
    from sklearn.ensemble import RandomForestRegressor 
    rf_model= RandomForestRegressor()

    rf_model.fit(X_train, Y_train)

    Y_pred= model.predict(X_test)
    
    #Creating Merged file for both 
    val1= counts1.index.values
    val1= pd.DataFrame(val1)
    val2= counte1.index.values
    val2= pd.DataFrame(val2)

    val3= val1.merge(val2, how= 'outer')
    #val3['Trip Duration']=0
    #val3= val3.drop(['Trip Duration'], axis=1)
    val3.columns=['idx']
    val3= val3.set_index(val3.iloc[:,0])
    val4= val3.copy()
    val4['Trip Duration']= -counts1['Trip Duration']
    
    val4.columns=['index1','Trip Duration']
    val4= val4.reset_index()           
    
    val6= np.intersect1d(val1[0], val2[0]).tolist()

    counte1['index1']=counte1.index
    counte1= counte1.reset_index()
    
    
    #Assigning values based on logic
    count=0 
    i=0
    for i in range(counts.shape[0]):
        if(val4.iloc[i]['idx'] in val6):
            x1= np.where(counte1['index1'] == val4.iloc[i]['idx'])
            val4.loc[i,'Trip Duration']=val4.loc[i]['Trip Duration'] + counte1.iloc[x1[0][0]]['Trip Duration']
            print(count)
            count=count+1
    
    count=0
    #i=125279
    #104083
    #for i in range(125279,val4.shape[0]):
    for i in range(104083,val4.shape[0]):
        x2= np.where(counte1['index1'] == val4.iloc[i]['idx'])
        val4.loc[i,'Trip Duration']= counte1.iloc[x2[0][0]]['Trip Duration']
        count= count+1
        print(count)

    #saving Net Bike Flow
    val4.to_csv("NetFlow1.csv", encoding='utf-8',index=True)
    
    
    finalData = pd.read_csv('NetFlowTemp.csv')
    finalData = pd.read_csv('NetFlowFinal2016.csv')
    
    finalData= finalData.loc[0:177399,'tripDate':'Wind Speed']
    #finalData= finalData.loc[0:177399,'tripDate']
    
    #finalData.columns=['tripDate','Hour','Station ID']

    finalData.insert(loc=4, column='Temp', value=0, allow_duplicates=True) 
    finalData.insert(loc=5, column='Humidity', value=0, allow_duplicates=True) 
    finalData.insert(loc=6, column='Rain/Snow', value=0, allow_duplicates=True) 
    finalData.insert(loc=7, column='Visibility', value=0, allow_duplicates=True) 
    finalData.insert(loc=8, column='Wind Speed', value=0, allow_duplicates=True)
    finalData.insert(loc=9, column='Holidays', value=0, allow_duplicates=True)
    
         
    for i in range(177399):
        wdata['tripDate'][i] = pd.to_datetime(wdata['tripDate'][i])

    wdata['tripDate']= pd.to_datetime(wdata['tripDate'])
    finalData['tripDate']= pd.to_datetime(finalData['tripDate'])
    demo=finalData
    wdata = pd.read_csv('Climate_1.csv')
    wdata = pd.read_csv('Climate2017.csv')
    wdata= wdata.drop(columns='Trips')
    wdata = pd.read_csv('climate2016.csv')
    wdata.columns=['tripDate','Temp','Humidity','Rain/Snow','Visibility','Wind Speed','Holidays']

    wdata= wdata.drop(columns=['Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10',
       'Unnamed: 11'])
    wdata.columns
    
    #i=315
    #j=203
    
    #from pandas import to_datetime
    #demo['tripDate']= pd.to_datetime(demo['tripDate']).date() to_datetime(demo['tripDate']).date()
    

    
    demo.dtypes
    wdata.dtypes
    #for i in range(0,177399):
    #finalData.iloc[177398,:]
    #finalData.iloc[177399,:]
    
    
     # Copying All weather data from climate file to Net Flow of Bikes
    i=0
    #i=177399
    while i < 177399:
    #while i < 147405:
        x3=np.where(wdata['tripDate']==demo['tripDate'][i])
        x4=np.where(demo['tripDate']==demo['tripDate'][i])
        #demo[np.where(demo['tripDate']==demo['tripDate'][i])]
        #demo(np.where(demo.index in x4[0])[0])        
        i=i+len(x4[0])
        print(i)
        #for j in range(x4[0][0],x4[0][0]+len(x4[0])-1):
        for j in range(len(x4[0])):
            #print(j)
            #demo.iloc[x4[0][j]]['Temp':'WindSpeed']= wdata.loc[x3[0][0]]['Temp':'Wind Speed']
            y=x4[0][j]
            demo.loc[y,'Temp':'Holidays']= wdata.loc[x3[0][0]]['Temp':'Holidays']
     
        
     # Exporting Net Flow with weather data
    demo.to_csv("demo1.csv", encoding='utf-8',index=True)
    

#############      Fitting Model       ##############
    del demo1
    demo1 = pd.read_csv('FinalData_1.csv')
    demo1 = pd.read_csv('NetFlowTemp.csv')
    demo1['Net Bike Count'].value_counts()
    demo1.summary
    #HOUR ENCODING
    L1 = LabelEncoder()
    O1 = OneHotEncoder()
    demo1['Hour_Encoded'] = L1.fit_transform(demo1['Hour'])
    XM1 = O1.fit_transform(demo1['Hour_Encoded'].values.reshape(-1,1)).toarray()
    XM1
    OneHot1 = pd.DataFrame(XM1, columns = ["Hour_"+str(int(i)) for i in range(XM1.shape[1])])
    demo1 = pd.concat([demo1, OneHot1], axis=1)
  

    #STATION ENCODING
    L2 = LabelEncoder()
    O2 = OneHotEncoder()
    demo1['Station_Encoded'] = L2.fit_transform(demo1['Station ID'])
    XM2 = O2.fit_transform(demo1['Station_Encoded'].values.reshape(-1,1)).toarray()
    XM2
    OneHot2 = pd.DataFrame(XM2, columns = ["Station_"+str(int(i)) for i in range(XM2.shape[1])])
    demo1 = pd.concat([demo1, OneHot2], axis=1)
    
    del X,Y
    X = demo1.iloc[:,1:206]
    Y = demo1.iloc[:,0]
    
    X= X.drop(columns='Hour_Encoded')
    X= X.drop(columns='Station_Encoded')
    
    import datetime as dt
    X['tripDate'] = pd.to_datetime(X['tripDate'])
    X['tripDate'] = X['tripDate'].map(dt.datetime.toordinal)
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0    
        
    
    
    ########################            MODELS              ###############################
    
    #LINEAR REGRESSION
    from sklearn.linear_model import LinearRegression
    regressorLin = LinearRegression()
    regressorLin.fit(X_train,Y_train)
    X_test.dtypes
    Y_pred = regressorLin.predict(X_test)    
    Y = Y.reshape(-1,1) 
    
    
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.metrics import mean_absolute_error

    mse_lin= mean_squared_error(Y_test, Y_pred)
    rmse_lin= sqrt(mse_lin)
    rmsle_lin= np.log10(rmse_lin)
    rmse_lin_crossval = np.sqrt(-cross_val_score(regressorLin, X, Y, scoring = 'neg_mean_squared_error', cv = 10))
    mae_lin= mean_absolute_error(Y_test, Y_pred)
    rmse_lin_crossval.mean()
    
    
    #RANDOM FOREST
    from sklearn.ensemble import RandomForestRegressor 
    rf_model= RandomForestRegressor(n_estimators=250)
    
    
    rf_model_200= RandomForestRegressor(n_estimators=200)
    rf_model_200.fit(X_train, Y_train)    
    Y_pred_200= rf_model_200.predict(X_test)
    mse_rf_200= mean_squared_error(Y_test, Y_pred_200)
    rmse_rf_200= sqrt(mse_rf_200)
    rmsle_rf_200= np.log10(rmse_rf_200)
    mae_rf_200= mean_absolute_error(Y_test, Y_pred_200) 
    
    rf_model_150= RandomForestRegressor(n_estimators=150)
    rf_model_150.fit(X_train, Y_train)    
    Y_pred_150= rf_model_150.predict(X_test)
    mse_rf_150= mean_squared_error(Y_test, Y_pred_150)
    rmse_rf_150= sqrt(mse_rf_150)
    rmsle_rf_150= np.log10(rmse_rf_150)
    mae_rf_150= mean_absolute_error(Y_test, Y_pred_150)    
    
    rf_model_100= RandomForestRegressor(n_estimators=100)
    rf_model_100.fit(X_train, Y_train)    
    Y_pred_100= rf_model_100.predict(X_test)
    mse_rf_100= mean_squared_error(Y_test, Y_pred_100)
    rmse_rf_100= sqrt(mse_rf_100)
    rmsle_rf_100= np.log10(rmse_rf_100)
    mae_rf_100= mean_absolute_error(Y_test, Y_pred_100)    
    
    rf_model_50= RandomForestRegressor(n_estimators=50)
    rf_model_50.fit(X_train, Y_train)    
    Y_pred_50= rf_model_50.predict(X_test)
    mse_rf_50= mean_squared_error(Y_test, Y_pred_50)
    rmse_rf_50= sqrt(mse_rf_50)
    rmsle_rf_50= np.log10(rmse_rf_50)
    mae_rf_50= mean_absolute_error(Y_test, Y_pred_50)
    
    from sklearn.metrics import mean_squared_log_error
    np.sqrt(mean_squared_log_error(Y_test, Y_pred_50))

    rf_model.fit(X_train, Y_train)
    Y_pred_rf = rf_model.predict(X_test) 
    mse_rf= mean_squared_error(Y_test, Y_pred_rf)
    rmse_rf= sqrt(mse_rf)
    rmsle_rf= np.log10(rmse_rf)
    mae_rf= mean_absolute_error(Y_test, Y_pred_rf)
    
    rf_model_2= RandomForestRegressor(n_estimators=10)
    rmse_rf_crossval = np.sqrt(-cross_val_score(rf_model_2, X, Y, scoring = 'neg_mean_squared_error', cv = 10))


    #Plotting importance
    importances = rf_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    indice= indices[0:15]
    
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(0,len(indice)), importances[indice],
            color="r", yerr=std[indice], align="center")
    plt.xticks(range(0,len(indice)), X.columns[indice], rotation = 45)
    plt.xlim([-1,len(indice)])
    plt.show()
    
    
    
    #LOGISTIC REGRESSION
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    
    Y_pred_log = logreg.predict(X_test)
    mse_log= mean_squared_error(Y_test, Y_pred_log)
    rmse_log= sqrt(mse_log)
    rmsle_log= np.log10(rmse_log)
    mae_log= mean_absolute_error(Y_test, Y_pred_log)
    
    rmse_log_crossval = np.sqrt(-cross_val_score(logreg, X, Y, scoring = 'neg_mean_squared_error', cv = 10))
    
    #SVM
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    
    Y_pred_svm= clf.predict(X_test)
    mse_svm= mean_squared_error(Y_test, Y_pred_svm)
    rmse_svm= sqrt(mse_svm)
    rmsle_svm= np.log10(rmse_svm)
    mae_svm= mean_absolute_error(Y_test, Y_pred_svm)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    