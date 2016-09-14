import time
import pyodbc
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy import stats
from scipy.interpolate import UnivariateSpline

###########################################################################################################################################################################
#######-------------------- DATA READ/WRITE ----------------------------------------------------#########################################################
###############################################################################################################################################

def make_Folders(string):
    folderName=string
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    folderPlot=str(folderName+'//Plots')
    if not os.path.exists(folderPlot): ### Create Plot directory if doesn't exists already
        os.makedirs(folderPlot)
    return folderPlot
        
        
        
def load_SQLdata(database_connect,sql_query,Time): 
    if(database_connect=='QDW') : conn_string='DRIVER={SQL Server};SERVER=217.10.152.6;DATABASE=Riskdata;UID=iymqdw;PWD=cDfBn8DDaVW8hvst'
    elif(database_connect=='OGI') : conn_string='DRIVER={SQL Server};SERVER=10.33.23.247;DATABASE=OpenGi;UID=ReportsUser;PWD=reportuser'
    else : print "Error in the database selected! The possible values of database_connect are : 'QDW' or 'OGI' "

    cnxn = pyodbc.connect(conn_string)
    #x=pd.DataFrame()
    if(Time==True):start = time.clock()
    x=pd.read_sql_query(sql_query,cnxn)  #load the informations for the day in the pd.Dataframe data_day
    if(Time==True):
        stop = time.clock()
        print 'time the for executiong SQL query : %.1f secs' % (stop-start)
    cnxn.close()     #<--- Close the connection
    return x

def load_Refs(conn_string,sql_query,Time):
    data=load_SQLdata(conn_string,sql_query,Time)
    Refs="'"+str(data.iloc[0,0])
    for i in range(1,data.shape[0]):
        Refs+="','"+str(data.iloc[i,0])
    Refs+="'"
    return Refs

### WRITE DATA TO EXCEL FILE     
def write_toExcel(x,filename) :
    start = time.clock()
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')  # Create a Pandas Excel writer using XlsxWriter as the engine.
    x.to_excel(writer, sheet_name='Sheet1',header=True,index=False)    ##saves the DataFrames in an Excel file, each new query is written under the previous one (startrow=4*i)
    stop = time.clock()
    print 'time the for writing on excell file : %.1f secs' % (stop-start)
    writer.save() # Close the Pandas Excel writer and output the Excel file.
    
### READ FROM EXCEL FILE and return numpy array 
def read_fromExcel(file_name):
    start = time.clock()
    tab = pd.read_excel(file_name,header = 0,index_col = None,convert_float=True)
    stop = time.clock()
    print 'time the for importing data from Excell : %.1f secs' % (stop-start)
    #tab=tab.as_matrix()
    return tab

    
    
    
    
    
###############################################################################################################################################
#######-------------------- PLOT ----------------------------------------------------#########################################################
###############################################################################################################################################

def Plot_Histo(x,save_str,NBin,col,lab,norm): 
    plt.clf()
    plt.hist(x,NBin,histtype='bar', color=col, label=lab,rwidth=0.88,normed=norm)
    plt.legend(bbox_to_anchor=(0.05, .92, 1., .104))
    plt.xlabel("Premium in pounds")
    plt.ylabel('Number of ')
    axes = plt.gca()          ##set axis range
    #axes.set_xlim([0,5000])
    plt.title("distribution as a function of Premium")
    #plt.title("Drift as a function of Premium")
    plt.grid(True)
    plt.show()
    plt.savefig(save_str)  


def autolabel(rects,data,perc,ax): ###-------------- attach some text labels to the histo bars
    count=0
    for rect in rects:
        height = rect.get_height()
        if (perc==True and int(data[count])>100) : ax.text(rect.get_x() + 0.5*rect.get_width(), 1.02*height,"%d%%" % int(data[count]),ha='center', va='bottom', rotation=90,fontsize=9)
        elif (perc==True and int(data[count])<100) : ax.text(rect.get_x() + 0.5*rect.get_width(), 1.02*height,"%.1f%%" % float(data[count]),ha='center', va='bottom', rotation=90,fontsize=9) ## if the displayed value is a fraction of 0, displays also the decimal part
        elif (perc==False) : ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,"%d" % int(data[count]),ha='center', va='bottom', rotation=90,fontsize=9)
        count+=1

		

def plot_Barplot_1set(x,ind,colNames,Labels,bar_width,Title,Xaxis,Yaxis,perc,spline_Fit,spline_smooth,save,save_str):
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, x, bar_width, color='b',alpha=1)
    ax.set_ylabel(Yaxis)
    ax.set_xlabel(Xaxis)
    ax.set_title(Title)
    ax.grid(False)
    ymax=np.amax(x) ## finds higher y value for axis limit
    ax.set_ylim(0,1.1*ymax)
    ax.set_xlim(np.amin(ind),np.amax(ind)+1)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    ax.set_xticks(ind + bar_width/2.)
    xtickNames=ax.set_xticklabels((colNames))
    ax.legend()#rects1[0],Labels)#,bbox_to_anchor=(0.45, 1), loc=2)
    plt.legend()
    plt.axvspan(xmin=15.85,xmax=20, facecolor='r', alpha=0.3) # colours the area between x=0 and x= 14.85
    plt.axvline(x=15.85, ymin=0, ymax = 1.1*ymax, linewidth=3, color='r') # red vertical line at x= 15.85
    plt.axvline(x=14.85, ymin=0, ymax = 1.1*ymax, linewidth=3, color='g') # green vertical line at x= 14.85
    plt.axvspan(xmin=0,xmax=14.85, facecolor='g', alpha=0.3) # colours the area between x=0 and x= 14.85
    plt.setp(xtickNames, rotation=90, fontsize=9)
    autolabel(rects1,x,perc,ax)
    if(spline_Fit==True):
        spl = UnivariateSpline(ind+0.5,x,s=spline_smooth ) #fitting spline
        xs = np.linspace(np.amin(ind), np.amax(ind)+0.5, 1000) 
        plt.plot(xs,spl(xs),c='r', lw=3)
        plt.show()
    if save==True : plt.savefig(save_str)
    plt.show() 

def plot_Barplot_2sets(x,ind,colNames,Labels,bar_width,Title,Xaxis,Yaxis,perc,spline_Fit,spline_smooth,save,save_str):
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, x[:,0], bar_width, color='r',alpha=0.3)
    rects2 = ax.bar(ind + bar_width, x[:,1], bar_width, color='b',alpha=0.3)
    ax.set_ylabel(Yaxis)
    ax.set_xlabel(Xaxis)
    ax.set_title(Title)
    ax.grid(True)
    ax.set_xticks(ind + bar_width)
    ymax=np.amax([np.amax(x[:,0]),np.amax(x[:,1])]) ## finds higher y value for axis limit
    ax.set_ylim(0,1.1*ymax)
    ax.set_xlim(np.amin(ind),np.amax(ind)+1)
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.3)
    xtickNames=ax.set_xticklabels((colNames))
    ax.legend((rects1[0], rects2[0]), (Labels),bbox_to_anchor=(0.55, 1), loc=2)
    plt.setp(xtickNames, rotation=90, fontsize=9)
    autolabel(rects1,x[:,0],perc[0],ax)
    autolabel(rects2,x[:,1],perc[1],ax)
    if(spline_Fit==True):
        spl = UnivariateSpline(ind+0.5,x[:,0],s=spline_smooth )## fitting spline
        xs = np.linspace(np.amin(ind), np.amax(ind)+0.5, 1000) 
        plt.plot(xs,spl(xs),c='r', lw=3)
        plt.show()
		
        spl1 = UnivariateSpline(ind+0.5,x[:,1],s=spline_smooth ) ## fitting spline
        xs1 = np.linspace(np.amin(ind), np.amax(ind)+0.5, 1000) 
        plt.plot(xs1,spl1(xs),c='b', lw=3)
        plt.show()		
    if save==True : plt.savefig(save_str)
    plt.show() 
	

    
    
    
###########################################################################################################################################################################
#######-------------------- HISTOGRAMS ----------------------------------------------------#########################################################
###############################################################################################################################################

## Makes 1D histogram with INTEGERS ON THE X AXIS, collapsing all the values > limit_Bin in the last bin (with position limit_Bin+1)
def make_Histo1D_int(x,xMin,xMax):
    Nbins=xMax-xMin
    y=np.zeros((Nbins,2)) 
    for i in range(Nbins): y[i,0]=xMin+i ## sets first column of y as the center (delta/2.) of each bin
    for i in range(len(x)):
        for j in range(Nbins):
            if(x[i]==y[j,0]):y[j,1]+=1
    ##---Bins out of range---##
    coun_tooBig=0  ### groups all the tails of the histogram (xMin<values<xMax) in the first and last bin
    coun_tooSmall=0 
    for i in range(len(x)): 
        if(x[i]> xMax):
            y[Nbins-1,1]+=1
            coun_tooBig+=1
        if(x[i]< xMin):
            y[0,1]+=1
            coun_tooSmall+=1
    if(coun_tooBig>0): print '%d bins bigger than %.1f'%(coun_tooBig,xMax)
    if(coun_tooSmall>0): print '%d bins smaller than %.1f'%(coun_tooSmall,xMin)
    return y
    
   ##------Makes GENERAL histogram, collapsing all the values out of range in the first and last bin (with position 0 and Nbins-1) 
def make_Histo1D(x,xMin,xMax,bins_type,Nbins):
    if(bins_type=='qualit'):
        x_low= np.char.lower(x.astype(str))
        qualit=pd.DataFrame(data=x_low,columns=['col1'])
        dat=pd.DataFrame({'count' : qualit.groupby(['col1']).size()}).reset_index() # don't ask me why, (ref : http://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-object-to-dataframe)
        y=dat.as_matrix()
        print y
        xMax=np.amax(y[:,1])
        xMax=np.amin(y[:,1])
        Nbins=len(y[:,0])
    elif(bins_type=='int'):
        Nbins=xMax-xMin
        y=np.zeros((Nbins,2)) 
        for i in range(Nbins): y[i,0]=xMin+i ## sets first column of y as the center (delta/2.) of each bin
        for i in range(len(x)):
            for j in range(Nbins):
                if(x[i]==y[j,0]):y[j,1]+=1
    elif(bins_type=='float'):
        delta=(xMax-xMin)/float(Nbins)
        y=np.zeros((Nbins,2)) 
        for i in range(Nbins): y[i,0]=xMin+(i+1./2.)*delta ## sets first column of y as the center (delta/2.) of each bin
        for i in range(len(x)):
            for j in range(Nbins):
                if(x[i]>xMin+j*delta and x[i]<xMin+(j+1)*delta):y[j,1]+=1
    else : print "Warning! bins_type is wrong! the only three types admitted are : 'qualit','int','float' "
    ##---Bins out of range---##
    coun_tooBig=0  ### groups all the tails of the histogram (xMin<values<xMax) in the first and last bin
    coun_tooSmall=0 
    for i in range(len(x)): 
        if(x[i]> xMax):
            y[Nbins-1,1]+=1
            coun_tooBig+=1
        if(x[i]< xMin):
            y[0,1]+=1
            coun_tooSmall+=1
    if(coun_tooBig>0): print '%d bins bigger than %.1f'%(coun_tooBig,xMax)
    if(coun_tooSmall>0): print '%d bins smaller than %.1f'%(coun_tooSmall,xMin)
    return y



def Histo(x,N):
    y=np.histogram(x,bins=N)
    x=np.zeros((N,2))
    x[:,0]=y[1][0:N]
    x[:,1]=y[0]
    return x

def Histo2D(x1,x2,N):
    y=np.histogram2d(x1,x2,bins=N)
    print y

## replace NaNs in the array with 0
def NaN_toZero(x):
    for i in range(len(x)):
        if(str(x[i])=='nan'):x[i]=0 
    return x   

    


  
  
  
  
###########################################################################################################################################################################
#######-------------------- CORRELATIONS ----------------------------------------------------#########################################################
###############################################################################################################################################

def make_ContingentTable(x):
    Max=x['tot'].max()
    threshold=0.00*Max
    idx=[]
    for i in range(len(x)):  ### CHECK for threshold
        if(x.iloc[i,2]<threshold or x.iloc[i,2]<1):
            idx.append(i) ## only need the column index (if only just one entry in the column is < threshold, then all the column has to be taken away)
            if(x.iloc[i,0]=='Female'): idx.append(i+1) ## remove both Sexes variable connected to marit
            elif(x.iloc[i,0]=='Male'): idx.append(i-1)
    y=x.drop(x.index[idx])## removes i-esima row (index)
    y=y.pivot(index='sex', columns='X', values='tot')  ##rearranges the data into a contingency table
    idx=[]
    for i in range(y.shape[1]):  ### CHECK for nulls
        if(pd.isnull(y.iloc[0,i]) or pd.isnull(y.iloc[1,i])):idx.append(i)
    y=y.drop(y.columns[idx],axis=1) #removes i-esima colums
    return y
    
def CramerV(chi2,data):
    x=np.array(data)
    Ndat=np.nansum(x)
    df=np.amin([len(x)-1,len(x[0])-1])
    return math.sqrt(chi2/float(Ndat*df))
    
def Correl(x):
    x=make_ContingentTable(x)
    Chi2,P_value,Dof,Tab_expected = sp.chi2_contingency(x)
    V=CramerV(Chi2,x)
    return(Chi2,P_value,V,Dof,Tab_expected)











