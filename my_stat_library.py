import numpy as np
import time
import math
import random




# Computes SQUARED Euclidian distance between point A and point B. Incredibly enough this trivial implementation is faster than the distance functions in numpy!
def dist2(a,b): 
    dist2=0
    Ndim=len(a) #Number of dimensions of the data (es: if the data contain the variables Age,Sex,FinalPremium, Ndim=3)
    if (len(b)!=Ndim) : print "The two array have different length"   
    for i in range(Ndim) : #the distance d = sqrt((x1-y1)^2+(x2-y2)^2...) is defined as the usual cartesian distance of two points in number of dimensions = Ndim
        dist2+=(a[i]-b[i])*(a[i]-b[i])
    return dist2
    
# Compute the scalar Euclidian distance between point A and point B
def dist(a,b):
    return math.sqrt(dist2(a,b))

# Sets the initial N=K centres on random chosen data
def set_ForgyStart(x,K):  
    Ndat=len(x)
    centre=[]
    rndm_indx=random.sample(np.arange(Ndat), K)
    for i in range(K):
        centre.append(x[rndm_indx[i],:])
    return centre
    
# Sets the initial centers having random coordinates (not necessarely corresponding to any data)
def set_randomCentreStart(K,xmin,xmax,ymin,ymax):
    centre=[]
    for i in range(K):
        centre.append([random.uniform(xmin,xmax),random.uniform(ymin,ymax)]) ## sets a random centre inside the range of datas 
    return centre


##sets the initial centres as equispaced between each other as possible (Kmeans++ by David Arthur and Sergei Vassilvitskii)
def Prob_dist(x,centr):
    Np=len(x[:,0]) ## number of points
    tot_dist2=0
    for i in range(Np):  ## this for cycle computes the normalization, i.e. the sum of the distances squared of each point witht he nearest centre
        centr_indx=assign_toCluster(x[i,:],centr) ## return index of closest centre
        d=dist(x[i,:],centr[centr_indx])  
        tot_dist2+=d*d  ## sum of distance^2 of each point with its nearest centre. It's used to normalized the dist^2 based probability
    prob_dist=np.zeros(Np)
    for i in range(Np): ## this for computes the probability of each point based on d^2
        centr_indx=assign_toCluster(x[i,:],centr) ## return index of closest centre
        d=dist(x[i,:],centr[centr_indx])  
        prob_dist[i]=d*d/tot_dist2
    return prob_dist
        
def set_KplusplusStart(x,K):
    from random import randint
    centre=[]
    Ndat=len(x[:,0])
    ind_start=randint(0,Ndat-1)
    print ind_start
    centre.append(x[ind_start,:]) ## 1) choose the first centre at random

    for j in range(K-1):
        prob=Prob_dist(x,centre) #returns an array of probability of each point to be choosen as a centre
        #print prob
        rnd_indx=np.arange(Ndat)
        np.random.shuffle(rnd_indx)
        #print rnd_indx
        stop=False
        while(stop==False):
            for i in range(Ndat):
                a=random.random()
                if(a<prob[rnd_indx[i]]): #samples at random (rnd_indx) the datapoints and asign them to the centre only if a random number is smaller than the probability defined (Prob_dist)
                    #print a,prob[rnd_indx[i]]
                    centre.append(x[rnd_indx[i],:])
                    #print stop
                    stop=True #if random is never < prob the loop has to be repeted, untill a centre is found (then stop=True)
                    #print stop
                    break
    return centre

## Assign an element x[i] to the cluster with the nearest centre
def assign_toCluster(x,Centre) :  ###return the index of the cluster which the point x is closest to
    K=len(Centre)
    Min=dist(Centre[0],x)  ## initialize Min to the first cluster
    indx=0               
    for i in range(K):   ## find the min distance between the clusters' centres
        d=dist(Centre[i],x)
        if Min>d: 
            Min=d
            indx=i 
    return indx


## calculates the new centres of the clusters
def Get_NewCentres(Cluster,K):
    new_centre=[]
    for i in range(0,K):  
        a=np.array(Cluster[i]) #each row of a consists in all the data in the i-th cluster
        new_centre.append(np.mean(a,axis=0)) ## computes the average coordinates of the points in cluster i, and saves them in new_centre[i]
    return new_centre
    
## Redefines the clusters based on the new centres
def step_Clustering(data,centre,K):
    Cluster=[]    
    for j in range(K):Cluster.append([]) ## Initialize empty cluster object of the dimensions of the number of clusters K    
    for i in range(len(data[:,0])) :
        clust_indx=assign_toCluster(data[i,:],centre)
        Cluster[clust_indx].append(data[i,:])
    for j in range(K):Cluster[j]=np.array(Cluster[j]) #makes sure the elements of each Cluster are in NUMPY FORMAT
    return Cluster    
    
## My homemade version of K-Means (10 times slower than the sklearn one and more unstable :'( )
def my_Kmeans(x,K,centres_start,time): 
    centres_new=centres_start  
    clusters=step_Clustering(x,centres_start,K)
       
    if(time==True): start = time.clock()
    Ndim=len(x[0,:])
    centres_old=np.zeros((K,Ndim))
    count=0
    
    #centres_old=centres_new
    print 'dio',centres_new
    print 'porco',centres_old
    while(np.not_equal(centres_new,centres_old).any()==True): #keeps going untill a stable minimum is found (i.e. the centres of the clusters don't change anymore (continues untill there is AT LEAST ONE still different (not_equal.any() ==True)))
        centres_old=centres_new
        clusters=step_Clustering(x,centres_old,K)
        centres_new=Get_NewCentres(clusters,K)
        print 'dio1',centres_new
        print 'porco1',centres_old
        count+=1
    if(time==True): 
        stop = time.clock()
        print 'time the for Kmean : %.2f secs' % (stop-start)
    return np.array(clusters),np.array(centres_new),count




   
   
   
   
   
   
   
   
   
   
   
   
   
    
