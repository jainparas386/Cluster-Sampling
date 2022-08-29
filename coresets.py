
import numpy as np

#lightweight coreset subsampling

def lightweight(data, m):


    """Reduces (NxD) data matrix from N to Np data points.
    Args:
        data: ndarray of shape [N, D]
        Np: number of data points in the coreset
    Returns:
        coreset: ndarray of shape [Np, D]
        weights: 1darray of shape [Np, 1]
    """
    N = data.shape[0]
    # compute mean
    u = np.mean(data, axis=0)

    # compute proposal distribution
    q = np.linalg.norm(data - u, axis=1)**2
    sum = np.sum(q)
    q = 0.5 * (q/sum + 1.0/N)


    # get sample and fill coreset
    samples = np.random.choice(N, m, replace=False ,p=q)
    coreset = data.iloc[samples]
    weights = 1.0 / (q[samples] * m)
    
    return coreset, weights

# uniform coreset subsampling

def uniform(data,m):
    N = data.shape[0]
    # compute mean

    # get sample and fill coreset
    samples = np.random.choice(N,m, replace=False)
    coreset = data.iloc[samples]
    weights = np.full(m,N/m)
    
    return coreset, weights

# adaptive coreset
def myfunc(e):
    return e[1]

def adaptive(data,m):
      
    data1=data
    b=[]

    # while loop
    while data1.shape[0]>20:
        samples=np.random.choice(data1.shape[0],20,replace=False)
        s=data1.iloc[samples]
        dist=[]
        # removing half points
        for i in range(len(data1)):
            dist.append(min(np.linalg.norm(s-data1.iloc[i],axis=1)))
        for ind,val in enumerate(dist):
            dist[ind]=(ind,val)
        dist.sort(key=myfunc)
        data2=[]
        for i in range(int(len(dist)/2), len(dist)):
            data2.append(data1.iloc[dist[i][0]])
        data1=pd.DataFrame(data2)
        b.extend(np.array(s))

    min_dist=[]
    b=pd.DataFrame(data=b, columns=data.columns)
    for i in range(len(data)):
        min_dist.append(min(np.linalg.norm(b-data.iloc[i],axis=1)**2))
    suma=sum(min_dist)
    ma=[]
    for i in range(len(data)):
        ma.append(min_dist[i]/suma)
    pr=[]
    suma=sum(ma)
    for i in range(len(data)):
        pr.append(ma[i]/suma)
    sample=np.random.choice(data.shape[0],m,replace=False,p=pr)
    c=data.iloc[sample]
    weights=[]
    for sam in sample:
        weights.append(1/(pr[sam]*m))
     
    return c, weights
