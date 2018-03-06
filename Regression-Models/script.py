import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
#import matplotlib.pyplot as plt
import pickle
import sys
import time

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    unique_y=np.unique(y)
    #print unique_y
    k_matmean = np.zeros((1,2))
    y=y.astype(int).flatten().tolist()
    for i in np.nditer(unique_y):
        mean = np.mean(X[np.where(y==i)], axis=0)
        k_matmean= np.vstack((k_matmean,mean))
    k_matmean = np.delete(k_matmean,0,0)
    means=np.transpose(np.reshape(k_matmean,(len(unique_y),X.shape[1])))
    covmat = np.cov(np.transpose(X),bias=True)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    unique_y=np.unique(y)
    k_matmean = []
    covmats = []
    y=y.astype(int).flatten().tolist()
    for i in np.nditer(unique_y):
        
        k_matmean= np.append(k_matmean,X[np.where(y==i)].mean(0),axis=0)
        if i > 1:
            covmats = np.vstack((covmats,np.cov(np.transpose(X[np.where(y==i)]),bias=True)))
        else:
            covmats = np.cov(np.transpose(X[np.where(y==i)]),bias=True)
    means=np.transpose(np.reshape(k_matmean,(len(unique_y),X.shape[1])))
    covmats=np.vsplit(covmats, len(unique_y))
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    inverse_cov = np.linalg.inv(covmat)
    det_cov = np.linalg.det(covmat)
    means=np.transpose(means)
    ypred = np.zeros((1))
    for row in Xtest:
        xmm = np.zeros((1))
        for row1 in means:
            ximm=row-row1
            prob=np.dot(np.dot(ximm,inverse_cov),np.transpose(ximm))
            prob = -1 * np.log(det_cov) - prob
            xmm=np.vstack((xmm,prob))
        xmm=np.delete(xmm,0,0)   
        ypred=np.vstack((ypred,np.argmax(xmm) + 1))
        xmm = []
    ypred = np.delete(ypred, 0, 0)
       
    acc= np.sum(ytest == ypred)
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    means=np.transpose(means)
    ypred = np.zeros((1))
    for row in Xtest:
        xmm = np.zeros((1))
        i = 0
        for row1 in means:
            ximm=row-row1
            inverse_cov = np.linalg.inv(covmats[i])
            det_cov = np.linalg.det(covmats[i])
            prob=np.dot(np.dot(np.transpose(ximm),inverse_cov), ximm)
            prob = -1 * np.log(det_cov) - prob
            xmm=np.vstack((xmm,prob))
            i = i + 1        
        xmm=np.delete(xmm,0,0)   
        ypred=np.vstack((ypred,np.argmax(xmm) + 1))
        xmm = []
    ypred = np.delete(ypred, 0, 0)
    acc= np.sum(ytest == ypred)
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD  
    xtxi = np.linalg.inv(np.dot(np.transpose(X),X))
    xtxit=np.dot(xtxi,np.transpose(X)) 
    w=  np.dot(xtxit,y)                                               
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD  
    idmat=np.identity(X.shape[1])
    inv=np.linalg.inv((lambd*idmat)+(np.dot(np.transpose(X),X)))
    ixt=np.dot(inv,np.transpose(X))
    w=np.dot(ixt,y)                                                  
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD    
    xw=np.dot(Xtest,w)   
    yxw= ytest-xw    
    mse=np.dot(np.transpose(yxw),yxw)/Xtest.shape[0]
    return mse
count = 0
def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    w=np.array(w) 
    w=w.reshape(X.shape[1],1)
    xw= np.dot(X,w)
    ymxw= np.subtract(y,xw)
    ymxwt=np.transpose(ymxw)
    p1=0.5*(np.dot(ymxwt,ymxw))
    p2=0.5*lambd*(np.dot(np.transpose(w),w))
    error = p1+p2       
    wtxt=np.dot(np.transpose(w), np.transpose(X))
        
    wtxtx = np.dot(wtxt, X)
    ytx = np.dot(np.transpose(y),X)
    lw = lambd * w
    
    error_grad_inter = wtxtx - ytx + np.transpose(lw)
    
    error_grad=np.array(error_grad_inter).flatten()
    global count
    count = count + 1                                           
    return error, error_grad

def calcpow(x,p):
    powe =[1]
    for i in range(p+1):
        if i>0:
            xp = pow(x,i)
            powe.append(xp)
    myarr= np.asarray(powe) 
    return myarr
    
def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    # IMPLEMENT THIS METHOD
    
    Z = np.zeros((1,p+1))
    for row in x:
        Y = calcpow(row,p)
        Z=np.vstack((Z,Y))
    Xd=np.delete(Z,0,0)
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('./sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('./sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()


zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
#plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
#plt.title('LDA')

#plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
#plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
#plt.title('QDA')

#plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('./diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('./diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
#w = learnOLERegression(X,y)
#mle = testOLERegression(w,X,y)

#w_i = learnOLERegression(X_i,y)
#mle_i = testOLERegression(w_i,X_i,y)
#print('Train')
#print('MSE without intercept '+str(mle))
#print('MSE with intercept '+str(mle_i))

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
#print('Test')
print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
#fig = plt.figure(figsize=[12,6])
#plt.subplot(1, 2, 1)
#plt.plot(lambdas,mses3_train)
#plt.title('MSE for Train Data')
#plt.subplot(1, 2, 2)
#plt.plot(lambdas,mses3)
#plt.title('MSE for Test Data')

#plt.show()

#print "Train MSE: ",np.min(mses3_train)
#print "Test MSE: ",np.min(mses3)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    #print "Count = ",count," Lambda = ",lambd
    count = 0
    i = i + 1
    
#print "Train MSE: ",np.min(mses4_train)
#print "Test MSE: ",np.min(mses4)
#fig = plt.figure(figsize=[12,6])
#plt.subplot(1, 2, 1)
#plt.plot(lambdas,mses4_train)
#plt.plot(lambdas,mses3_train)
#plt.title('MSE for Train Data')
#plt.legend(['Using scipy.minimize','Direct minimization'])

#plt.subplot(1, 2, 2)
#plt.plot(lambdas,mses4)
#plt.plot(lambdas,mses3)
#plt.title('MSE for Test Data')
#plt.legend(['Using scipy.minimize','Direct minimization'])
#plt.show()


# Problem 5
pmax = 7
lambda_opt = np.argmin(mses3)*0.01 # REPLACE THIS WITH lambda_opt estimated from Problem 3
#print("Optimal Lambda = "+lambda_opt)
#lambda_opt = np.argmin(mses3_train)*0.01 # REPLACE THIS WITH lambda_opt estimated from Problem 3
#print "Optimal Lambda train = ",lambda_opt
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

#fig = plt.figure(figsize=[12,6])
#plt.subplot(1, 2, 1)
#plt.plot(range(pmax),mses5_train)
#plt.title('MSE for Train Data')
#plt.legend(('No Regularization','Regularization'))
#plt.subplot(1, 2, 2)
#plt.plot(range(pmax),mses5)
#plt.title('MSE for Test Data')
#plt.legend(('No Regularization','Regularization'))
#plt.show()
