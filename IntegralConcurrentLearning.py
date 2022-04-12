import numpy as np
from numpy.linalg import eigvals
from collections import namedtuple

LearningData=namedtuple('LearningData',('Y','tau','t'))
#class for Two link dynamics
class integralconcurrentlearning():
    #constructor to initialize a Dynamics object
    def __init__(self, lambdaCL=0.1,YYminDiff=0.1,deltaT=0.1):
        """
        Initialize the learning \n
        Inputs:
        -------
        \t lambdaCL: minimum eigenvalue for the sum \n
        \t YYminDiff: minimum difference between data to save it to the buffer \n
        
        Returns:
        -------
        """
       
        
        self.Ybuffint=[]  #buffer to calculate the integral
        self.deltaT=deltaT  #initial integration window
        self.YYsum=np.zeros((5,5),dtype=np.float32)
        self.Ytausum=np.zeros(5,dtype=np.float32)
        self.Ybuff=[]
        self.lambdaCL=lambdaCL
        self.lambdaCLmet=False
        self.YYminDiff=YYminDiff
        self.YYsumMinEig=0.0 #current minimum eigenvalue of the sum
        self.TCL=0.0 #initial time minimum eigenvalue is satisfied
        
    
    def append(self,Y,tau,t):
        """
       Adds the new data to the buffer if it is different enough and the minimum eigenvalue is not satisfied
       Inputs:
       -------
       \t Y: regressor \n
       \t tau: torque \n
       
       Returns:
       -------
       """
       #To get Yicl
       

        if not self.lambdaCLmet:
            self.Ybuffint.append(LearningData(Y,tau,t))
            Yint=np.zeros((2,5))
            tauint=0.0
            
            #Integrate if there is enought data
            if len(self.Ybuffint)>2:
                
                #If the data is larger than the data needed to calculate the integral then delete the oldest entry
                deltacurr=self.Ybuffint[-1].t-self.Ybuffint[1].t   #current time window (difference between the last and the 2nd entry excluding the 1st(oldest))
                if deltacurr > self.deltaT:
                    self.Ybuffint.pop(0)
        
            #Calculate Yicl
            for ii in range(len(self.Ybuffint)-1):
                dti=self.Ybuffint[ii+1].t-self.Ybuffint[ii].t
                Yavg= 0.5*(self.Ybuffint[ii+1].Y+self.Ybuffint[ii].Y)
                Tauavg= 0.5*(self.Ybuffint[ii+1].tau+self.Ybuffint[ii].tau)
                Yint += dti*Yavg
                tauint += dti*Tauavg
            
        else:
            return
        
                
            
        #don't add the data if the minimum eigenvlaue is good or the new data has a good minimum singular value
        _,YSD,_=np.linalg.svd(Y)
            
        if (np.min(YSD) > self.YYminDiff) and (np.linalg.norm(tauint) > self.YYminDiff):
            #check if the data is different enough from the other data
            #find the minimum difference
            minDiff=100
            for Yi in self.Ybuff:
                deltaYi=np.linalg.norm(Yi-Yint)
                if deltaYi < minDiff:
                    minDiff = deltaYi
                        
            #If the minimum difference is large enough add the data        
            if minDiff > self.YYminDiff:
                self.Ybuff.append(Yint)
                YY = Yint.T@Yint
                Ytau = Yint.T@tauint
                self.YYsum += YY
                self.Ytausum += Ytau
                self.YYsumMinEig=np.min(eigvals(self.YYsum))
                    
                #Check if the new data makes the eigen value large enough
                if self.YYsumMinEig > self.lambdaCL:
                    self.lambdaCLmet=True
                    self.TCL=t
                    
    def getState(self):
        """
        Returns select parameters CL \n
        Inputs:
        -------
        
        Returns:
        -------
        \t YYsumMinEig: current minimum eigenvalue of sum of the Y^T*Y terms \n
        \t TCL: time of the minimum eigenvalue found \n
        \t YYsum: Y^T*Y sum \n
        \t YtauSum: Y^T*tau sum \n
        """
        return self.YYsumMinEig,self.TCL,self.YYsum,self.Ytausum
                    
                                    
                            
                

