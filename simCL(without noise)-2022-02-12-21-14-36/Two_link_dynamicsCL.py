import numpy as np
from math import sin
from math import cos
from ConcurrentLearning import concurrentlearning

#Defining a class for the dynamics
class dynamics():
    #constructor to initialize a Dynamica object
    def __init__(self, alpha=0.25*np.ones(2,dtype=np.float32), beta=0.1*np.ones(2, dtype=np.float32), gamma=0.01*np.ones(5,dtype=np.float32), lambdaCL=0.1, YYminDiff=0.1, kCL=0.9):
        
        """
        Initialize the dynamics \n
        Inputs:
        -------
        \t alpha: error gain \n
        \t beta:  filtered error gain \n
        \t gamma: parameter update gain \n
        \t kCL: CL parameter update gain \n
        
        Returns:
        -------
        """
        #defining the gaines
        self.alpha=np.diag(alpha)
        self.beta=np.diag(beta)
        self.Gamma=np.diag(gamma)
        self.kCL=kCL
        
        #defining the rigid body parameters
        self.m=np.array([2.0, 2.0],dtype=np.float32)
        self.l=np.array([0.5, 0.5],dtype=np.float32)
        self.mBounds=np.array([1.0, 3.0], dtype=np.float32)
        self.lBounds=np.array([0.25, 0.75],dtype=np.float32)
        self.g=9.8
        
        #defining the desired trajectory parameters
        self.phidMag=np.array([np.pi/8, np.pi/4],dtype=np.float32)
        self.freq=0.2
        self.a=np.pi/2
        self.b=np.array([np.pi/2, np.pi/4],dtype=(np.float32))
        
        #Initialize state
        self.phi,_,_=self.getDesiredstate(0.0) #seting the inititial angles to the desired angles
        self.phiD=np.zeros(2,dtype=np.float32) #initial angular velocity 
        self.phiDD=np.zeros(2,dtype=(np.float32))   #initial angular acceleration
        
        # concurrent learning
        self.concurrentlearning=concurrentlearning(lambdaCL,YYminDiff)
        self.tau = np.zeros(2,np.float32)
        
        # unknown parameters
        self.theta = self.getTheta(self.m, self.l)
        self.thetaH= self.getTheta(self.mBounds[0]*np.ones(2,dtype=np.float32),self.lBounds[0]*np.ones(2,dtype=np.float32)) #initialize theta estimate to the lowerbounds
        
        
        
    def getDesiredstate(self,t):
        """
       Determines the desired state of the system \n
       Inputs:
       -------
       \t t: time \n
       
       Returns:
       -------
       \t phid:   desired angles \n
       \t phiDd:  desired angular velocity \n
       \t phiDDd: desired angular acceleration
       """
        #To get the desired anglular displacements
        
        phid=np.array([self.phidMag[0]*sin(2*np.pi*self.freq*t - self.a) - self.b[0],
                     self.phidMag[1]*sin(2*np.pi*self.freq*t-self.a)+self.b[1]], dtype=np.float32)
        
        #To get the desired angular velocity
        phiDd=np.array([2*np.pi*self.freq*self.phidMag[0]*cos(2*np.pi*self.freq*t- self.a),
                       2*np.pi*self.freq*self.phidMag[1]*cos(2*np.pi*self.freq*t- self.a)], dtype=np.float32)
        
        #To get the desired angular acceleration
        phiDDd=np.array([-((2*np.pi*self.freq)**2)*self.phidMag[0]*sin(2*np.pi*self.freq*t - self.a),
                        -((2*np.pi*self.freq)**2)*self.phidMag[1]*sin(2*np.pi*self.freq*t - self.a)], dtype=np.float32)
        
        return phid, phiDd, phiDDd
   
    #Returns Theta vector
    def getTheta(self,m,l):
        """
        Inputs:
        -------
        \t m: link masses \n
        \t l: link lengths \n
        
        Returns:
        -------
        \t theta: parameters
        """
    
        theta = np.array([(m[0]+m[1])*l[0]**2+m[1]*l[1]**2,
                          m[1]*l[0]*l[1],
                          m[1]*l[1]**2,
                          (m[0]+m[1])*l[0],
                          m[1]*l[1]],dtype=np.float32)
        return theta
    
    #Returns the inertial matrix
    def getM(self,m,l,phi):
        """
        Determines the inertia matrix \n
        Inputs:
        -------
        \t m:   link masses \n
        \t l:   link lengths \n
        \t phi: angles \n
        
        Returns:
        -------
        \t M: inertia matrix
        """
        m1=m[0]
        m2=m[1]
        c2=cos(phi[1])
        l1=l[0]
        l2=l[1]
        
        M= np.array([[(m1*l1**2) + m2*((l1**2) + 2*l1*l2*c2 +l2**2), (m2*(l1*l2*c2 + l2**2))],
                    [m2*(l1*l2*c2 + l2**2), m2*l2**2]], dtype=np.float32)
        return M
                    
    # Returns the Centripital Coriolis matirx                
    def getC(self,m,l,phi,phiD):
        """
        Determines the centripetal coriolis matrix \n
        Inputs:
        -------
        \t m:    link masses \n
        \t l:    link lengths \n
        \t phi:  angles \n
        \t phiD: angular velocities \n
        
        Returns:
        -------
        \t C: cetripetal coriolis matrix
        """
        m2=m[1]
        s2=sin(phi[1])
        l1=l[0]
        l2=l[1]
        
        C=np.array([-2*m2*l1*l2*s2*phiD[0]*phiD[1] - m2*l1*l2*s2*(phiD[1]**2),
                   m2*l1*l2*s2*phiD[0]**2], dtype=np.float32)
        return C
    
    #Returns the Grativational Matrix    
    def getG(self, m, l, phi):
        """
       Determines the gravity matrix \n
       Inputs:
       -------
       \t m:   link masses \n
       \t l:   link lengths \n
       \t phi: angles \n
       
       Returns:
       -------
       \t G: gravity matrix
       """
        m1=m[0]
        m2=m[1]
        c1=cos(phi[0])
        c12=cos(phi[0]+phi[1])
        l1=l[0]
        l2=l[1]
                        
        G=np.array([((m1+m2)*self.g*l1*c1 + m2*self.g*l2*c12),
                    (m2*self.g*l2*c12)], dtype=np.float32)    
        return G
    
    #Returns the inertial matrix regressor
    def getYM(self, vphi, phi):
        
        """
        Determines the inertia matrix regressor \n
        Inputs:
        -------
        \t vphi: phiDDd+alpha*eD or phiDD \n
        \t phi:  angles \n
        
        Returns:
        -------
        \t YM: inertia matrix regressor
        """
        c2=cos(phi[1])
        
        #YM equation had errors - corrected
        
        YM=np.array([[vphi[0], 2*c2*vphi[0] + c2*vphi[1], vphi[1], 0.0, 0.0],
                      [0.0, c2*vphi[0], vphi[0]+vphi[1], 0.0, 0.0]], dtype=np.float32)
        
        return YM
    
    #Returns the centripital coriolis matrix regressor
    def getYC(self, phi, phiD):
        """
       Determines the centripetal coriolis matrix regressor \n
       Inputs:
       -------
       \t phi:  angles \n
       \t phiD: angular velocity \n
       
       Returns:
       -------
       \t YC: centripetal coriolis matrix regressor
       """
        s2=sin(self.phi[1])
        
        YC= np.array([[ 0.0, -2*s2*phiD[0]*phiD[1]-s2*phiD[1]**2, 0.0, 0.0, 0.0],
                       [0.0, s2*phiD[0]**2, 0.0, 0.0, 0.0]], dtype=np.float32)
        return YC
    
    #Returns the Gravitational matrix regressor
    def getYG(self, phi):
        """
        Determines the gravity matrix regressor \n
        Inputs:
        -------
        \t phi: angles \n
        
        Returns:
        -------
        \t YG: gravity matrix regressor
        """
        c1=cos(phi[0])
        c12=cos(phi[0]+phi[1])
        
        YG=np.array([[ 0.0, 0.0, 0.0, self.g*c1, self.g*c12],
                      [0.0, 0.0, 0.0, 0.0, self.g*c12]], dtype=(np.float32))
        return YG
    
    #Returns the M_dot matrix regressor
    def getYM_dot(self, phi, phiD, r):
        """
        Determines the inertia derivative regressor \n
        Inputs:
        -------
        \t phi:  angles \n
        \t phiD: angular velocoty \n
        \t r:    filtered tracking error \n
        
        Returns:
        -------
        \t YM_dot: inertia matrix derivative regressor
        """
        s2=sin(phi[1])
        
        YM_dot=np.array([[ 0.0, -2*s2*phiD[1]*r[0] - s2*phiD[1]*r[1], 0.0, 0.0, 0.0],
                          [0.0, -s2*phiD[1]*r[0], 0.0, 0.0, 0.0]], dtype=(np.float32))
        return YM_dot
    
    #returns the state
    def getState(self, t):
        """
       Returns the state of the system and parameter estimates \n
       Inputs:
       -------
       \t t: time \n
       
       Returns:
       -------
       \t phi:    angles \n
       \t phiD:   angular velocity \n
       \t phiDD:  angular acceleration \n
       \t thetaH: parameter estimate \n
       \t thetaH: parameter
       """
        return self.phi, self.phiD, self.phiDD, self.thetaH, self.theta
    
    #returns the error states
    def getErrorStates(self, t):
        """
        Returns the errors \n
        Inputs:
        -------
        \t t:  time \n
        
        Returns:
        -------
        \t e:          tracking error \n
        \t eD:         tracking error derivative \n
        \t r:          filtered tracking error \n
        \t thetatilda: parameter estimate error
        """
        
        #gets the desired states
        phid, phiDd,_= self.getDesiredstate(t)
        
        #gets the errors
        e = phid - self.phi
        eD = phiDd - self.phiD
        r = eD + self.alpha@e
        
        #calculate thetatilda
        
        thetatilda=self.theta-self.thetaH
        
        return e, eD, r, thetatilda
    
    #returns the inputs and update law
    def getTauThetaHD(self, t):
        """
        Calculates the input and adaptive update law \n
        Inputs:
        -------
        \t t:  time \n
        
        Returns:
        -------
        \t tau:     control input \n
        \t thetaHD: parameter estimate adaptive update law \n
        \t tauff:   input from the feedforward portion of control \n
        \t taufb:   input from the feedback portion of control \n
        \t thetaCL: approximate of theta from CL \n
        """
        #gets the desired states
        _,_, phiDDd= self.getDesiredstate(t)
        
        #get the error
        e,eD,r,_=self.getErrorStates(t)
        
        #get the regressors
        vphi = phiDDd + self.alpha@eD
        YM = self.getYM(vphi, self.phi)
        YC =self.getYC(self.phi, self.phiD)
        YG =self.getYG(self.phi)
        YM_dot=self.getYM_dot(self.phi, self.phiD, r)
        Y = YM +YC +YG + 0.5*YM_dot
        
        #calculating the contoroller's update law
        taufb=e + self.beta@r
        tauff=Y@self.thetaH
        tau = taufb + tauff
        
        #update the CL stack and the update law
        YYsumMinEig,_,YYsum,YtauSum = self.concurrentlearning.getState()
        thetaCL= np.zeros_like(self.theta,np.float32)
        if YYsumMinEig > 0.001:
            thetaCL = np.linalg.inv(YYsum)@YtauSum
        thetaHD = self.Gamma@Y.T@r + self.kCL*self.Gamma@(YtauSum - YYsum @ self.thetaH)
        return tau, thetaHD, tauff, taufb, thetaCL
    
    def getCLState(self):
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
        
        return self.concurrentlearning.getState()
    
    #steping the system
    def step(self, dt, t):
        """
        Steps the internal state using the dynamics \n
        Inputs:
        -------
        \t dt: time step \n
        \t t:  time \n
        
        Returns:
        -------
        """
        #get the dynamics
        M = self.getM(self.m, self.l, self.phi)
        C =self.getC(self.m, self.l, self.phi, self.phiD)
        G = self.getG(self.m, self.l, self.phi)
        
        #get the input and the update law
        tau, thetaHD,_,_,_ =self.getTauThetaHD(t)
        
        #calculate the dynamics using the input
        self.phiDD =np.linalg.inv(M)@(-C-G+tau)
        
        #update the internal states
        self.phi += dt*self.phiD
        self.phiD += dt*self.phiDD
        self.thetaH += dt*thetaHD
        
        #update the concurrent learning 
        #get the inertia regressor for CL
        
        YMCL=self.getYM(self.phiDD,self.phi)
        YC=self.getYC(self.phi, self.phiD)
        YG=self.getYG(self.phi)
        YCL=YMCL+YC+YG
        self.concurrentlearning.append(YCL,tau,t+dt)
            
            
    
    
                    
    
    
    
                          
        
