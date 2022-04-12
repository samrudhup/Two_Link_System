
import numpy as np
import two_link_noisy_single_layer_nn_dynamics
import csv
import os
import datetime
import matplotlib
import matplotlib.pyplot as plot
from matplotlib import rc


if __name__=='__main__':
    dt=0.005 #time interval
    tf=60 #final time
    t=np.linspace(0.0, tf, int(tf/dt), dtype=(np.float32))
    
    alpha = 3.0*np.identity(2)
    betar = 1.5*np.identity(2)
    betaeps = 0.001*np.identity(2)
    gammath = 0.75
    gammaw = 0.75
    lambdaCL = 0.0001
    
    YYmindiff=0.5
    deltaT = 1.5
    kCL=0.2
    L = 10
    Lmod = 4*L +1
    
    tauNoise=1.0
    phiNoise=0.04
    phiDNoise=0.05
    phiDDNoise=0.06
    useCL=True
    useNN=True
    useYth=True
    
   
    dyn= two_link_noisy_single_layer_nn_dynamics.dynamics(alpha=alpha,betar=betar,betaeps=betaeps,gammath=gammath,gammaw=gammaw,lambdaCL=lambdaCL,YYminDiff=YYmindiff,kCL=kCL,tauN=tauNoise,phiN=phiNoise,phiDN=phiDNoise,phiDDN=phiDDNoise,L=L,deltaT=deltaT,useCL=useCL,useNN=useNN,useYth=useYth)
    
    phiHist = np.zeros((2,len(t)),dtype=np.float32)
    phiDHist= np.zeros((2,len(t)),dtype=np.float32)
    phiDDHist= np.zeros((2,len(t)),dtype=np.float32)
    
    phidHist= np.zeros((2,len(t)),dtype=np.float32)
    phiDdHist= np.zeros((2,len(t)),dtype=np.float32)
    phiDDdHist= np.zeros((2,len(t)),dtype=np.float32)
    
    eHist= np.zeros((2,len(t)),dtype=np.float32)
    emHist = np.zeros((2,len(t)),dtype=np.float32)
    eNormHist= np.zeros(len(t),dtype=np.float32)
    
    rHist=np.zeros(len(t),dtype=np.float32)
    rNormHist= np.zeros(len(t),dtype=np.float32)
    
    thetaHist=np.zeros((5,len(t)),dtype=np.float32)
    
    
    WHHist = np.zeros((2*Lmod,len(t)),dtype=np.float64)
    fHist = np.zeros((2,len(t)),dtype=np.float64)
    fHHist = np.zeros((2,len(t)),dtype=np.float64)
    fDiffNormHist = np.zeros(len(t),dtype=np.float64)
    
    tauHist=np.zeros((2,len(t)),dtype=np.float32)
    tauffHist=np.zeros((2,len(t)),dtype=np.float32)
    taufbHist=np.zeros((2,len(t)),dtype=np.float32)
    lambdaCLMinHist=np.zeros(len(t),dtype=np.float32)
    
    TCL = 0
    TCLindex = 0
    TCLfound = False
    TCLm = 0
    TCLmindex = 0
    TCLmfound = False
    
    #start save file
    savePath="/Users/sam/Documents/Spring 2022"
    now=datetime.datetime.now()
    nownew = now.strftime("%Y-%m-%d-%H-%M-%S")
    path=savePath+"/sim-"+nownew
    os.mkdir(path)
    file=open(path+"/data.csv","w",newline='')
    
    #writting the header into the file
    with file: 
        write = csv.writer(file)
        write.writerow(["time","e1","e2","em1","em2","r1","r2","rm1","rm2","tau1","tau2","taum1","taum2"])
    file.close()
    
    #loop through
    for jj in range(0, len(t)):
        #get the states and input data
        phij, phimj, phiDj, phiDmj, phiDDj, phiDDmj, thetaHj, WHj = dyn.getState(t[jj])
        phidj, phiDdj, phiDDdj = dyn.getDesiredstate(t[jj])
        ej,emj,_,_,rj,rmj = dyn.getErrorStates(t[jj])
        tauj,tauffj, taufbj,_,_,_ = dyn.getTau(t[jj],phi=phimj,phiD=phiDmj,thetaH=thetaHj,WH=WHj) 
        lambdaCLMinj,TCLj,_,_ = dyn.getCLState()
        fj,fHj = dyn.getfuncComp(phi=phimj,phiD=phiDmj,phiDD=phiDDmj,tau=tauj,thetaH=thetaHj,WH=WHj)
        fDiffNormj = np.linalg.norm(fj-fHj)
        
        if not TCLfound:
            if TCLj > 0:
                TCL = TCLj
                TCLindex = jj
                TCLfound = True
    
        
        #save the data to the buffers
        phiHist[:,jj]=phimj
        phiDHist[:,jj]=phiDj
        phiDDHist[:,jj]=phiDDj
        phidHist[:,jj]= phidj
        phiDdHist[:,jj]=phiDdj
        phiDDdHist[:,jj]=phiDDdj
        eHist[:,jj]=ej
        emHist[:,jj]=emj
        eNormHist[jj]=np.linalg.norm(emj)
        rHist[jj]=np.linalg.norm(rmj)
        
        
        thetaHist[:,jj]=thetaHj
        lambdaCLMinHist[jj]=lambdaCLMinj
        tauHist[:,jj]=tauj
        taufbHist[:,jj]=taufbj
        tauffHist[:,jj]=tauffj
        fHist[:,jj] = fj
        fHHist[:,jj] = fHj
        fDiffNormHist[jj]= fDiffNormj
        WHHist[:,jj]= np.reshape(WHj.T,(2*Lmod))
        if np.linalg.norm(phimj) > 5.0*np.linalg.norm(phidj) or np.linalg.norm(tauj) > 1000:
            print("GOING UNSTABLE")
            break
        
        #save the internal data to file
        file = open(path+"/data.csv","a",newline='')
        #writing the data into the file
        with file: 
            write = csv.writer(file)
            write.writerow([t[jj],eHist[:,jj],rHist[jj],tauHist[:,jj]])
        file.close()
        
        #step the internal state of the dynamics
        dyn.step(dt,t[jj])
    
    #plot the data
    #plot the angles
    phiplot,phiax = plot.subplots()
    phiax.plot(t,phidHist[0,:],color='orange',linewidth=2,linestyle='-')
    phiax.plot(t,phiHist[0,:],color='orange',linewidth=2,linestyle='--')
    phiax.plot(t,phidHist[1,:],color='blue',linewidth=2,linestyle='-')
    phiax.plot(t,phiHist[1,:],color='blue',linewidth=2,linestyle='--')
    phiax.set_xlabel("$t$ $(sec)$")
    phiax.set_ylabel("$\phi$")
    phiax.set_title("Angles")
    phiax.legend(["$\phi_{d1}$","$\phi_1$","$\phi_{d2}$","$\phi_2$"],loc='upper right')
    phiax.grid()
    phiplot.savefig(path+"/angle.pdf")
    
    #plot the error norm
    eNplot,eNax = plot.subplots()
    eNax.plot(t,eNormHist,color='orange',linewidth=2,linestyle='-')
    eNax.set_xlabel("$t$ $(sec)$")
    eNax.set_ylabel("$\Vert e \Vert$")
    eNax.set_title("Error Norm RMS = "+str(np.around(np.sqrt(np.mean(eNormHist**2)),decimals=2)))
    eNax.grid()
    eNplot.savefig(path+"/errorNorm.pdf")
    
    #plot the filter tracking error norm
    eNplot,eNax = plot.subplots()
    eNax.plot(t,rHist,color='orange',linewidth=2,linestyle='-')
    eNax.set_xlabel("$t$ $(sec)$")
    eNax.set_ylabel("$\Vert e \Vert$")
    eNax.set_title("Filter Tracking Error Norm RMS = "+str(np.around(np.sqrt(np.mean(rHist**2)),decimals=2)))
    eNax.grid()
    eNplot.savefig(path+"/filtertrackingerrorNorm.pdf")
    
    #plot the inputs
    uplot,uax = plot.subplots()
    uax.plot(t,tauHist[0,:],color='red',linewidth=2,linestyle='-')
    uax.plot(t,tauffHist[0,:],color='green',linewidth=2,linestyle='-')
    uax.plot(t,taufbHist[0,:],color='blue',linewidth=2,linestyle='-')
    uax.plot(t,tauHist[1,:],color='red',linewidth=2,linestyle='--')
    uax.plot(t,tauffHist[1,:],color='green',linewidth=2,linestyle='--')
    uax.plot(t,taufbHist[1,:],color='blue',linewidth=2,linestyle='--')
    uax.set_xlabel("$t$ $(sec)$")
    uax.set_ylabel("$input$")
    uax.set_title("Control Input")
    uax.legend(["$\\tau_1$","$\\tau_{ff1}$","$\\tau_{fb1}$","$\\tau_2$","$\\tau_{ff2}$","$\\tau_{fb2}$"],loc='upper right')
    uax.grid()
    uplot.savefig(path+"/input.pdf")

    #plot the parameter estiamtes
    thHplot,thHax = plot.subplots()
    for ii in range(5):
        thHax.plot(t,thetaHist[ii,:],color=np.random.rand(3),linewidth=2,linestyle='--')
    thHax.set_xlabel("$t$ $(sec)$")
    thHax.set_ylabel("$\theta_"+str(ii)+"$")
    thHax.set_title("Structured Parameter Estimates")
    thHax.grid()
    thHplot.savefig(path+"/thetaHat.pdf")

    #plot the parameter estiamtes
    WHplot,WHax = plot.subplots()
    for ii in range(Lmod):
        WHax.plot(t,WHHist[ii,:],color=np.random.rand(3),linewidth=2,linestyle='--')
    WHax.set_xlabel("$t$ $(sec)$")
    WHax.set_ylabel("$W_i$")
    WHax.set_title("Unstructured Parameter Estimates")
    WHax.grid()
    WHplot.savefig(path+"/WHat.pdf")

    #plot the approx
    fplot,fax = plot.subplots()
    fax.plot(t,fHist[0,:],color='orange',linewidth=2,linestyle='-')
    fax.plot(t,fHHist[0,:],color='orange',linewidth=2,linestyle='--')
    fax.plot(t,fHist[1,:],color='blue',linewidth=2,linestyle='-')
    fax.plot(t,fHHist[1,:],color='blue',linewidth=2,linestyle='--')
    fax.set_xlabel("$t$ $(sec)$")
    fax.set_ylabel("$function$")
    fax.set_title("Function Approximate")
    fax.legend(["$f_1$","$\hat{f1}$","$f_2$","$\hat{f}_2$"],loc='upper right')
    fax.grid()
    fplot.savefig(path+"/fapprox.pdf")

    #plot the approx norm
    fdplot,fdax = plot.subplots()
    fdax.plot(t,fDiffNormHist,color='orange',linewidth=2,linestyle='-')
    fdax.set_xlabel("$t$ $(sec)$")
    fdax.set_ylabel("$function$")
    fdax.set_title("Function Difference Norm RMS = "+str(np.around(np.sqrt(np.mean(fDiffNormHist**2)),decimals=2)))
    fdax.grid()
    fdplot.savefig(path+"/fdiffnorm.pdf")

    #plot the approx norm
    fdplot,fdax = plot.subplots()
    fdax.plot(t[TCLindex:],fDiffNormHist[TCLindex:],color='orange',linewidth=2,linestyle='-')
    fdax.set_xlabel("$t$ $(sec)$")
    fdax.set_ylabel("$function$")
    fdax.set_title("Function Difference Norm After Learn")
    fdax.grid()
    fdplot.savefig(path+"/fdiffnormafter.pdf")

     
    #plot the minimum eigenvalue
    eigplot,eigax = plot.subplots()
    eigax.plot(t,lambdaCLMinHist,color='orange',linewidth=2,linestyle='-')
    eigax.plot([TCL,TCL],[0.0,lambdaCLMinHist[TCLindex]],color='black',linewidth=1,linestyle='-')
    eigax.set_xlabel("$t$ $(sec)$")
    eigax.set_ylabel("$\lambda_{min}$")
    eigax.set_title("Minimum Eigenvalue $T_{CL}$="+str(round(TCL,2)))
    eigax.grid()
    eigplot.savefig(path+"/minEig.pdf")
     

    tsigmas = np.linspace(0.0,5,int(5/dt),dtype=np.float64)
    phidsigmas = np.zeros((2,len(tsigmas)),dtype=np.float64)
    phiDdsigmas = np.zeros((2,len(tsigmas)),dtype=np.float64)
    sigmaHist = np.zeros((Lmod,len(tsigmas)),dtype=np.float64)
    for jj in range(0,len(tsigmas)):
        # get the state and input data
        phidj,phiDdj,_ = dyn.getDesiredstate(tsigmas[jj])
        sigmaj = dyn.getsigma(phidj,phiDdj)
        sigmaHist[:,jj] = sigmaj
        phidsigmas[:,jj] = phidj
        phiDdsigmas[:,jj] = phiDdj

    #plot the sigmas estiamtes
    sigmaplot,sigmaax = plot.subplots()
    for ii in range(L):
        sigmaax.plot(phidsigmas[0,:],sigmaHist[4*ii,:],color=np.random.rand(3),linewidth=2,linestyle='-')
    # sigmaax.plot(tsigmas,phidsigmas[0,:],color='orange',linewidth=2,linestyle='-')
    sigmaax.set_xlabel("$position$")
    sigmaax.set_ylabel("$sigma$")
    sigmaax.set_title("Sigmas Position1")
    sigmaax.grid()
    sigmaplot.savefig(path+"/sigmasPosition1.pdf")

    #plot the sigmas estiamtes
    sigmaplot,sigmaax = plot.subplots()
    for ii in range(L):
        sigmaax.plot(phidsigmas[1,:],sigmaHist[4*ii+1,:],color=np.random.rand(3),linewidth=2,linestyle='-')
    # sigmaax.plot(tsigmas,phidsigmas[1,:],color='orange',linewidth=2,linestyle='-')
    sigmaax.set_xlabel("$position$")
    sigmaax.set_ylabel("$sigma$")
    sigmaax.set_title("Sigmas Position2")
    sigmaax.grid()
    sigmaplot.savefig(path+"/sigmasPosition2.pdf")

    #plot the sigmas estiamtes
    sigmaplot,sigmaax = plot.subplots()
    for ii in range(L):
        sigmaax.plot(phiDdsigmas[0,:],sigmaHist[4*ii+2,:],color=np.random.rand(3),linewidth=2,linestyle='-')
    # sigmaax.plot(tsigmas,phiDdsigmas[0,:],color='orange',linewidth=2,linestyle='-')
    sigmaax.set_xlabel("$velocity$")
    sigmaax.set_ylabel("$sigma$")
    sigmaax.set_title("Sigmas Velocity1")
    sigmaax.grid()
    sigmaplot.savefig(path+"/sigmasvelocity1.pdf")

    #plot the sigmas estiamtes
    sigmaplot,sigmaax = plot.subplots()
    for ii in range(L):
        sigmaax.plot(phiDdsigmas[1,:],sigmaHist[4*ii+3,:],color=np.random.rand(3),linewidth=2,linestyle='-')
    # sigmaax.plot(tsigmas,phiDdsigmas[1,:],color='orange',linewidth=2,linestyle='-')
    sigmaax.set_xlabel("$velocity$")
    sigmaax.set_ylabel("$sigma$")
    sigmaax.set_title("Sigmas Velocity2")
    sigmaax.grid()
    sigmaplot.savefig(path+"/sigmasvelocity2.pdf")