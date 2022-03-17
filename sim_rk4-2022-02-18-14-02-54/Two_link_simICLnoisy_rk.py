import numpy as np
import Two_link_dynamicsICLnoisy_rk
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
    alpha= 2.0*np.ones(2,dtype=np.float32)
    beta= 1.00*np.ones(2,dtype=np.float32)
    gamma= 0.5*np.ones(5,dtype=np.float32)
    lambdaCL=1.0
    YYmindiff=0.5
    kCL=0.2
    
    tauNoise=0.2
    phiNoise=0.1
    phiDNoise=0.15
    phiDDNoise=0.2
    addNoise=True
    
    dyn= Two_link_dynamicsICLnoisy_rk.dynamics(alpha=alpha, beta=beta, gamma=gamma, lambdaCL=lambdaCL, YYminDiff=YYmindiff, kCL=kCL,addNoise=addNoise,tauN=tauNoise,phiN=phiNoise,phiDN=phiDNoise,phiDDN=phiDDNoise)
    
    phiHist = np.zeros((4,len(t)),dtype=np.float32)
    phiDHist= np.zeros((4,len(t)),dtype=np.float32)
    phiDDHist= np.zeros((4,len(t)),dtype=np.float32)
    
    phidHist= np.zeros((2,len(t)),dtype=np.float32)
    phiDdHist= np.zeros((2,len(t)),dtype=np.float32)
    phiDDdHist= np.zeros((2,len(t)),dtype=np.float32)
    
    eHist= np.zeros((4,len(t)),dtype=np.float32)
    eNormHist= np.zeros((2,len(t)),dtype=np.float32)
    
    rHist=np.zeros((4,len(t)),dtype=np.float32)
    rNormHist= np.zeros((2,len(t)),dtype=np.float32)
    
    thetaHist=np.zeros((5,len(t)),dtype=np.float32)
    thetaHHist=np.zeros((10,len(t)),dtype=np.float32)
    thetaCLHist=np.zeros((10,len(t)),dtype=np.float32)
    thetaTildaHist=np.zeros((10,len(t)),dtype=np.float32)
    thetaTildaNormHist=np.zeros((2,len(t)),dtype=np.float32)
    thetaClMinHist=np.zeros((2,len(t)),dtype=np.float32)
    
    tauHist=np.zeros((4,len(t)),dtype=np.float32)
    tauffHist=np.zeros((4,len(t)),dtype=np.float32)
    taufbHist=np.zeros((4,len(t)),dtype=np.float32)
    lambdaCLMinHist=np.zeros((2,len(t)),dtype=np.float32)
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
        phij, phimj, phiDj, phiDmj, phiDDj, phiDDmj, thetaHj, thetaHmj, thetaj = dyn.getState(t[jj])
        phidj, phiDdj, phiDDdj = dyn.getDesiredstate(t[jj])
        ej,emj,_,_,rj,rmj,thetaTildaj,thetaTildamj = dyn.getErrorStates(t[jj])
        tauj,taumj,_,_,tauffj,tauffmj, taufbj, taufbmj, thetaCLj, thetaCLmj = dyn.getTauThetaHD(t[jj])
        YsumMinEigj,YsumMInEigmj,TCLj,TCLmj,_,_,_,_=dyn.getCLState()
        
        if not TCLfound:
            if TCLj > 0:
                TCL = TCLj
                TCLindex = jj
                TCLfound = True
        
        if not TCLmfound:
            if TCLmj > 0:
                TCL = TCLmj
                TCLmindex=jj
                TCLmfound = True
                
        
        #save the data to the buffers
        phiHist[:,jj]=[phij[0],phij[1],phimj[0],phimj[1]]
        phiDHist[:,jj]=[phiDj[0],phiDj[1],phiDmj[0],phiDmj[1]]
        phiDDHist[:,jj]=[phiDDj[0],phiDDj[1],phiDDmj[0],phiDDmj[1]]
        phidHist[:,jj]= phidj
        phiDdHist[:,jj]=phiDdj
        phiDDdHist[:,jj]=phiDDdj
        eHist[:,jj]=[ej[0],ej[1],emj[0],emj[1]]
        eNormHist[:,jj]=[np.linalg.norm(ej),np.linalg.norm(emj)]
        rHist[:,jj]=[rj[0],rj[1],rmj[0],rmj[1]]
        rNormHist[:,jj]=[np.linalg.norm(rj),np.linalg.norm(rmj)]
        thetaHist[:,jj]=thetaj
        thetaHHist[:,jj]=[thetaHj[0],thetaHj[1],thetaHj[2],thetaHj[3],thetaHj[4],thetaHmj[0],thetaHmj[1],thetaHmj[2],thetaHmj[3],thetaHmj[4]]
        thetaCLHist[:,jj]=[thetaCLj[0],thetaCLj[1],thetaCLj[2],thetaCLj[3],thetaCLj[4],thetaCLmj[0],thetaCLmj[1],thetaCLmj[2],thetaCLmj[3],thetaCLmj[4]]
        thetaTildaHist[:,jj]=[thetaTildaj[0],thetaTildaj[1],thetaTildaj[2],thetaTildaj[3],thetaTildaj[4],thetaTildamj[0],thetaTildamj[1],thetaTildamj[2],thetaTildamj[3],thetaTildamj[4]]
        thetaTildaNormHist[:,jj]=[np.linalg.norm(thetaTildaj),np.linalg.norm(thetaTildamj)]
        lambdaCLMinHist[:,jj]=[YsumMinEigj,YsumMInEigmj]
        tauHist[:,jj]=[tauj[0],tauj[1],taumj[0],taumj[1]]
        taufbHist[:,jj]=[taufbj[0],taufbj[1],taufbmj[0],taufbmj[1]]
        tauffHist[:,jj]=[tauffj[0],tauffj[1],tauffmj[0],tauffmj[1]]
        
        #save the internal data to file
        file = open(path+"/data.csv","a",newline='')
        #writing the data into the file
        with file: 
            write = csv.writer(file)
            write.writerow([t[jj],eHist[0,jj],eHist[1,jj],eHist[2,jj],eHist[3,jj],rHist[0,jj],rHist[1,jj],rHist[2,jj],rHist[3,jj],tauHist[0,jj],tauHist[1,jj],tauHist[2,jj],tauHist[3,jj]])
            file.close()
        
        #step the internal state of the dynamics
        dyn.step(dt,t[jj])
    
    #plot the data
    #plot the angles
    phiplot,phiax = plot.subplots()
    phiax.plot(t,phidHist[0,:],color='orange',linewidth=2,linestyle='--')
    phiax.plot(t,phiHist[0,:],color='orange',linewidth=2,linestyle='-')
    phiax.plot(t,phidHist[1,:],color='blue',linewidth=2,linestyle='--')
    phiax.plot(t,phiHist[1,:],color='blue',linewidth=2,linestyle='-')
    phiax.plot(t,phiHist[2,:],color='green',linewidth=2,linestyle='-')
    phiax.plot(t,phiHist[3,:],color='red',linewidth=2,linestyle='-')
    
    phiax.set_xlabel("$t$ $(sec)$")
    phiax.set_ylabel("$\phi_i$ $(rad)$")
    phiax.set_title("Angle")
    phiax.legend(["$\phi_{1d}$","$\phi_1$","$\phi_{2d}$","$\phi_2$","$\phi_m1$","$\phi_m2$"],loc='upper right')
    phiax.grid()
    phiplot.savefig(path+"/angles.pdf")
    
    #plot the error
    eplot,eax = plot.subplots()
    eax.plot(t,eHist[0,:],color='orange',linewidth=2,linestyle='-')
    eax.plot(t,eHist[1,:],color='orange',linewidth=2,linestyle='-')
    eax.plot(t,eHist[2,:],color='blue',linewidth=2,linestyle='-')
    eax.plot(t,eHist[3,:],color='blue',linewidth=2,linestyle='-')
    eax.set_xlabel("$t$ $(sec)$")
    eax.set_ylabel("$e_i$ $(rad)$")
    eax.set_title("Error")
    eax.legend(["$e_1$","$e_2$","$e_m1$","$e_m2$"],loc='upper right')
    eax.grid()
    eplot.savefig(path+"/error.pdf")

    #plot the error norm
    eNplot,eNax = plot.subplots()
    eNax.plot(t,eNormHist[0,:],color='orange',linewidth=2,linestyle='-')
    eNax.plot(t,eNormHist[1,:],color='blue',linewidth=2,linestyle='-')
    eNax.set_xlabel("$t$ $(sec)$")
    eNax.set_ylabel("$\Vert e \Vert$ $(rad)$")
    eNax.set_title("Error Norm")
    eNax.legend(["$e_norm$","$e_mnorm$"],loc='upper right')
    eNax.grid()
    eNplot.savefig(path+"/errorNorm.pdf")

    #plot the anglular velocity
    phiDplot,phiDax = plot.subplots()
    phiDax.plot(t,phiDdHist[0,:],color='orange',linewidth=2,linestyle='--')
    phiDax.plot(t,phiDHist[0,:],color='orange',linewidth=2,linestyle='-')
    phiDax.plot(t,phiDdHist[1,:],color='blue',linewidth=2,linestyle='--')
    phiDax.plot(t,phiDHist[1,:],color='blue',linewidth=2,linestyle='-')
    phiDax.plot(t,phiDHist[2,:],color='orange',linewidth=2,linestyle='-')
    phiDax.plot(t,phiDHist[3,:],color='blue',linewidth=2,linestyle='-')
    phiDax.set_xlabel("$t$ $(sec)$")
    phiDax.set_ylabel("$anglular velocity$ $(rad/sec)$")
    phiDax.set_title("Anglular Velocity")
    phiDax.legend(["$\dot{\phi}_{1d}$","$\dot{\phi}_1$","$\dot{\phi}_{2d}$","$\dot{\phi}_2$","$\dot{\phi}_m1$","$\dot{\phi}_m2$"],loc='upper right')
    phiDax.grid()
    phiDplot.savefig(path+"/anglularVelocity.pdf")

    #plot the filtered error
    rplot,rax = plot.subplots()
    rax.plot(t,rHist[0,:],color='orange',linewidth=2,linestyle='-')
    rax.plot(t,rHist[1,:],color='blue',linewidth=2,linestyle='-')
    rax.plot(t,rHist[2,:],color='orange',linewidth=2,linestyle='-')
    rax.plot(t,rHist[3,:],color='blue',linewidth=2,linestyle='-')
    rax.set_xlabel("$t$ $(sec)$")
    rax.set_ylabel("$r_i$ $(rad/sec)$")
    rax.set_title("Filtered Error")
    rax.legend(["$r_1$","$r_2$","$r_m1$","$r_m2$"],loc='upper right')
    rax.grid()
    rplot.savefig(path+"/filteredError.pdf")

    #plot the filtered error norm
    rNplot,rNax = plot.subplots()
    rNax.plot(t,rNormHist[0,:],color='orange',linewidth=2,linestyle='-')
    rNax.plot(t,rNormHist[1,:],color='blue',linewidth=2,linestyle='-')
    rNax.set_xlabel("$t$ $(sec)$")
    rNax.set_ylabel("$\Vert r \Vert$ $(rad)$")
    rNax.set_title("Filtered Error Norm")
    rNax.legend(["$r_norm$","$r_mnorm$"])
    rNax.grid()
    rNplot.savefig(path+"/filteredErrorNorm.pdf")

    #plot the anglular acceleration
    phiDDplot,phiDDax = plot.subplots()
    phiDDax.plot(t,phiDDdHist[0,:],color='orange',linewidth=2,linestyle='--')
    phiDDax.plot(t,phiDDHist[0,:],color='orange',linewidth=2,linestyle='-')
    phiDDax.plot(t,phiDDdHist[1,:],color='blue',linewidth=2,linestyle='--')
    phiDDax.plot(t,phiDDHist[1,:],color='blue',linewidth=2,linestyle='-')
    phiDDax.plot(t,phiDDHist[2,:],color='orange',linewidth=2,linestyle='-')
    phiDDax.plot(t,phiDDHist[3,:],color='blue',linewidth=2,linestyle='-')
    phiDDax.set_xlabel("$t$ $(sec)$")
    phiDDax.set_ylabel("$anglular acceleration$ $(rad/sec^2)$")
    phiDDax.set_title("Anglular Acceleration")
    phiDDax.legend(["$\ddot{\phi}_{1d}$","$\ddot{\phi}_1$","$\ddot{\phi}_{2d}$","$\ddot{\phi}_2$","$ddot{\phi}_m1$","$ddot{\phi}_m2$"],loc='upper right')
    phiDDax.grid()
    phiDDplot.savefig(path+"/anglularAcceleration.pdf")

    #plot the inputs
    tauplot,tauax = plot.subplots()
    tauax.plot(t,tauHist[0,:],color='orange',linewidth=2,linestyle='-')
    tauax.plot(t,tauHist[1,:],color='blue',linewidth=2,linestyle='-')
    tauax.plot(t,tauHist[2,:],color='orange',linewidth=2,linestyle='-')
    tauax.plot(t,tauHist[3,:],color='blue',linewidth=2,linestyle='-')
    
    tauax.plot(t,tauffHist[0,:],color='orange',linewidth=2,linestyle='--')
    tauax.plot(t,tauffHist[1,:],color='blue',linewidth=2,linestyle='--')
    tauax.plot(t,tauffHist[2,:],color='orange',linewidth=2,linestyle='--')
    tauax.plot(t,tauffHist[3,:],color='blue',linewidth=2,linestyle='--')
    
    tauax.plot(t,taufbHist[0,:],color='orange',linewidth=2,linestyle='-.')
    tauax.plot(t,taufbHist[1,:],color='blue',linewidth=2,linestyle='-.')
    tauax.plot(t,taufbHist[2,:],color='orange',linewidth=2,linestyle='-.')
    tauax.plot(t,taufbHist[3,:],color='blue',linewidth=2,linestyle='-.')
    
    tauax.set_xlabel("$t$ $(sec)$")
    tauax.set_ylabel("$input$ $(Nm)$")
    tauax.set_title("Control Input")
    tauax.legend(['$\\tau_1$',"$\\tau_2$","$\\tau_m1$","\\tau_m2$" ,"$\\tau_{ff1}$","$\\tau_{ff2}$","$\\tau_{mff1}$","$\\tau_{mff2}$","$\\tau_{fb1}$","$\\tau_{fb2}$","$\\tau_{mfb1}$","$\\tau_{mfb2}$"],loc='upper right')
    tauax.grid()
    tauplot.savefig(path+"/input.pdf")

 #plot the parameter estiamtes
    thetaHplot,thetaHax = plot.subplots()
    thetaHax.plot(t,thetaHist[0,:],color='red',linewidth=2,linestyle='--')
    thetaHax.plot(t,thetaHist[1,:],color='green',linewidth=2,linestyle='--')
    thetaHax.plot(t,thetaHist[2,:],color='blue',linewidth=2,linestyle='--')
    thetaHax.plot(t,thetaHist[3,:],color='orange',linewidth=2,linestyle='--')
    thetaHax.plot(t,thetaHist[4,:],color='magenta',linewidth=2,linestyle='--')
    
    thetaHax.plot(t,thetaHHist[0,:],color='red',linewidth=2,linestyle='-')
    thetaHax.plot(t,thetaHHist[1,:],color='green',linewidth=2,linestyle='-')
    thetaHax.plot(t,thetaHHist[2,:],color='blue',linewidth=2,linestyle='-')
    thetaHax.plot(t,thetaHHist[3,:],color='orange',linewidth=2,linestyle='-')
    thetaHax.plot(t,thetaHHist[4,:],color='magenta',linewidth=2,linestyle='-')
    thetaHax.plot(t,thetaHHist[5,:],color='red',linewidth=2,linestyle='-')
    thetaHax.plot(t,thetaHHist[6,:],color='green',linewidth=2,linestyle='-')
    thetaHax.plot(t,thetaHHist[7,:],color='blue',linewidth=2,linestyle='-')
    thetaHax.plot(t,thetaHHist[8,:],color='orange',linewidth=2,linestyle='-')
    thetaHax.plot(t,thetaHHist[9,:],color='magenta',linewidth=2,linestyle='-')
    
    thetaHax.plot(t,thetaCLHist[0,:],color='red',linewidth=2,linestyle='-.')
    thetaHax.plot(t,thetaCLHist[1,:],color='green',linewidth=2,linestyle='-.')
    thetaHax.plot(t,thetaCLHist[2,:],color='blue',linewidth=2,linestyle='-.')
    thetaHax.plot(t,thetaCLHist[3,:],color='orange',linewidth=2,linestyle='-.')
    thetaHax.plot(t,thetaCLHist[4,:],color='magenta',linewidth=2,linestyle='-.')
    thetaHax.plot(t,thetaCLHist[5,:],color='red',linewidth=2,linestyle='-.')
    thetaHax.plot(t,thetaCLHist[6,:],color='green',linewidth=2,linestyle='-.')
    thetaHax.plot(t,thetaCLHist[7,:],color='blue',linewidth=2,linestyle='-.')
    thetaHax.plot(t,thetaCLHist[8,:],color='orange',linewidth=2,linestyle='-.')
    thetaHax.plot(t,thetaCLHist[9,:],color='magenta',linewidth=2,linestyle='-.')
    
    thetaHax.set_xlabel("$t$ $(sec)$")
    thetaHax.set_ylabel("$\\theta_i$")
    thetaHax.set_title("Parameter Estimates")
    thetaHax.legend(["$\\theta_1$","$\\theta_2$","$\\theta_3$","$\\theta_4$","$\\theta_5$","$\hat{\\theta}_1$","$\hat{\\theta}_2$","$\hat{\\theta}_3$","$\hat{\\theta}_4$","$\hat{\\theta}_5$","$\hat{\\theta}_m1$","$\hat{\\theta}_m2$","$\hat{\\theta}_m3$","$\hat{\\theta}_m4$","$\hat{\\theta}_m5$" ,"$\hat{\\theta}_{CL1}$","$\hat{\\theta}_{CL2}$","$\hat{\\theta}_{CL3}$","$\hat{\\theta}_{CL4}$","$\hat{\\theta}_{CL5}$","$\hat{\\theta}_{mCL1}$","$\hat{\\theta}_{mCL2}$","$\hat{\\theta}_{mCL3}$","$\hat{\\theta}_{mCL4}$","$\hat{\\theta}_{mCL5}$"],loc='lower right',bbox_to_anchor=(1.05, -0.15),ncol=3)
    thetaHax.grid()
    thetaHplot.savefig(path+"/thetaHat.pdf")
    
    #plot the parameter estiamtes
    thetaplot,thetaax = plot.subplots()
    thetaax.plot(t,thetaTildaHist[0,:],color='red',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildaHist[1,:],color='green',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildaHist[2,:],color='blue',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildaHist[3,:],color='orange',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildaHist[4,:],color='magenta',linewidth=2,linestyle='-')
    
    thetaax.plot(t,thetaTildaHist[5,:],color='red',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildaHist[6,:],color='green',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildaHist[7,:],color='blue',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildaHist[8,:],color='orange',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTildaHist[9,:],color='magenta',linewidth=2,linestyle='-')
    
    thetaax.set_xlabel("$t$ $(sec)$")
    thetaax.set_ylabel("$\\tilde{\\theta}_i$")
    thetaax.set_title("Parameter Error")
    thetaax.legend(["$\\tilde{\\theta}_1$","$\\tilde{\\theta}_2$","$\\tilde{\\theta}_3$","$\\tilde{\\theta}_4$","$\\tilde{\\theta}_5$","$\\tilde{\\theta}_m1$","$\\tilde{\\theta}_m2$","$\\tilde{\\theta}_m3$","$\\tilde{\\theta}_m4$","$\\tilde{\\theta}_m5$"],loc='upper right')
    thetaax.grid()
    thetaplot.savefig(path+"/thetaTilde.pdf")

    #plot the parameter estiamtes norm
    thetaNplot,thetaNax = plot.subplots()
    thetaNax.plot(t,thetaTildaNormHist[0,:],color='orange',linewidth=2,linestyle='-')
    thetaNax.plot(t,thetaTildaNormHist[1,:],color='blue',linewidth=2,linestyle='-')
    thetaNax.set_xlabel("$t$ $(sec)$")
    thetaNax.set_ylabel("$\Vert \\tilde{\\theta} \Vert$")
    thetaNax.set_title("Parameter Error Norm")
    thetaNax.legend("$\\tilde{\\theta}_norm$","$\\tilde{\\theta}_mnorm$",loc='upper right')
    thetaNax.grid()
    thetaNplot.savefig(path+"/thetaTildeNorm.pdf")
    
    #plot the minimum eigenvalue
    eigplot,eigax = plot.subplots()
    eigax.plot(t,lambdaCLMinHist[0,:],color='orange',linewidth=2,linestyle='-')
    eigax.plot(t,lambdaCLMinHist[1,:],color='orange',linewidth=2,linestyle='-')
    eigax.plot([TCL,TCL],[0.0,lambdaCLMinHist[0,TCLindex]],color='black',linewidth=1,linestyle='-')
    eigax.plot([TCLm,TCLm],[0.0,lambdaCLMinHist[1,TCLindex]],color='black',linewidth=1,linestyle='-')
    eigax.set_xlabel("$t$ $(sec)$")
    eigax.set_ylabel("$\lambda_{min}$")
    eigax.set_title("Minimum Eigenvalue $T_{CL}$="+str(round(TCL,2)))
    
    eigax.grid()
    eigplot.savefig(path+"/minEig.pdf")
    TCL = TCLj
                   

    

