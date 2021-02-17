# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12, 2020


author: Burkhard Heer

book: Dynamic General Equilibrium Modeling: Computational
        Methods and Applications, with Alfred Maussner

Chapter 8.2., (simplified) Krusell-Smith Algorith

Algorithm 8.2.1 in Heer/Maussner (2009)

In order to run you need to install the library linearmodels
from Sargent and Starchurski (for the OLS regression):
    "!pip install linearmodels"


computation of the dynamics in the heterogenous-agent neoclassical
growth model with value function iteration

transition dynamics for given initial distribution

initial distribution: uniform

distribution is characterized by the first moment

-   linear interpolation between kbar (aggregate capital stock):

-   linear interpolation between k (individual capital stock):

-   Maximization: golden section search method
    golden.g



"""

# Part 1: import libraries
import numpy as np
from scipy.linalg import inv
from scipy import interpolate
import time
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col


exp = math.exp


# Part 2: functions
# equivec1 computes the ergodic distribution
# see pages 654-656 in Heer/Maussner (2009)
# on DSGE modeling, 2nd edition
def equivec1(p):
    nrows,ncols = p.shape
    for i in range(nrows):
        p[i,i] = p[i,i]-1
    
    q = p[:,0:nrows-1]
    # appends column vector
    q = np.c_[q,np.ones(nrows)]  
    x = np.zeros(nrows-1)
    # appends element in row vector
    x = np.r_[x,1]
    y =  np.transpose(x) @ inv(q)
    return np.transpose(y)


#
# wage function
def wage_rate(k,l):
    return (1-alpha) * k**alpha * l**(-alpha)

# interest rate function
def interest_rate(k,l):
    return alpha * k**(alpha - 1) * l**(1-alpha) - delta


# utility function 
def utility(x): 
    if sigma==1:
        return  np.log(x) 
    else:
        return x**(1-sigma) / (1-sigma)


# right-hand-side of the Bellman equation
# input:
# a0 - wealth a in period t
# a1 - wealth a' in period t+1
# y - employment status in period t
# output:
# rhs of Bellman eq.    
def bellman(a0,a1,y):
   if y==0:     # y=0: employed, y=1: unemployed
      c = (1+(1-tau)*r) * a0 + (1-tau)*w - a1
   else:
       c = (1+(1-tau)*r)*a0 + b - a1
   
   if c<0:
      return negvalue 
   
   if a1>amax:
      return utility(c) + beta*(prob[y,0]*value(amax,0) + prob[y,1]*value(amax,1))
  
   # output: right-hand side of the Bellman equation  
   return utility(c) + beta*(prob[y,0]*value(a1,0) + prob[y,1]*value(a1,1)) 



# value interpolates value function vold[a,e] at a
def value(x,y):
    if y==0:
        return ve1_polate(x)
    else:
        return vu1_polate(x)
    

# value1 rewrites the Bellman function
# as a function of one variable x=a', the next-period asset level
# for given present period assets a[i] and employment status e
# -> function in one argument x necessary for applying
# Golden Section Search
def value1(x):
    return bellman(ainit,x,e)

# searches the MAXIMUM using golden section search
# see also Chapter 11.6.1 in Heer/Maussner, 2009,
# Dynamic General Equilibrium Modeling: Computational
# Methods and Applications, 2nd ed. (or later)
def GoldenSectionMax(f,ay,by,cy,tol):
    r1 = 0.61803399 
    r2 = 1-r1
    x0 = ay
    x3 = cy  
    if abs(cy-by) <= abs(by-ay):
        x1 = by 
        x2 = by + r2 * (cy-by)
    else:
        x2 = by 
        x1 = by - r2 * (by-ay)
    
    f1 = - f(x1)
    f2 = - f(x2)

    while abs(x3-x0) > tol*(abs(x1)+abs(x2)):
        if f2<f1:
            x0 = x1
            x1 = x2
            x2 = r1*x1+r2*x3
            f1 = f2
            f2 = -f(x2)
        else:
            x3 = x2
            x2 = x1
            x1 = r1*x2+r2*x0
            f2 = f1
            f1 = -f(x1)
            
    if f1 <= f2:
        xmin = x1
    else:
        xmin = x2
    
    return xmin

def testfunc(x):
    return -x**2 + 4*x + 6

# test goldensectionsearch
xmax = GoldenSectionMax(testfunc,-2.2,0.0,10.0,0.001)
print(xmax)

start_time = time.time()

# Part 3: Numerical parameters
tol = 0.00001             # stopping criterion for final solution of K 
tol1 = 1e-7             # stopping criterion for golden section search 
tolg = 0.0000001        # stopping criterion for distribution function 
negvalue = -1e10             # initialization for the value function 
eps = 0.05              # minimum distance between grid point amin and amin+varepsilon
                         # in order to check if amin is binding lower bound
nit = 100                # maximum number of maximum iterations over value function 
crit1 = 1 + tol          # percentage deviation in the iteration over the aggregate capital stock
critg = 1 + tol          # deviation of two successive solutions for distribution function:
                         # kritg=sumc(abs(gk0-gk));
nq = 50                  # maximum number of iterations over K, outer loop
#nq=1
nt = 100                 # number of transition periods
nt1 = nt-1               # number of periods in OLS regression of K' on K

psi1 = 0.95              # linear update of K and tau in the outer loop
                         # K^{i+1} = psi1 * K^i + (1-psi1) * K^{i*}, where K^{i*} new solution in iteration i
kbarq = np.zeros(nq)
k1barq = np.zeros(nq)           # convergence of kk1 ?
kritvq = np.zeros(nq)       # convergence of value function 
crit = 1+tol                    # error in value function: crit=meanc(abs(vold-v));

tolgam = 0.00001
kritgam = 1.0 + tolgam          # convergence of the law of motion for K?

gam0q = np.zeros(nq)
gam1q = np.zeros(nq)


# Part 4: Parameterization of the model
alpha = 0.36
beta = 0.995
delta = 0.005
sigma = 2
tau = 0.015
rep = 0.25
pp0 = np.array([[0.9565,0.0435],[0.5,0.5]]) # for the computaton of the
                                            # ergodic distribution
prob = pp0 + 0  # entries are the same as in pp0


# law of motion for capital: initial guess,
# ln(kbar')=gam0+gam1*ln(kbar) 
gam0=0.09
gam1=0.95
gam0 = np.log(330)
gam1 = 0

phi1=0.5  # updating of gam0,gam1 


# asset grid: Value function
amin = -2               # lower bound asset grid over value function
amax = 600             # upper bound    
na = 101                 # number of equispaced points
a =  np.linspace(amin, amax, na)   # asset grid 



# asset grid: Aggregate capital stock K
kmin = 140               # lower bound asset grid over value function
kmax = 340             # upper bound    
nk = 6                 # number of equispaced points
k =  np.linspace(kmin, kmax, nk)   # asset grid 


# asset grid: Distribution function: Step 5.1 from Algorithm 7.2.2
ng = 2*na            # number of asset grid points over distribution 
ag =  np.linspace(amin, amax, ng)   # asset grid for distribution function


# initialization distribution function
gk = np.zeros((nk,2))

# Part 5: Computation of the
# stationary employment /unemployment
# with the help of the ergodic distribution 
print(prob)
pp1 = equivec1(pp0)   
nn0 = pp1[0] # measure of employed households in economy
print(pp1)
print(prob)



# Part 6: Initial guess K, tau 
kk0 = (alpha/(1/beta-1+delta))**(1/(1-alpha))*nn0

# Compute w and r and b
w0 = wage_rate(kk0,nn0)
r0 = interest_rate(kk0, nn0)
b =  rep * (1-tau) * w0  # unemployment insurance


# Part 7:
# initialization of the value function, consumption and next-period asset
#              policy functions
# assumption: employed/unemployed consumes his income permanently 

ve = np.zeros((na,nk))   # value function of employed, individual wealth a[i]
                            # aggregate capital stock k[ik]
vu = np.zeros((na,nk))   # value function of unemployed

for i in range(nk):
    ve[:,i] = utility( (1-tau)*r0*a+(1-tau)*w0) # utility of the employed with asset a
    vu[:,i] = utility( (1-tau)*r0*a+b )    # utility of the unemployed with asset a


ve = ve/(1-beta)
vu = vu/(1-beta) 
copte = np.zeros((na,nk))         # optimal consumption
coptu = np.zeros((na,nk))         # optimal consumption 
aopte = np.zeros((na,nk))         # optimal next-period assets 
aoptu = np.zeros((na,nk))         # optimal next-period assets 


# outer loop over aggregate capital stock K
q = -1
while q<nq-1 and (kritgam>tolgam or q<30):    # iteration over K
    q = q+1
    # ----------------------------------------------------------------
    #
    #    Step 2: Iteration of the value function over a and K
    #
    #    ----------------------------------------------------------------- 

    crit=tol+1
    j = -1
    while j<49 or (j<nit-1 and crit>tol):
        j=j+1
        
        #print("iteration q~j: " +str([q,j]))
        #print("gam0~gam1: " + str([gam0,gam1]))        
        #sec = (time.time() - start_time)
        #ty_res = time.gmtime(sec)
        #res = time.strftime("%H : %M : %S", ty_res)
        #print(res)
        #print("error value function: " +str(crit))

        volde = ve + 0 
        voldu = vu +0
        
        for m in range(nk):
            # equilibrium factor prices with capital stock k[m]
            r = interest_rate(k[m], nn0)
            w = wage_rate(k[m], nn0)
            # equilibrium income tax rate that balances budget
            tau = pp1[1]*w*rep/(k[m]**alpha*nn0**(1-alpha)+pp1[1]*w*rep)            
            b = (1-tau)*rep*w
            # next period capital stock: K' = F(K), eq. (8.9) in book
            k1 = exp(gam0+gam1*np.log(k[m]))  # next period aggregate capital stock
            
            ve1 = np.zeros(na)    # value function of employed with k'=k1 
            vu1 = np.zeros(na)    # value function of unemployed with k'=k1 
            
            for i in range(na):
               if k1 <= kmin:
                   ve1[i] = volde[i,0]
                   vu1[i] = voldu[i,0]
               elif k1 >= kmax:
                   ve1[i] = volde[i,nk-1]
                   vu1[i] = voldu[i,nk-1]
               else:
                   ve_polate = interpolate.interp1d(k,volde[i,:])
                   ve1[i] =  ve_polate(k1)
                   vu_polate = interpolate.interp1d(k,voldu[i,:])
                   vu1[i] =  vu_polate(k1)
                   
            # preparing the value function at k1 for interpolation
            # in calling function value()
            ve1_polate = interpolate.interp1d(a,ve1)
            vu1_polate = interpolate.interp1d(a,vu1)
               
            # iteration over the employment status 
            # e=0 employed, e=1 unemployed 

            for e in range(2): 
                l0 = -1  # initialization of a'
                    # exploiting monotonocity of a'(a)
                    # iteration over the asset grid a = a_1,...a_na
                for i in range(na):
                    ainit = a[i]
                    l = l0
                    v0 = negvalue
                    # iteration over a' to bracket the maximum of 
                    # the rhs of the Bellman equation,
                    # ax < bx < cx and a' is between ax and cx
                
                    ax = a[0] 
                    bx = a[0] 
                    cx = amax
                
                    while l<na-1:
                        l = l+1
                        if e==0:
                            c0 = (1+(1-tau)*r) * ainit + (1-tau)*w - a[l]
                        else:
                            c0 = (1+(1-tau)*r) * ainit + b - a[l]
                    
                    
                        if c0>0:
                            v1 = bellman(ainit,a[l],e)
                            if v1>v0:
                                if e==0:    # employed in period t
                                    ve[i,m] = v1
                                elif e==1:  # unemployed in period t
                                    vu[i,m] = v1
                                    
                                if l==0:
                                    ax = a[0]
                                    bx = a[0] 
                                    cx = a[1]
                                elif l==na-1:
                                    ax = a[na-2] 
                                    bx = a[na-1] 
                                    cx = a[na-1]
                                else:
                                    ax = a[l-1] 
                                    bx = a[l] 
                                    cx = a[l+1]
                            
                                v0 = v1
                                l0 = l-1
                            else:
                                l=na-1   # concavity of value function 
                        
                        else:
                            l=na-1
                    
                    
                    if ax==bx:  # boundary optimum, ax=bx=a[1]  
                        bx = ax+eps*(a[1]-a[0])
                        if bellman(ainit,bx,e)<bellman(ainit,ax,e):
                            if e==0:
                                aopte[i,m] = a[0]
                            else:
                                aoptu[i,m] = a[0]                                
                        else:
                            aopt = GoldenSectionMax(value1,ax,bx,cx,tol1)
                            if e==0:
                                aopte[i,m] = aopt
                            else:
                                aoptu[i,m] = aopt                      
                            
                    elif bx==cx:  # boundary optimum, bx=cx=a[na-1] 
                        bx = cx-eps*(a[na-1]-a[na-2])
                        if bellman(ainit,bx,e) < bellman(ainit,cx,e):
                            if e==0:
                                aopte[i,m] = a[na-1]
                            else:
                                aoptu[i,m] = a[na-1]                                
                        else:
                            aopt = GoldenSectionMax(value1,ax,bx,cx,tol1)
                            if e==0:
                                aopte[i,m] = aopt
                            else:
                                aoptu[i,m] = aopt                      

                    else:   # interior solution ax < bx < cx
                        aopt = GoldenSectionMax(value1,ax,bx,cx,tol1)
                        if e==0:
                            aopte[i,m] = aopt
                        else:
                            aoptu[i,m] = aopt
                    if e==0:
                        ve[i,m] = bellman(a[i],aopte[i,m],e)
                    else:
                        vu[i,m] = bellman(a[i],aoptu[i,m],e)


        if q==0 and j==0: # plotting the value and policy function
            fig, ax = plt.subplots()
            label1 = 'employed'
            label2 = 'unemployed'
            ax.plot(a,ve[:,m], linewidth=2, label=label1)
            ax.plot(a,vu[:,m], linewidth=2, label=label2)
            ax.legend()
            plt.show()



        x0 = abs(volde-ve)
        x1 = abs(voldu-vu)
        x = np.c_[x0,x1]
        crit = x.mean(0)  # mean of the columns, value functions of employed
                          # and unemployd
        crit = max(crit) 

    for m in range(nk):       
        # iteration over value function complete              
        copte[:,m] = (1+(1-tau)*r)*a + (1-tau)*w - aopte[:,m]
        coptu[:,m] = (1+(1-tau)*r)*a + b - aoptu[:,m]


    kritvq[q] = crit
    
    # Step 5.4: iteration of the distribution function 
    #
    print("computation of invariant distribution of wealth..")
    print("iteration q: " + str(q))
    print("kbarq: ") 
    print(kbarq[0:q])
	
    
    # Variant 1: uniform distribution

    gk = np.zeros((ng,2))
    gk[0:na,0] = np.ones(na)* pp1[0] / na
    gk[0:na,1] = np.ones(na) * pp1[1] / na
     
    
    copt0 = np.zeros((na,2)) 
    aopt0 = np.zeros((na,2))

     
    kk0 = sum(np.transpose(gk) @ ag) # initial aggregate capital stock 
    
    
    if q==10: # increase number of transition periods as value function
                # gets more accurate
        nt = 2000
        nt1 = 1000
    
    
    kt = np.zeros(nt) 
    
    for t in range(nt):   # iteration over periods (dynamics of distribution)
        gk0 = gk + 0             # distribution in period t
        gk = np.zeros((ng,2))    # distribution in period t+1
        kt[t] = kk0 
         
        r = interest_rate(kk0, nn0)
        w = wage_rate(kk0, nn0)
        # equilibrium income tax rate that balances budget
        tau = pp1[1]*w*rep/(kk0**alpha*nn0**(1-alpha)+pp1[1]*w*rep)            
        b = (1-tau)*rep*w
         
         
        #print("iteration q: " +str(q))
        #print("transition t: " + str(t))
        #print("kk0~tau~b: " + str([kk0,tau,b]))
        #print("gam0~gam1: " + str([gam0,gam1]))
       
            
        
        
        for i in range(na): 
            aopte_polate = interpolate.interp1d(k,aopte[i,:])
            aoptu_polate = interpolate.interp1d(k,aoptu[i,:])
            copte_polate = interpolate.interp1d(k,copte[i,:])
            coptu_polate = interpolate.interp1d(k,coptu[i,:])
            copt0[i,0] = copte_polate(kk0)
            copt0[i,1] = coptu_polate(kk0)
            aopt0[i,0] = aopte_polate(kk0)
            aopt0[i,1] = aoptu_polate(kk0)
        
        
        gk = np.zeros((ng,2)) # distribution in period t+1
        
         
        for l in range(2):     # iteration over employment types in period t
            # prepare interpolation of optimal next-period asset policy
             
             if l==0:
                 aopt_polate = interpolate.interp1d(a,aopt0[:,0])
             elif l==1:
                 aopt_polate = interpolate.interp1d(a,aopt0[:,1])
             
             for i in range(ng):  # iteration over assets in period t
                 a0 = ag[i]
                 if a0 <= amin:
                     a1 = aopt0[0,l]
                 elif a0 >= amax:
                     a1=aopt0[na-1,l]
                 else:
                    a1 = aopt_polate(a0) # linear interpolation for a'(a) 
                                
                 if a1 <= amin:
                     gk[0,0] = gk[0,0] + gk0[i,l]*prob[l,0]
                     gk[0,1]= gk[0,1] + gk0[i,l]*prob[l,1]
                 elif a1 >= amax:
                     gk[ng-1,0] = gk[ng-1,0] + gk0[i,l]*prob[l,0]
                     gk[ng-1,1] = gk[ng-1,1] + gk0[i,l]*prob[l,1]
                 elif (a1>amin) and (a1<amax):
                     j = sum(ag<=a1) # a1 lies between ag[j-1] and ag[j]
                     n0 = (a1-ag[j-1]) / (ag[j]-ag[j-1])
                     gk[j,0] = gk[j,0] + n0*gk0[i,l]*prob[l,0]
                     gk[j,1] = gk[j,1] + n0*gk0[i,l]*prob[l,1]
                     gk[j-1,0] =gk[j-1,0] + (1-n0)*gk0[i,l]*prob[l,0]
                     gk[j-1,1] =gk[j-1,1] + (1-n0)*gk0[i,l]*prob[l,1]
            
        gk=gk/sum(sum(gk))
        
        
        if t==0:
            np.save('gk1',gk0)
        
        if t==10:
            np.save('gk10',gk)
        
        if t==20:
            np.save('gk20',gk)
        
        if t==1000:
            np.save('gk1000',gk)
        
        
        kk1 = np.transpose(gk[:,0]+gk[:,1]) @ ag # new mean capital stock
        kk0 = kk1
        critg=sum(sum(abs(gk0-gk)))

            
        if kk0 > kmax:
            kk0 = kmax
        elif kk0 < kmin:
            kk0 = kmin


    # ols-estimate gam0,gam1 */
    xi = np.log(kt[1:nt1])
    yi = np.log(kt[2:nt1+1])    
    xi = np.c_[np.ones(nt1-1),xi]    
    # regression step
    reg1 = sm.OLS(endog=yi, exog=xi, missing='drop')    
    results = reg1.fit()
    
    if q==0:
        print(results.summary())

    gamma = results.params
    
    print("iteration q: " +str(q))
    print("kk0~tau~b: " + str([kk0,tau,b]))
    print("gam0~gam1: " + str([gam0,gam1]))      
    sec = (time.time() - start_time)
    ty_res = time.gmtime(sec)
    res = time.strftime("%H : %M : %S", ty_res)
    print(res)
    print("error value function: " +str(crit))
    gam01=gamma[0]
    gam11=gamma[1]
    
    
    np.save('kkt',kt)
    

    kritgam = sum([abs(gam0-gam01),abs(gam1-gam11)])
    
    print("kritgam: ")
    print(kritgam)
    gam0 = phi1 * gam0 + (1-phi1) * gam01
    gam1 = phi1*gam1 + (1-phi1)*gam11
    
    gam0q[q] = gam0
    gam1q[q] = gam1
    kbarq[q]=kk0
    
#plt.axis([0, 20, 0 , 40])
plt.xlabel('Period t')
plt.ylabel('Aggregate capital stock K')
plt.title('Convergence of capital stock')   
plt.plot(kt)
plt.show()


print(results.summary())

plt.xlabel('Capital stock in period t')
plt.ylabel('Capital stock in period t+1')
plt.title('Scatter plot of $K_t$ and $K_{t+1}$') 
plt.scatter(xi[:,1], yi)
plt.show()
 

fig, ax = plt.subplots()
label1 = 't=0'
label2 = 't=10'
label3 = 't=20'
label4 = 't=1000'

gk1 = np.load('gk1.npy')
gk10 = np.load('gk10.npy')
gk20 = np.load('gk20.npy')
gk1000 = np.load('gk1000.npy')

gk1sum = gk1[:,0] + gk1[:,1]
gk10sum = gk10[:,0] + gk10[:,1]
gk20sum = gk20[:,0] + gk20[:,1]
gk1000sum = gk1000[:,0] + gk1000[:,1]

ax.plot(ag,gk1sum, linewidth=2, label=label1)
ax.plot(ag,gk10sum, linewidth=2, label=label2)
ax.plot(ag,gk20sum, linewidth=2, label=label3)
ax.plot(ag,gk1000sum, linewidth=2, label=label4)
ax.legend()
plt.show()
