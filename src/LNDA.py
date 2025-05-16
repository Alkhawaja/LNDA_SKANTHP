# Standard Libraries
import warnings
# Data Science and Machine Learning Libraries
import numpy as np
# PyTorch Libraries
import torch
# Special Functions and Utilities
from scipy.special import psi, polygamma



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx
def init_lda(docs, vocab,testings,y_trains,x, gibbs=False, random_state=0):
    # global V, T, N, D, alpha, Beta, gamma, Phi,rho,A,theta,mue,F,phi,L,Omega
    V = len(vocab)
    N = np.array([doc.shape[0] for doc in docs])
    D = len(docs)
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    A=np.shape(x)[2]
    L=np.shape(x)[3]
    T=np.shape(x)[4]

 
    alpha = np.random.gamma(shape=100, scale=0.01, size=T) 
    mue=np.random.gamma(shape=100, scale=0.01,size=[D,max(N),A,L])
    Beta = np.random.gamma(shape=100, scale=0.01,size=[T,D])
    
    
   
    '''pi'''
    gamma = alpha + np.ones((D, T)) * (N**0.1).reshape(-1, 1) / T
    '''Y'''
    # rho = np.ones((D, max(N),A,L)) 
    '''Z'''
    # Phi = np.ones((D, max(N),A,L,T)) / T
    rho = np.random.gamma(shape=100, scale=0.01,size=[D, max(N),A,L]) 
    Phi = np.random.gamma(shape=100, scale=0.01,size=[D, max(N),A,L,T]) / T
    # zero padding for vectorized operations
    for d, N_d in enumerate(N):
        mue[d,N_d:,:,:]=0 
        Phi[d, N_d:, :] = 0  
        rho[d, N_d:, :] = 0 
        x[d,N_d:,:,:]=0
        
    '''theta'''
    Omega=np.zeros((D,A,L))
    for d in range (D):
        for i in range(len(y_trains[0])):
            for j in range(A):
                Omega[d,j,(find_nearest(np.arange(0,np.ceil(np.max(testings[0])),np.ceil(np.max(testings[0]))/L),np.average(y_trains[d][i,j]))[1])]=1
    phi=np.ones((A,L,T))*0
    # print(f"γ: dim {gamma.shape},\nρ: dim {rho.shape},\nΩ: dim {Omega.shape},\nϕ: dim {phi.shape}\nΦ: dim {Phi.shape},\nN_d max N={max(N)}")
    alpha,mue,Beta,Phi,rho,phi,Omega,gamma=torch.from_numpy(alpha).to('cuda'),torch.from_numpy(mue).to('cuda'),torch.from_numpy(Beta).to('cuda'),torch.from_numpy(Phi).to('cuda'),torch.from_numpy(rho).to('cuda'),torch.from_numpy(phi).to('cuda'),torch.from_numpy(Omega).to('cuda'),torch.from_numpy(gamma).to('cuda')
    return L,Omega,N, D, alpha, Beta, gamma, Phi,rho,A,mue,T,phi
def E_step(Omega,D,T,N,X,Phi, gamma, alpha, Beta,rho,A,mue,x,phi,L):
    """
    Minorize the joint likelihood function via variational inference.
    This is the E-step of variational EM algorithm for LDA.
    """
    # optimize Phi
    for d in range(D):
        for t in (range(T)):
            Phi[d, :N[d], :,:,t] = torch.exp(torch.sum(psi(gamma[d, :].cpu()) - psi(gamma[d, :].cpu().sum()).reshape(-1,1))-rho[d,:N[d],:,:]*phi[:,:,t])
        
        for n in (range(N[d])):
            rho[d, n, :]=torch.exp(torch.log(mue[d,n,:,:]+(torch.sum(x[d,:n]*(1/(n+1))*x[d,:n],axis=0)@Beta)[:,:,d])+torch.log(torch.max(torch.stack([torch.ones(A,L)*1e-20,X[n]-torch.sum(rho[d,n]*Omega[d])-rho[d,n]*Omega[d]]),axis=0).values))

        # Normalize phi
        Phi[d, :N[d]] /= Phi[d, :N[d]].sum(axis=-1).reshape([-1,A,L,1])

    # optimize gamma
    gamma = alpha + Phi.sum(axis=1).sum(axis=1).sum(axis=1)
    
    return Phi, gamma,rho

def E_step_nested(Omega,D,T,N,X,Phi, gamma, alpha, Beta,rho,A,mue,x,phi,L,K):
    """
    Minorize the joint likelihood function via variational inference.
    This is the E-step of variational EM algorithm for LDA.
    """
    # optimize Phi
    for d in range(D):
        for t in (range(T)):
            Phi[d, :N[d], :,:,t] = torch.exp(torch.sum(psi(gamma[d, :].cpu()) - psi(gamma[d, :].cpu().sum()).reshape(-1,1))-rho[d,:N[d],:,:]*phi[:,:,t])
        
        for n in (range(N[d])):
            rho[d, n, :]=torch.exp(torch.log(mue[d,n,:,:]+(torch.sum(x[d,:n]*(1/(n+1))*x[d,:n],axis=0)@Beta)[:,:,d])+torch.log(torch.max(torch.stack([torch.ones(A,L)*1e-20,X[n]-torch.sum(rho[d,n]*Omega[d])-rho[d,n]*Omega[d]]),axis=0).values))

        # Normalize phi
        Phi[d, :N[d]] /= Phi[d, :N[d]].sum(axis=-1).reshape([-1,A,L,1])

    # optimize gamma
    for k in range(K+1):
        if k < 2:
            gamma[:,int(k*T/2):int((k+1)*T/2)]  = alpha[int(k*T/2):int((k+1)*T/2)] +  ((Phi.sum(axis=1)).sum(axis=1)).sum(axis=1)[:,int(k*T/2):int((k+1)*T/2)]
        else:
            gamma [:,T:]= alpha[int(T):] + ((Phi.sum(axis=1)).sum(axis=1)).sum(axis=1)[:,int(T):]
    
    return Phi, gamma,rho


def E_step_enhanced(Omega,D,T,N,X,Phi, gamma, alpha, rho,A,phi,L,LAMBDAS):
    """
    Minorize the joint likelihood function via variational inference.
    This is the E-step of variational EM algorithm for LDA.
    """
    # optimize Phi
    for d in range(D):
        for t in (range(T)):
            Phi[d, :N[d], :,:,t] = torch.exp(torch.sum(psi(gamma[d, :].cpu()) - psi(gamma[d, :].cpu().sum()).reshape(-1,1))-rho[d,:N[d],:,:]*phi[:,:,t])
        
        for n in (range(N[d])):
            THP=LAMBDAS[d][n].to('cuda') if LAMBDAS[d][n].device.type == 'cpu' else LAMBDAS[d][n]
            THP
            rho[d, n, :]=torch.exp(torch.log(THP)+torch.log(torch.max(torch.stack([torch.ones(A,L)*1e-20,X[n]-torch.sum(rho[d,n]*Omega[d])-rho[d,n]*Omega[d]]),axis=0).values))

        # Normalize phi
        # Phi[d, :N[d]] /= Phi[d, :N[d]].sum(axis=-1).reshape([-1,A,L,1])

    # optimize gamma
    gamma = alpha + Phi.sum(axis=1).sum(axis=1).sum(axis=1)
    
    return Phi, gamma,rho

def E_step_nested_enhanced(Omega,D,T,N,X,Phi, gamma, alpha, rho,A,phi,L,LAMBDAS,K):
    """
    Minorize the joint likelihood function via variational inference.
    This is the E-step of variational EM algorithm for LDA.
    """
    # optimize Phi
    for d in range(D):
        for t in (range(T)):
            try:
                Phi[d, :N[d], :,:,t] = torch.exp(torch.sum(psi(gamma[d, :].cpu()) - psi(gamma[d, :].cpu().sum()).reshape(-1,1))-rho[d,:N[d],:,:]*phi[:,:,t])

            except:
                Phi[d, :N[d], :,:,t] = torch.exp(torch.sum(psi(gamma[d, :].cpu().detach().numpy()) - psi(gamma[d, :].cpu().detach().numpy().sum()).reshape(-1,1))-rho[d,:N[d],:,:]*phi[:,:,t])
        
        for n in (range(N[d])):
            THP=LAMBDAS[d][n].to('cuda') if LAMBDAS[d][n].device.type == 'cpu' else LAMBDAS[d][n]
            rho[d, n, :]=torch.exp(torch.log(THP)+torch.log(torch.max(torch.stack([torch.ones(A,L)*1e-20,X[n]-torch.sum(rho[d,n]*Omega[d])-rho[d,n]*Omega[d]]),axis=0).values))

        # Normalize phi
        Phi[d, :N[d]] /= Phi[d, :N[d]].sum(axis=-1).reshape([-1,A,L,1])

    # optimize gamma
    for k in range(K+1):
        if k < 2:
            gamma[:,int(k*T/2):int((k+1)*T/2)]  = alpha[int(k*T/2):int((k+1)*T/2)] +  ((Phi.sum(axis=1)).sum(axis=1)).sum(axis=1)[:,int(k*T/2):int((k+1)*T/2)]
        else:
            gamma [:,T:]= alpha[int(T):] + ((Phi.sum(axis=1)).sum(axis=1)).sum(axis=1)[:,int(T):]
    
    return Phi, gamma,rho

def _update(var, vi_var, const, max_iter=10000, tol=1e-6):
    """
    From appendix A.2 of Blei et al., 2003.
    For hessian with shape `H = diag(h) + 1z1'`
    
    To update alpha, input var=alpha and vi_var=gamma, const=M.
    To update eta, input var=eta and vi_var=lambda, const=k.
    """
    for _ in range(max_iter):
        # store old value
        var0 = var.cpu().clone().detach() 
        
        # g: gradient 
        psi_sum = psi(vi_var.sum(axis=1).cpu().detach().numpy()).reshape(-1, 1)
        g = const * (psi(var.sum().cpu().detach().numpy() )- psi(var.cpu().detach().numpy())) \
            + (psi(vi_var.cpu().detach().numpy()) - psi_sum).sum(axis=0)

        # H = diag(h) + 1z1'
        z = const * polygamma(1, var.cpu().sum()) # z: Hessian constant component
        h = -const * polygamma(1, var.cpu())       # h: Hessian diagonal component
        c = (g / h).sum() / (1./z + (1./h).sum())
        # update var
        var -= torch.tensor((g - c)).to('cuda') / torch.tensor(h).to('cuda')
        
        # check convergence
        err = torch.sqrt(torch.mean((var.cpu() - var0) ** 2))
        crit = err < tol
        if crit:
            break
    else:
        warnings.warn(f"max_iter={max_iter} reached: values might not be optimal.")
    
    return var

def decay(n):
    return 1/(n+2)-1/(n+1)
def M_step(gamma, alpha, Beta, D, N, x,mue,F,step=0.01):
    """
    maximize the lower bound of the likelihood.
    This is the M-step of variational EM algorithm for (smoothed) LDA.
    
    update of alpha follows from appendix A.2 of Blei et al., 2003.
    """
    # update alpha
    alpha = _update(alpha, gamma, D)
    
    b=0
    Nk=torch.zeros((F,D))
    for d in range (D):
        for n in (range(N[d])):
            sum=(torch.sum(x[d,:n]*(1/(n+1))*x[d,:n],axis=0)@Beta)[:,:,d]
            mue[d,n,:,:]=mue[d,n,:,:]/(mue[d,n,:,:]+sum)/N[d]

            b=torch.sum(torch.sum(torch.sum(x[d,:]*(decay(n))*x[d,:],axis=0),axis=0),axis=0)
            sum1=torch.sum(torch.sum(torch.sum(x[d,:]*(1/(n+2))*x[d,:],axis=0),axis=0),axis=0)
            Nk[:,d]=torch.sum(Beta[:,d]*sum1[:]/(torch.sum(mue[d,n,:,:]+sum)),axis=0)

        Beta[:,d]=0.5*step**(-1)*(-b+(b**2+4*step*Nk[:,d])**0.5)

    return alpha, Beta, mue
def M_step_nested(T,gamma, alpha, Beta, D, N, x,mue,F,step=0.01):
    """
    maximize the lower bound of the likelihood.
    This is the M-step of variational EM algorithm for (smoothed) LDA.
    
    update of alpha follows from appendix A.2 of Blei et al., 2003.
    """
    # update alpha
    alpha[:int(T/2)] = _update(alpha[:int(T/2)], gamma[:,:int(T/2)], D)
    alpha[int(T/2):int(T)] = _update(alpha[int(T/2):int(T)], gamma[:,int(T/2):int(T)], D)
    alpha[int(T):] = _update(alpha[int(T):], gamma[:,int(T):], D)

    b=0
    Nk=torch.zeros((F,D))
    for d in range (D):
        for n in (range(N[d])):
            sum=(torch.sum(x[d,:n]*(1/(n+1))*x[d,:n],axis=0)@Beta)[:,:,d]
            mue[d,n,:,:]=mue[d,n,:,:]/(mue[d,n,:,:]+sum)/N[d]

            b=torch.sum(torch.sum(torch.sum(x[d,:]*(decay(n))*x[d,:],axis=0),axis=0),axis=0)
            sum1=torch.sum(torch.sum(torch.sum(x[d,:]*(1/(n+2))*x[d,:],axis=0),axis=0),axis=0)
            Nk[:,d]=torch.sum(Beta[:,d]*sum1[:]/(torch.sum(mue[d,n,:,:]+sum)),axis=0)

        Beta[:,d]=0.5*step**(-1)*(-b+(b**2+4*step*Nk[:,d])**0.5)

    return alpha, Beta, mue
