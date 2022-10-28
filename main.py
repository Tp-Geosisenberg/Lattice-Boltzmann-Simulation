import numpy as np
from matplotlib import pyplot
from tqdm import tqdm

fig = pyplot.figure()

def main():
    Nx = 400
    Ny = 100
    tau = 0.6    # collision timescale
    Nt = 500       
    rho0 = 100    # average density
        #lattice speeds and weights

    NL = 9
    """
        Part of array will be talk about Molecular Dynamics(Microscpic) which involve Lattice Boltzman(Mesoscopic) 
        are fundemental of Navier Stokes and Navier Stokes(Macroscopic) that observe natural.
        If you want to study about Navier Stoke Equation(NSE) are know something about Lattice Boltzman
    """
        # Array of Mesocopic
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
    X, Y = np.meshgrid(range(Nx), range(Ny))

        #initial coditions
    F = np.ones((Ny,Nx,NL)) + 0.01*np.random.randn(Ny,Nx,NL)
    F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] *= rho0 / rho


        #Object 
    cylinder = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**2
        #Simulation <Main>

    for it in tqdm(range(Nt)):
        pyplot.title(f"Frame :{it+1}")
        for i , cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)


        #Reflective boundaries

        bndryF = F[cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]] # Set of nodes opposite velocity when it hits the object(Cylinder)


        # Calculate Fluid

        rho = np.sum(F,2)
        ux  = np.sum(F*cxs,2) / rho
        uy  = np.sum(F*cys,2) / rho


        F[cylinder ,:] = bndryF
        ux[cylinder] = 0
        uy[cylinder] = 0

        # Collision

        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:,:,i] = rho*w* (1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)
      
            # This is define the equilibrium state
        """
         This is define the equilibrium state as a result of collison which depends on the fluid model's equation of state(EOS)
        """
        F += -(1.0/tau) * (F - Feq)
        F[cylinder, :] =bndryF

        #pyplot.imshow(np.sqrt(ux**2 + uy **2))
        
        im = pyplot.imshow(np.sqrt(ux**2 + uy **2), animated=True)
        
        pyplot.pause(0.01)
    
    import matplotlib.animation as animation
    im.set_array(main())

    ani = animation.FuncAnimation(fig, main, interval=50, blit=True)
    pyplot.show()

if __name__ == '__main__':
    main()
