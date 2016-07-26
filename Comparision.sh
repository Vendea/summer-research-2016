mpirun -np 4 python cifar10/LBFGS.py
mpirun -np 4 python cifar100/LBFGS.py
mpirun -np 4 python SVHN/LBFGS.py
mpirun -np 4 python MNIST/LBFGS.py

mpirun -np 4 python cifar10/MCMC.py
mpirun -np 4 python cifar100/MCMC.py
mpirun -np 4 python SVHN/MCMC.py
mpirun -np 4 python MNIST/MCMC.py

mpirun -np 4 python cifar10/SGD.py
mpirun -np 4 python cifar100/SGD.py
mpirun -np 4 python SVHN/SGD.py
mpirun -np 4 python MNIST/SGD.py

