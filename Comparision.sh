python GetData.py

cd cifar10
echo cifar10
mpirun -np 4 python LBFGS.py
mpirun -np 4 python SGD.py
mpirun -np 4 python MCMC.py
cd ..

cd cifar100
echo cifar100
mpirun -np 4 python MCMC.py
mpirun -np 4 python SGD.py
mpirun -np 4 python LBFGS.py
cd ..

cd SVHN
echo svhn
mpirun -np 4 python SGD.py
mpirun -np 4 python MCMC.py
mpirun -np 4 python LBFGS.py
cd ..

cd MNIST
echo mnist
mpirun -np 4 python SGD.py
mpirun -np 4 python MCMC.py
mpirun -np 4 python LBFGS.py
