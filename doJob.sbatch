#!/bin/bash

# load the openmpi module
#SBATCH --job-name=hw1
#SBATCH --output=hw1.out
#SBATCH --error=hw1.err
#SBATCH --nodes=2
#SBATCH --partition=sandyb
#SBATCH --constraint=ib

module load openmpi

#--exclusive          
# Run the process with mpirun. Notice -n is not required. mpirun will
# automatically figure out how many processes to run from the slurm options


echo "Static strategy"
echo "1 cores" 
mpirun -np 1 ./hw1 1 
echo "2 cores" 
mpirun -np 2 ./hw1 1 
echo "4 cores" 
mpirun -np 4 ./hw1 1 
echo "8 cores" 
mpirun -np 8 ./hw1 1 
echo "16 cores"
mpirun -np 16 ./hw1 1
echo "32 cores"
mpirun -np 32 ./hw1 1

echo "Dynamic strategy, Chunk Size is 10000" 
echo "Dynamic strategy should need at least 2 cores." 
echo "2 cores" 
mpirun -np 2 ./hw1 2 10000 
echo "4 cores" 
mpirun -np 4 ./hw1 2 10000 
echo "8 cores" 
mpirun -np 8 ./hw1 2 10000 
echo "16 cores" 
mpirun -np 16 ./hw1 2 10000 
echo "32 cores" 
mpirun -np 32 ./hw1 2 10000
