cd simulations/$1
mpirun -np 8 ~/prefix/bin/qpic_traj &> log &
echo $! > pid
