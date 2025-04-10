#!/bin/bash -l
#
#-------------SECTION 1----------------
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -J Sim1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 16
#SBATCH -t 5-00:00:00
#!--mem-per-cpu=9500M
#!-C "scarf15|scarf16"
#!SBATCH --exclusive
#SBATCH --mail-user=scarf709
#SBATCH --mail-type=NONE

#NONE, BEGIN, END, FAIL, REQUEUE, ALL, TIME_LIMIT, 
#TIME_LIMIT_90, TIME_LIMIT_80, TIME_LIMIT_50


WORKDIR=/work3/astec/scarf777/scarf777/New_epoch_particle_id/Sim1
EXEC=$WORKDIR/epoch3d

#BINARYLOC=/work3/astec/scarf777/scarf777/New_epoch_particle_id/Sim1
#cp -rf $BINARYLOC $EXEC

DATA=$WORKDIR/Data
OUTLOG=$WORKDIR/log.txt

chmod 744 $EXEC
mkdir -p $DATA
if [ -e $WORKDIR/input.deck ]
	then
	  cp -rf $WORKDIR/input.deck $DATA/input.deck
fi

STDIN=/dev/null

#-----------------SECTION 2---------------------
# Load modules from the cluster, needed to run EPOCH. These are the modules that were used when compiling.

module load intel/18.0.3 intel/mpi/18.0.3

#----------------SECTION 3-------------------------

#HDF5-INTEL17
#export PATH=$PATH:/users/taperera/lib/hdf5-1.10.3p-intel2018u1/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/taperera/lib/hdf5-1.10.3p-intel2018u1/lib

#szip
#export PATH=$PATH:/users/taperera/lib/szip-2.1.1-intel2018u1/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/taperera/lib/szip-2.1.1-intel2018u1/lib

#Jsonfortran
#export PATH=$PATH:/users/taperera/lib/json-fortran-6.9.0-intel2018u1/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/taperera/lib/json-fortran-6.9.0-intel2018u1/lib

#----------------------SECTION 4--------------------

echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

echo Executable file:                              
echo MPI parallel job.                                  
echo -------------  
echo Job output begins                                           
echo -----------------                                           
echo

hostname

echo "Print the following environmetal variables:"
echo "Job name                     : $SLURM_JOB_NAME"
echo "Job ID                       : $SLURM_JOB_ID"
echo "Job user                     : $SLURM_JOB_USER"
echo "Job array index              : $SLURM_ARRAY_TASK_ID"
echo "Submit directory             : $SLURM_SUBMIT_DIR"
echo "Temporary directory          : $TMPDIR"
echo "Submit host                  : $SLURM_SUBMIT_HOST"
echo "Queue/Partition name         : $SLURM_JOB_PARTITION"
echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Hostname of 1st node         : $HOSTNAME"
echo "Number of nodes allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Number of processes          : $SLURM_NTASKS"
echo "Number of processes per node : $SLURM_TASKS_PER_NODE"
echo "Requested tasks per node     : $SLURM_NTASKS_PER_NODE"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"
echo "Scheduling priority          : $SLURM_PRIO_PROCESS"




echo "Running parallel job:"

mpirun  $EXEC <<< $DATA  > $OUTLOG
ret=$? #This stores error info in the variable $ret... (see last line)



#To be written to the log when the mpirun task completes successfully
echo   
echo ---------------                                           
echo Job output ends                                           
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   
exit $ret