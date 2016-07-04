#!/bin/bash
QUAD_P=( 1e-03 1e-02 1e-04 )
REGP_P=( 1e-03 1e-04 1e-05 1e-06 1e-07 )



the_job() {
   echo "$(date +%T) - Start of job $1"
   docker run -e PYTHONPATH='/' -v `pwd`/data:/data -v `pwd`/Tests:/Tests -i joint python -u scripts/reuters.py --quadratic ${QUAD_P[${i}]} --loss 1. --regP ${REGP_P[${j}]} --nit 100 --ndims 4 >> run${1}.log
   echo "$(date +%T) - End   of job $1"
}

SUBMIT_JOB_LIMIT=4
SUBMIT_JOB_SLEEP=60

job_cnt=0
job_act=0
for (( i=0; i< ${#QUAD_P[@]}; i++)); do
    for (( j=0; j< ${#QUAD_P[@]}; j++ )); do
#        for (( k=0; k< ${#REGQ_P[@]}; k++)); do
             job_act=$(jobs | wc -l)
             while ((job_act >= $SUBMIT_JOB_LIMIT))
             do
                sleep $SUBMIT_JOB_SLEEP
                job_act=$(jobs | wc -l)
#                python -u theano/reuters_v1.py --quadratic ${BLUN_P[${i}]} --loss 1. --regP ${REGP_P[${j}]} --nit 100 --ndims 4 >> run1.log
             done
             (( job_cnt += 1))
             the_job $(($job_cnt % SUBMIT_JOB_LIMIT)) ${QUAD_P[${i}]} ${REGP_P[${j}]} &
#        done
    done
done
