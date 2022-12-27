import subprocess, os, time, copy, sys
from tqdm import tqdm
from termcolor import cprint


def run_process(runid=None,
                astrocytes=None,
                HAGA=None,
                stim_strength=None,
                nass=None,
                rotate_every_ms=None,
                stim_time_ms=None,
                total_time_ms=None):
    """ run a subprocess """
    cmd = f"python pattern_stimulation.py \
        --runid={runid} \
        --astrocytes={astrocytes} \
        --HAGA={HAGA} \
        --stim_strength={stim_strength} \
        --nass={nass} \
        --rotate_every_ms={rotate_every_ms} \
        --stim_time_ms={stim_time_ms} \
        --total_time_ms={total_time_ms} \
        "

    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)


def marshal(P, n_jobs):
    """ block until the number of unfinished processes is less than n_jobs """
    while len(P) > n_jobs:
        time.sleep(0.1)
        for i in range(len(P)):
            if P[i].poll() is None:
                continue
            if P[i].poll() in [0, 1]:
                P.pop(i)
                break


def checkNumJobs():
    """ check the number of jobs running on slurm """
    subproc = subprocess.Popen("ssh deigo 'squeue'", shell=True, stdout=subprocess.PIPE)
    ret = subproc.stdout.read().decode('UTF-8').split('\n')[1:-1]
    n_jobs_now = len(ret)
    msg = f'Number of jobs running: {n_jobs_now}'
    sys.stdout.write(msg)
    sys.stdout.flush()
    sys.stdout.write("\b" * len(msg))
    return n_jobs_now


def MakeAJobFile(_dna, genID, cell_id):
    """ compile a job file to run each job in slurm """
    global path_to_job_file_from_precision, job_file_name, output_folder_on_slurm
    dna = copy.deepcopy(_dna)
    # dna['gen'] = genID
    # dna['cell_id'] = cell_id
    dna_str = " ".join(f"--{k}={v}" if k not in ['script', 'program'] else f"{v}" for k, v in dna.items()) + "\n"
    with open(f'{path_to_job_file_from_precision}/{job_file_name}', 'w') as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --job-name=JOBNAME1\n")
        f.writelines(f"#SBATCH --mail-user=roman.koshkin@oist.jp\n")
        f.writelines(f"#SBATCH --partition=short\n")
        f.writelines(f"#SBATCH --ntasks=1\n")
        f.writelines(f"#SBATCH --cpus-per-task=1\n")
        f.writelines(f"#SBATCH --mem-per-cpu=1g\n")
        f.writelines(f"#SBATCH --output=./{output_folder_on_slurm}/%j.out\n")
        f.writelines(f"#SBATCH --array=1-1\n")  # submit 4 jobs as an array, give them individual id from 1 to 4
        f.writelines(f"#SBATCH --time=0:15:00\n")
        f.writelines(dna_str)
        # f.writelines("python sim.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $1\n")


def runJob():
    """ run a slurm job based on the file just compiled and written to deigo """
    global path_to_job_file_on_deigo, job_file_name
    commands = f'"cd {path_to_job_file_on_deigo} && sbatch {job_file_name}"'
    os.system(f"ssh deigo {commands}")


if __name__ == '__main__':

    stim_strength = 0.55
    rotate_every_ms = 20
    stim_time_ms = 150000
    total_time_ms = 300000
    n_jobs = 25
    slurm = True
    P = []

    # SLURM params:
    path_to_job_file_on_deigo = '/flash/FukaiU/roman/GA'
    path_to_job_file_from_precision = '/home/roman/flash/GA'
    output_folder_on_slurm = 'output'
    job_file_name = 'start_job.sh'
    immutable_genes = ["program", "script"]
    max_slurm_jobs = 1990

    if 'simres' in os.listdir(f'{path_to_job_file_from_precision}/data'):
        os.remove(f'{path_to_job_file_from_precision}/data/simres')
    with open(f'{path_to_job_file_from_precision}/data/simres', 'a') as f:
        f.write(f't,mod,argnass,nass,HAGA,astrocytes,runid\n')
    pbar = tqdm([2, 4, 5, 6, 8, 10, 12, 14, 16, 20])
    for nass in pbar:
        try:
            for runid in range(10):
                for HAGA in [1, 0]:
                    for astrocytes in [1, 0]:
                        pbar.set_description(desc=f'nass {nass} | HAGA {HAGA} | astr {astrocytes} | runid: {runid}')

                        if not slurm:
                            p = run_process(
                                runid=runid,
                                HAGA=HAGA,
                                astrocytes=astrocytes,
                                stim_strength=stim_strength,
                                nass=nass,
                                rotate_every_ms=rotate_every_ms,
                                stim_time_ms=stim_time_ms,
                                total_time_ms=total_time_ms,
                            )
                            P.append(p)
                            time.sleep(1)
                            marshal(P, n_jobs)  # block if the number of unfinished jobs is > num_jobs
                        else:
                            dna = {
                                "program": "python",
                                "script": "pattern_stimulation.py",
                                "runid": runid,
                                "HAGA": HAGA,
                                "astrocytes": astrocytes,
                                "stim_strength": stim_strength,
                                "nass": nass,
                                "rotate_every_ms": rotate_every_ms,
                                "stim_time_ms": stim_time_ms,
                                "total_time_ms": total_time_ms,
                            }
                            num_unfinished_jobs = checkNumJobs()
                            while num_unfinished_jobs > max_slurm_jobs:
                                time.sleep(5)
                                num_unfinished_jobs = checkNumJobs()
                            MakeAJobFile(dna, None, None)
                            runJob()
                            time.sleep(1)

        except Exception as e:
            cprint(f'Exception: {e}', color='red')