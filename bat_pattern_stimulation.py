import subprocess, os, time, copy, sys
from tqdm import tqdm
from termcolor import cprint
from modules.utils import SlurmJobDispatcher


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

    if slurm:
        dispatcher = SlurmJobDispatcher(
            path_to_job_file_on_deigo,
            path_to_job_file_from_precision,
            job_file_name,
            output_folder_on_slurm,
            max_slurm_jobs,
        )

    if 'simres' in os.listdir(f'{path_to_job_file_from_precision}/data'):
        os.remove(f'{path_to_job_file_from_precision}/data/simres')
    with open(f'{path_to_job_file_from_precision}/data/simres', 'a') as f:
        f.write(f't,mod,argnass,nass,HAGA,astrocytes,runid\n')
    pbar = tqdm([2, 4, 5, 6, 8, 10, 12, 14, 16, 20])  # number of cell assemblies to be "learned"
    for nass in pbar:
        try:
            for runid in range(20):
                for HAGA in [1, 0]:  # stp-dependent STDP or not
                    for astrocytes in [1, 0]:  # fixed uniform release prob or gamma-distributed
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
                            dispatcher.next_job(dna)

        except Exception as e:
            cprint(f'Exception: {e}', color='red')