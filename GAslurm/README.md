# quick start

## GA on slurm
- make the following folders on `deigo`
    - /flash/FukaiU/roman/GA/data
    - /flash/FukaiU/roman/GA/output
- copy `sim.py` to /flash/FukaiU/roman/GA
- copy `configs`, `modules`, `scripts` and `src` to /flash/FukaiU/roman
- start session on `short`, build binaries with `sh build.sh`
- be careful with folder structure, with slurm it's more complicated than if you run a smaller-scale evolutionary HP sweep on you local machine (with a max of num_cores of children in a generation). With slurm you can have up to 2000 children in a generation.
- to run, cd to `GA` on precision, and run `python runGA.py`.

## GA on precision
- `cd CODE/spiking-autoencoder/GA`
- `python runGA.py`