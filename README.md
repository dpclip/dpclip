# Training a Differentially Private CLIP

## Useful Commands

To create a new tmux session to close your laptop while running: `tmux new` (can also name your session if you want to access it later).

Use conda to create an environment with all installed packages: `conda activate myenv`

Request for resources from the FAS cluster: `salloc -p gpu_test -t 0-01:00 --mem 8000 --gres=gpu:1`
Can specify which gpu to use, e.g. `nvidia_a100-sxm4-80gb`

OR (Recommended) do `srun -p seas_gpu -t 6-10:00 --mem 80000 --gres=gpu:nvidia_a100-sxm4-80gb:1 nohup python3 xxx.py &`
`seas_gpu` is optional

To view currently running jobs: `squeue`

To cancel a job: `scancel <job_id>`

To run a job: `python3 dpclip.py > output.txt &`

To direct output to `o.txt` and process/error to another file `e.txt` and show the pid: `nohup python3 programname.py 1> o.txt  2> e.txt &`

To run a job on the i^th gpu: `CUDA_VISIBLE_DEVICES=i`

To show jobs running on gpus: `nvidia-smi`

To cancel all jobs: `scancel -u username`



## Fashion MNIST/MNIST benchmarks

- https://proceedings.neurips.cc/paper/2021/file/67ed94744426295f96268f4ac1881b46-Paper.pdf
- https://arxiv.org/pdf/2301.13389v1.pdf
