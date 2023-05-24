for bs in 32 64 128 256 512 1024
do 
for lr in 2e-5 1e-5 9e-6
#1e-6 1e-7 1e-8
do
for C in 0.1 0.5
do
for wd in 1e-8
#1e-1 1e-4 1e-7 1e-10
do
# for des in True False
# do
for epoch in 30
do
    out_file_name="large_series_tune/out_eps.5_desT_bs${bs}lr${lr}_C${C}_wd${wd}_e${epoch}.txt"
    pro_file_name="large_log/pro_eps.5_desT_bs${bs}lr${lr}_C${C}_wd${wd}_e${epoch}.txt"
    srun -p seas_gpu -t 1-10:00 --mem 20000 --gres=gpu:nvidia_a100-sxm4-80gb:1 python3 dpclip_svhn_gr.py $bs $lr $C $wd $epoch 0.5 1> $out_file_name 2> $pro_file_name &
    echo "start::: bs${bs}lr${lr}_C${C}_wd${wd}_e${epoch}"
# done
done
done
done
done
done
