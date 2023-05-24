for C in 1
do
for lr in 1e-6 1e-7 1e-8
do
for e in 15 
do
for bs in 64 128
do
for eps in 0.5 1 3 10
do
    out_file_name="dp_results1/out_eps=${eps}_bs${bs}_lr${lr}_C${C}_wd1e-4_e${e}_dp.txt"
    pro_file_name="dp_log1/pro_eps=${eps}_bs${bs}_lr${lr}_C${C}_wd1e-4_e${e}_dp.txt"
    srun -t 1-10:00 --mem 30000 --gres=gpu:nvidia_a100-sxm4-80gb:1 python3 train_blip_nondp.py $bs $lr $C 1e-4 $e $eps 1 1> $out_file_name 2> $pro_file_name &
# done
    echo "Done"
done
done
done
done
done
