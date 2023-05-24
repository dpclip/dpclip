for lr in 1e-6 1e-7 1e-8
do
for e in 15 
do
for bs in 64 128
do
    out_file_name="ndp_results/out_bs${bs}_lr${lr}_wd1e-4_e${e}_ndp.txt"
    pro_file_name="ndp_log/pro_bs${bs}_lr${lr}_wd1e-4_e${e}_ndp.txt"
    srun -p seas_gpu -t 1-10:00 --mem 30000 --gres=gpu:nvidia_a100-sxm4-80gb:1 python3 train_blip_nondp.py $bs $lr 1 1e-4 $e 1 0 1> $out_file_name 2> $pro_file_name &
# done
    echo "Done"
done
done
done

