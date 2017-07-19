# Run six processes in parallel, each on a different sample
for i in {0..5}
do
    bash full_dataset_analyze_median_weekday.sh input/train_1_split${i} output/full_dataset_pred_median_weekday_split_${i}.dat &
done

wait
