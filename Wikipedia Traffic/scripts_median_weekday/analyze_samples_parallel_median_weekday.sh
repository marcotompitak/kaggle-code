# Run six processes in parallel, each on a different sample
for i in {1..6}
do
    bash analyze_median_weekday.sh input/train_sample_${i}.csv output/scores_median_weekday_sample_${i}.dat &
done

wait
