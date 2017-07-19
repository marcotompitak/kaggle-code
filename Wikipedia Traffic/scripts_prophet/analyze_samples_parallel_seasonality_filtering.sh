# Run six processes in parallel, each on a different sample
for i in {1..6}
do
    bash analyze_seasonality_filtering.sh input/train_sample_${i}.csv output/scores_seasonality_filtering_sample_${i}.dat &
done

wait
