# Since I cannot load the entire test set into memory,
# do it in batches
for i in {0..100063..472}; do
# Use the tsp task spooler to manage the jobs
tsp -n bash -c "python3 cnn_pred.py ${i} $((i + 472))"
done

# Set up submission file
echo "img,rle_mask" > submission.csv

# The python script called above wrote run-length-encoded
# masks to csv files; here we concatenate them
for i in {0..100063..472}; do 
tail -n+2 subm_${i}-$((i + 472)).csv >> submission.csv; 
done
