# Clear output file
> $2

# Get length of input file
read lines words chars filename <<< $(wc $1)

reallines="$(($lines-1))"

# Loop over input and feed to Python script
for i in $(seq 1 $reallines)
    do
        python3 forecast_median.py $1 $i $2 > /dev/null
    done
