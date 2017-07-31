### Preprocessing the test data


# Get group names
echo -n > tmp; for file in *.jpg; do echo $file | cut -d'_' -f 1 >> tmp; done; sort -u tmp > filenames.txt; rm tmp

# Shrink all images
mogrify -path ./small/ -resize 25% *.jpg
cd small

# Generate average per group
cat ../filenames.txt | while read -r line; do convert ${line}_*.jpg -average ${line}_avg.png; done

# Generate median per group
cat ../filenames.txt | while read -r line; do convert ${line}_*.jpg -evaluate-sequence median ${line}_mdn.png; done

# Get difference with average for each image
cat ../filenames.txt | while read -r line; do for i in {01..16}; do composite ${line}_${i}.jpg ${line}_avg.png -compose difference ${line}_${i}_avgdiff.png; done; done

# Get difference with median for each image
cat ../filenames.txt | while read -r line; do for i in {01..16}; do composite ${line}_${i}.jpg ${line}_mdn.png -compose difference ${line}_${i}_mdndiff.png; done; done


## Trimming the images 

mkdir avgdiff_trimmed
mkdir mdndiff_trimmed
mkdir img_trimmed

# Trim resized source images to relevant area

cat ../filenames.txt | while read -r line; 
do for i in {01..16}; 
    do convert ${line}_${i}.jpg \
    -background none \
    -crop \
        `convert ${line}_${i}_avgdiff.png \
        -fuzz 27% \
        -trim \
        -format '%wx%h%O' info:` \
    -extent 480x320 \
    -page \
        `convert ${line}_${i}_avgdiff.png \
        -fuzz 27% \
        -trim \
        -format '%O' info:` \
    -flatten  \
    img_trimmed/${line}_${i}.png; 
    done; 
done

# Trim avgdiff images

cat ../filenames.txt | while read -r line; 
do for i in {01..16}; 
    do convert ${line}_${i}_avgdiff.png \
    -background none \
    -crop \
        `convert ${line}_${i}_avgdiff.png \
        -fuzz 27% \
        -trim \
        -format '%wx%h%O' info:` \
    -extent 480x320 \
    -page \
        `convert ${line}_${i}_avgdiff.png \
        -fuzz 27% \
        -trim \
        -format '%O' info:` \
    -flatten  \
    avgdiff_trimmed/${line}_${i}_avgdiff.png; 
    done; 
done

# Trim median diff images

cat ../filenames.txt | while read -r line; 
do for i in {01..16}; 
    do convert ${line}_${i}_mdndiff.png \
    -background none \
    -crop \
        `convert ${line}_${i}_avgdiff.png \
        -fuzz 27% \
        -trim \
        -format '%wx%h%O' info:` \
    -extent 480x320 \
    -page \
        `convert ${line}_${i}_avgdiff.png \
        -fuzz 27% \
        -trim \
        -format '%O' info:` \
    -flatten  \
    mdndiff_trimmed/${line}_${i}_mdndiff.png; 
    done; 
done

# Clean up

mkdir img_mdn
mv *_mdn.png img_mdn/
mkdir img_avg
mv *_avg.png img_avg/
mkdir avgdiff
mv *_avgdiff.png avgdiff/
mkdir mdndiff
mv *_mdndiff.png mdndiff/
