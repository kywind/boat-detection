# python detect.py --save-txt --conf 0.1 --img 608 --output inference/testimgs --source testimgs/ --cfg cfg/satellite-anchor.cfg --name cfg/satellite.names --weights runs/exp39_test-allanno-350resume/weights/best_test-allanno-350resume.pt
array=(2001 2002 2011 2012 2013 2014 2015 2016 2018 2019 2020)
for year in ${array[@]}
do
    python detect.py --save-txt --conf 0.02 --img 608 \
    --output inference/${year} \
    --source detect_buffer_jpg_${year}/ \
    --cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
    --weights runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt
done


# python detect_new.py --conf 0.02 --img 608 --output inference/no_water \
# --source imglist/val_nowater.txt --save-img \
# --cfg cfg/satellite-anchor.cfg --name cfg/satellite.names \
# --weights weights/V2_best_608.pt