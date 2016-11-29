cur_dir=$(pwd)
echo $cur_dir

redo=false
data_root_dir="/media/data/"
dataset_name="test_pp"
mapfile="$cur_dir/data/$dataset_name/labelmap.prototxt"
anno_type="detection"
label_type="json"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in train test
do
  python $cur_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $cur_dir/data/$dataset_name/$subset.txt $data_root_dir/opt/arch/$db/$dataset_name"_"$subset"_"$db $cur_dir/examples/$dataset_name 2>&1 | tee $cur_dir/data/$dataset_name/$subset.log
done
