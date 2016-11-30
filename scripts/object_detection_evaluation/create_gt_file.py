import json
import sys
from google.protobuf import text_format
from caffe.proto import caffe_pb2



def main():
    argv = sys.argv
    output_result = {'annotations':[], 'categories':[]}

    if len(argv) == 1:
        print 'Note enough parameters'
        return

    list_file, labelmap_file, out_json = tuple(argv[1:])
    # read paths to annotations. Expected that list file will have format <image path> <annotation path>
    ann_list = [line.split()[1] for line in open(list_file).read().splitlines()]

    # load labelmap_file
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # parse labels
    for i in range(len(labelmap.item)):
        output_result['categories'].append({
            'id':labelmap.item[i].name,
            'name': labelmap.item[i].display_name
        })

    # parse annotations
    for ann in ann_list:
        json_data = json.load(open(ann))
        image_id = (json_data['image']['file_name']).split('.')[0]
        for json_obj in json_data['annotation']:
            output_result['annotations'].append({
                'image_id': image_id,
                'category_id': json_obj['category_id'],
                'bbox': json_obj['bbox'],
                'id': json_obj['id']
            })
    #save results
    json.dump(output_result, open(out_json, 'w'))


if __name__ == '__main__':
    main()