'''
Script for generating train/test list of objects
'''

import json
import os.path as osp
from os import makedirs
import argparse
import random


class AnnotaionConverter:
    def __init__(self, ann_list_file, label_map_file, output_dir, save_empty=False):
        self.inputs = {}
        self.inputs['ann_list_file'] = ann_list_file
        self.inputs['label_map_file'] = label_map_file
        self.inputs['output_dir'] = output_dir
        self.list_of_ann_in = []
        self.list_of_ann_out = []
        self.category_map = {}
        self.result_map = {}
        self.output_dir = output_dir
        self.ann_dir = osp.join(output_dir, 'Annotanions')
        self.save_empty_json = save_empty

        if not osp.exists(self.output_dir) or not osp.exists(self.ann_dir):
            makedirs(self.ann_dir)

        self.load()

    '''
    Load labelmap and annotation list
    '''
    def load(self):
        self.load_label_map()
        self.load_list_of_annotations()

    '''
    Loads category mappings
    '''
    def load_label_map(self):
        label_counter = 1
        with open(self.inputs['label_map_file']) as f:
            lines = f.read().splitlines()
            for line in lines:
                line_split = line.split()
                self.result_map[label_counter] = {'name':line_split[0], 'mapped_ids':[]}
                self.result_map[label_counter]['new_id'] = line_split[1]
                for label_val in line_split[1:]:
                    if not self.result_map[label_counter]['new_id'] == label_val:
                        self.result_map[label_counter]['new_id'] += '_' + label_val
                    self.result_map[label_counter]['mapped_ids'].append(label_val)
                    self.category_map[label_val] = label_counter

                label_counter += 1


    '''
    Loads list of images-annotations
    '''
    def load_list_of_annotations(self):
        with open(self.inputs['ann_list_file']) as f_list:
            self.list_of_ann_in = f_list.read().splitlines()
            print '%s contain %d records'%(self.inputs['ann_list_file'], len(self.list_of_ann_in))

    '''
    Parses json; changes category_id by mapped value, if there no such category_id in mapped values - record will be deleted
    '''
    def process_json(self, file_path):
        json_data_out = {'annotation':[], 'image':{}}
        with open(file_path) as json_file:
            json_data_in = json.load(json_file)
            json_data_out['image'] = json_data_in['image']
            for i_obj, obj in enumerate(json_data_in['annotation']):
                obj_ann = json_data_in['annotation'][i_obj]
                if self.category_map.has_key(obj['category_id']):
                    obj_ann['category_id'] = str(self.result_map[self.category_map[obj['category_id']]]['new_id'])
                    obj_ann['bbox'] = map(int, obj_ann['bbox'])
                    json_data_out['annotation'].append(obj_ann)
                # else:
                #     raise Exception('No such category_id '+ obj['category_id'])


        return json_data_out

    '''
    '''
    def process(self):
        file_with_ann_out = open(osp.join(self.output_dir, 'annotations.txt'), 'w')
        for line in self.list_of_ann_in:
            ann_file = line.split()[-1]
            processed_json = self.process_json(ann_file)
            if processed_json['annotation'] == [] and not self.save_empty_json:
                pass
            else:
                out_json = osp.join(self.ann_dir, osp.basename(ann_file))
                with open(out_json, 'w') as json_file:
                    json.dump(processed_json, json_file)
                    ann_line = line.split()[0] + ' ' + out_json
                    file_with_ann_out.write(ann_line + '\n')
                    self.list_of_ann_out.append(ann_line)

        self.save_labelmap()


    '''
    Saves results label_map in caffe format
    '''
    def save_labelmap(self):
        counter = 1
        with open(osp.join(self.output_dir, 'labelmap.prototxt'), 'w') as labelmap:
            labelmap.write('item {\n  name: "none_of_the_above"\n  label: 0\n  display_name: "background"\n}\n')
            for key in sorted(self.result_map):
                cat_id = 0
                if len(self.result_map[key]['mapped_ids']) > 1:
                    cat_id = self.result_map[key]['mapped_ids'][0]
                    for m_id in self.result_map[key]['mapped_ids'][1:]:
                        cat_id +='_' + m_id
                else:
                    cat_id = self.result_map[key]['mapped_ids'][0]
                labelmap.write(
                    'item {\n  name: "%s"\n  label: %s\n  display_name: "%s"\n}\n' % (cat_id, str(counter), self.result_map[key]['name']))

                counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create new annotated dataset")
    #ann_list_file, label_map_file, output_dir)
    parser.add_argument("ann_list",
                        help="File with list of image/annotation pathes")
    parser.add_argument("--label_map", default=False,
                        help="File which specifies mappings in format: <new_category_name> <mapped_id_1> <mapped_id_2> ...")
    parser.add_argument("--output_dir", default=False,
                        help="Directory where new annotations and file_list wil be stored")
    parser.add_argument("--train_size", default=80,
                        help="Size of the training set relatively to all data in percent (default=80)")
    parser.add_argument("--save_neg", default=False,
                        help="Save empty json without any suitable annotation (default=False)")
    args = parser.parse_args()

    converter = AnnotaionConverter(args.ann_list, args.label_map, args.output_dir, args.save_neg)
    converter.process()
    processed_annotations = random.sample(converter.list_of_ann_out, len(converter.list_of_ann_out))
    train_f = open(osp.join(args.output_dir, 'train.txt'), 'w')
    test_f = open(osp.join(args.output_dir, 'test.txt'), 'w')

    copy_threshold = round(len(processed_annotations) * args.train_size *0.01)
    for i, line in enumerate(processed_annotations):
        if i < copy_threshold:
            train_f.write(line + '\n')
        else:
            test_f.write(line + '\n')

    train_f.close()
    test_f.close()
    print 'Done. Train/Test lists contain %d/%d annotations '%(copy_threshold, len(processed_annotations) - copy_threshold)