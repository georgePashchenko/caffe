'''
This script provides evaluation of your trained object detector
'''
import json
import numpy as np
import sys

class EvalCOCO:

    def __init__(self, cocoGT, cocoPR):
        self.minOverlap = 0.5
        self.minProb = 0.01
        self.cocoGT_file = cocoGT
        self.cocoPR_file = cocoPR
        self.cocoGT = []
        self.cocoPR = []
        self.matched = []   #store ids with matched gt bboxs
        self.imageIdsGT = []
        self.categories = {}

        self.fp = []    # false positives
        self.tp = []    # true positives
        self.catGT = []   # category ground truth indexes
        self.catPR = []   # category predicted indexes
        self.scores = []

        self.mAP = 0

        self.numberOfSegments = 10
        self.load_data()

    # load json data
    def load_data(self):
        try:
            file_gt, file_pr = open(self.cocoGT_file), open(self.cocoPR_file)
            cocoGT_all, self.cocoPR = json.load(file_gt), json.load(file_pr)
            self.cocoGT, self.categories = cocoGT_all['annotations'], cocoGT_all['categories']

            file_gt.close(), file_pr.close()

            # sift predictions < minProb
            self.cocoPR = [prediction for prediction in self.cocoPR if prediction['score'] > self.minProb]
            self.imageIdsGT = np.asarray([gt['image_id'] for gt in self.cocoGT])

            self.tp = np.zeros((len(self.cocoPR), 1)).flatten()
            self.fp = np.zeros((len(self.cocoPR), 1)).flatten()
            self.catPR = np.chararray(len(self.cocoPR), 100).flatten()
            self.catGT = np.chararray(len(self.cocoPR), 100).flatten()
            self.scores = np.zeros((len(self.cocoPR), 1)).flatten()

        except ValueError:
            print('Could not load files ' + self.cocoGT_file + ' ' + self.cocoPR_file)

    # get max overlap
    def get_max_overlap(self, prediction):
        ovmax = -10500
        ovmax_id = -1

        # find all bboxs from the image with the same id
        predicted_id = prediction['image_id']
        gt_bbxs = np.where(self.imageIdsGT == predicted_id)[0]
        #gt_bbxs = filter(lambda search_id: search_id['image_id'] == prediction["image_id"], self.cocoGT)
        pr_bbox = prediction['bbox']

        # temporary check for debugging
        # if len(gt_bbxs) == 0:
        #    print('Image %s does not have any bboxs '%(prediction["image_id"]))

        for gt_id, gt_index in enumerate(gt_bbxs):
            gt_bbox = self.cocoGT[gt_index]['bbox']
            # get intersection rectangle
            bi = [max(gt_bbox[0], pr_bbox[0]),
                  max(gt_bbox[1], pr_bbox[1]),
                  min(gt_bbox[0] + gt_bbox[2], pr_bbox[0] + pr_bbox[2]),
                  min(gt_bbox[1] + gt_bbox[3], pr_bbox[1] + pr_bbox[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1

            # check if the bboxes are intersects
            if iw > 0 and ih > 0:
                # compute overlap as area of intersection / area of union

                # squre(gt_bbx) + square(pr_bbx) - square(int_bbx)
                ua = (gt_bbox[2] + 1) * (gt_bbox[3] + 1) + (pr_bbox[2] + 1) * (pr_bbox[3] + 1) - iw * ih
                ov = iw * ih / ua
                if ov > ovmax:
                    ovmax = ov
                    ovmax_id = gt_id
        if ovmax > self.minOverlap:
            return self.cocoGT[gt_bbxs[ovmax_id]]
        else:
            return {}

    # show results
    def get_result(self, class_id, eval_type = 'COCO'):
        if class_id == 0:
            tp, fp, sc = self.tp, self.fp, self.scores
            len_cat_gt = len(self.cocoGT)
        else:
            cat_ind = np.where(self.catPR == class_id)[0]
            len_cat_gt = len(np.where(self.catGT == class_id)[0])

            if len(cat_ind) == 0:
                #print 'There is no results for category ' + cat_ind
                return 0,0

            tp, fp, scores = np.zeros((len(cat_ind), 1)).flatten(), np.zeros((len(cat_ind), 1)).flatten(), np.zeros((len(cat_ind), 1)).flatten()

            # fill tp and fp
            for i,j in enumerate(cat_ind):
                tp[i] = self.tp[j]
                fp[i] = self.fp[j]
                scores[i] = self.scores[j]

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        #tp[tp == 0] = 0.000000000000000001
        recall = tp / len_cat_gt
        precission = np.multiply(tp, 1 / (fp + tp))

        # compute average precision
        mAP, mAR = 0, 0

        if eval_type == 'COCO':
            max_recall = max(recall)
        elif eval_type == 'VOC':
            max_recall = 1.0
        else:
            raise Exception('There is no such type as ' + eval_type + " Nown types: COCO, VOC")

        for t in np.arange(0, max_recall, 1.0 / self.numberOfSegments):
            p = max(precission[recall >= t]) if len(precission[recall >= t]) else 0
            r = min(recall[recall >= t]) if len(recall[recall >= t]) else max(recall)
            mAP += p / self.numberOfSegments
            mAR += r / self.numberOfSegments

        return mAP, mAR

    # evaluate
    def evaluateAll(self):
        # sort predictions by score in descending order
        self.cocoPR = sorted(self.cocoPR, key=lambda k: k['score'], reverse=True)

        for pr_index, prediction in enumerate(self.cocoPR):
            sys.stdout.write('\rProcessing prediction %d/%d'%(pr_index + 1, len(self.cocoPR)))
            gt_bbx = self.get_max_overlap(prediction)
            if not gt_bbx == {}:
                if gt_bbx['id'] in self.matched:
                    self.fp[pr_index] = 1
                elif not gt_bbx['category_id'] == str(prediction['category_id']):
                    self.fp[pr_index] = 1
                else:
                    self.tp[pr_index] = 1
                    self.matched.append(gt_bbx['id'])
            else:
                self.fp[pr_index] = 1

            self.scores[pr_index] = prediction['score']
            self.catPR[pr_index] = str(prediction['category_id'])
        self.catGT = np.asarray([cat_ind['category_id'] for cat_ind in self.cocoGT])





def main():

    argv = sys.argv
    if len(argv) ==1 :
        print 'Incorrect number of parameters. \n <gt_json> <tested_json>'
        return



    gt_json, pr_json = tuple(argv[1:])

    coco = EvalCOCO(cocoGT=gt_json, cocoPR=pr_json)
    coco.evaluateAll()
    mmAP, mmAR = 0,0

    sys.stdout.write('\n|| Name || mAR || mAP||\n')
    for cat in coco.categories:
        if not cat['id'] == 'none_of_the_above' and not cat['name'] == 'background' :
            mAP, mAR = coco.get_result(str(cat['id']), 'COCO')
            sys.stdout.write('|| %s || %.2f%% | %.2f%%|\n'%(cat['name'], mAR*100, mAP*100))
            mmAP += mAP/len(coco.categories)
            mmAR += mAR / len(coco.categories)
    sys.stdout.write('|| ave || %.2f%% || %.2f%%||\n' % (mmAR* 100, mmAP * 100))


    print 'Done'

if __name__ == '__main__':
    main()