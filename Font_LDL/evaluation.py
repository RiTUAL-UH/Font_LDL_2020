import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import config
import os


def average(lst):
    return sum(lst) / len(lst)

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def flatten_lst(lst):
    flat_list = [item for sublist in lst for item in sublist]
    return flat_list

def compute(all_scores, all_labels):
    """
    This function computes Font recall and F-score metrics.
    :param all_scores: model scores
    :param all_labels: ground_truth labels
    :return:
    """
    top_m = [1, 3, 5]
    m1_score_lst =[]
    m1_label_lst =[]
    m3_score_lst = []
    m3_label_lst = []
    m5_score_lst = []
    m5_label_lst = []
    for m in top_m:
        score_lst = []

        # computing scores:
        for s in all_scores:
            h = m
            s = np.array(s)
            ind_score = np.argsort(s)[-h:]
            score_lst.append(ind_score.tolist())
            if m == 1:
                m1_score_lst.append(ind_score.tolist())
            elif m == 3:
                m3_score_lst.append(ind_score.tolist())
            elif m == 5:
                m5_score_lst.append(ind_score.tolist())
            else:
                print("[LOG] ERROR Wrong value")

        # computing labels:
        label_lst = []
        for l in all_labels:
            # if label list contains several top values with the same amount we consider all
            h = m
            if len(l) > h:
               while (h < (len(l)) and l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]] ):
                    h += 1
            l = np.array(l)
            ind_label = np.argsort(l)[-h:]
            label_lst.append(ind_label.tolist())

            if m==1:
                m1_label_lst.append(ind_label.tolist())
            elif m==3:
                m3_label_lst.append(ind_label.tolist())
            elif m==5:
                m5_label_lst.append(ind_label.tolist())
            else:
                print("[LOG] ERROR Wrong value")

        # computing the intersection between scores and ground_truth labels:
        for i in range(len(score_lst)):
            intersect = intersection(score_lst[i], label_lst[i])

            if m == 1 and len(m1_label_lst[i]) > 1:
                m1_label_lst[i] = intersect if len(intersect) > 0 else [m1_label_lst[i][-1]]

            elif m == 3 and len(m3_label_lst[i]) > 3:
                m3_label_lst[i], intersect = bring_intersect_last(lst = m3_label_lst[i], intersect = intersect)
                m3_label_lst[i] = m3_label_lst[i][-3:]

            elif m == 5 and len(m5_label_lst[i]) > 5:
                m5_label_lst[i], intersect = bring_intersect_last(lst = m5_label_lst[i], intersect = intersect)
                m5_label_lst[i] = m5_label_lst[i][-5:]
            else:
                pass

    # Computing F1_score:
    #*************************  M1:
    m1_score_lst = flatten_lst(m1_score_lst)
    m1_label_lst = flatten_lst(m1_label_lst)
    assert len(m1_label_lst) == len(m1_score_lst), "not equal len!"
    m1_f_score_weighted = f1_score(m1_label_lst, m1_score_lst, average="weighted")
    # ************************ M3:
    binarozer = MultiLabelBinarizer().fit(m3_label_lst)
    m3_f_score_weighted = f1_score(binarozer.transform(m3_label_lst), binarozer.transform(m3_score_lst), average="weighted")
    # ************************ M5:
    binarozer = MultiLabelBinarizer().fit(m5_label_lst)
    m5_f_score_weighted = f1_score(binarozer.transform(m5_label_lst), binarozer.transform(m5_score_lst), average="weighted")

    # Computing Font Recall:
    font_recall_m3 = compute_tr(m3_score_lst, m3_label_lst)
    font_recall_m5 = compute_tr(m5_score_lst, m5_label_lst)


    print("M3_score_lst: {}".format(m3_score_lst[:10]))
    print("M3_label_lst: {}".format(m3_label_lst[:10]))
    print("FR 3: {}".format(font_recall_m3))
    print("FR 5: {}".format(font_recall_m5))
    print("Top 1 f_weighted: {}".format( m1_f_score_weighted))
    print("Top 3 f_weighted: {}".format(m3_f_score_weighted))
    print("Top 5 f_weighted: {}".format(m5_f_score_weighted))

def bring_intersect_last(lst, intersect):
    for i in intersect:
        lst.remove(i)
        lst =  lst + [i]
    return lst, intersect

def read_lines(filename):
    with open(filename, 'r') as fp:
        lines = [line.strip() for line in fp]
    return lines

def read_file(filename):
    lines = read_lines(filename)
    all_labels = []
    for line in lines:
        if line:
            splitted = line.split("\t\t")
            lists_of_labels = splitted[1].split("\t")
            all_labels.append(lists_of_labels)
    return all_labels

def compute_tr(score_lst, label_lst):
    """
    This function computes Tag Recall for all ten classes
    :param score_lst: score_lst
    :param label_lst: label_lst
    :return: Font recall score
    """
    assert len(score_lst) ==len(label_lst)
    ret = [0] * 10
    for f in [0,1,2,3,4,5,6,7,8,9]:
        correct_count = 0
        all_count = 0
        for i in range(len(label_lst)):
            if f in label_lst[i]:
                all_count += 1
            if f in label_lst[i] and f in score_lst[i]:
                correct_count += 1
        ret[f] = correct_count/float(all_count)
    return average(ret)



if __name__ == '__main__':
    path = config.score_output
    score_name =  "score_test.txt"
    target_name = "target_test.txt"
    scores = read_file(os.path.join(path, score_name ))
    targets = read_file(os.path.join(path,target_name))

    scores_f =[]
    targets_f = []
    for i in range(len(scores)):
        targets_f.append([float(k) for k in targets[i]])
        scores_f.append([float(k) for k in scores[i]])

    compute(scores_f, targets_f)
    print("[LOG] path: ", path)


