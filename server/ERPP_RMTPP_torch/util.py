import pandas
from tqdm import tqdm
import numpy as np
import torch
from collections import Counter
import math
import pprint




def sigmoid(x): 
    return 1 / (1 + math.exp(-x))



class ATMDataset:
    """
    helper class for the neural network that
    is in charge of doing the sliding window algorithm
    over the sequences and get the time differences
    in the timestamp column of the data. 

    it can be seen as a wrapper that does some further 
    preprocessing that is very especific to the NN.        
    """
    def __init__(self, config, data, case_id, timestamp_key,event_key):
        self.id = list(data[case_id])
        self.time = list(data[timestamp_key] )
        self.event = list(data[event_key])
        
        self.config = config
        self.seq_len = config.seq_len
        self.time_seqs, self.event_seqs = self.generate_sequence()
        self.statistic()

    def generate_sequence(self):
        """
        use the sliding window algorithm so that the sequences
        fit in the NN. 
        """
        MAX_INTERVAL_VARIANCE = 1
        pbar = tqdm(total=len(self.id) - self.seq_len + 1) #tqdm is the progress bar
        time_seqs = []
        event_seqs = []
        cur_end = self.seq_len - 1
        #: this is a sliding window algorithm to cut each input sequence into sub sequences of the same length
        while cur_end < len(self.id):
            pbar.update(1)
            cur_start = cur_end - self.seq_len + 1
            if self.id[cur_start] != self.id[cur_end]:
                cur_end += 1
                continue

            subseq = self.time[cur_start:cur_end + 1]
            #print(subseq)
            # if max(subseq) - min(subseq) > MAX_INTERVAL_VARIANCE:
            #     if self.subset == "train":
            #         cur_end += 1
            #         continue
            time_seqs.append(subseq)
            event_seqs.append(self.event[cur_start:cur_end + 1])
            cur_end += 1
        return time_seqs, event_seqs

    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):
        """
        :return: two tensors, one containing the time differences
        between adjacent time stamps and the other one the events.  
        """
        #print(batch)
        times, events = [], []
        for time, event in batch:
            time = np.array([time[0]] + time)

            time = np.diff(time)
            times.append(time)
            events.append(event)

        #return torch.FloatTensor(times), torch.LongTensor(events)
        return torch.FloatTensor(np.asarray(times)), torch.LongTensor(np.asarray(events))

    def statistic(self):
        print("TOTAL SEQs:", len(self.time_seqs))
        intervals = np.diff(np.array(self.time))
        for thr in [0.001, 0.01, 0.1, 1, 10, 100]:
            print(f"<{thr} = {np.mean(intervals < thr)}")

    def importance_weight(self, count):
        """
        used for CrossEntropyLoss 
        """
        weight = [len(self.event) / count[k] for k in sorted(count.keys())]
        return weight


def abs_error(pred, gold):
    return np.mean(np.abs(pred - gold))


def clf_metric(pred, gold, n_class):
    """
    compute test metrics.
    :return:
    - recall 
    - precision
    - f1 score
    """
    gold_count = Counter(gold)
    pred_count = Counter(pred)
    prec = recall = 0
    pcnt = rcnt = 0
    for i in range(n_class):
        #print(np.logical_and(pred == gold, pred == i))
        match_count = np.logical_and(pred == gold, pred == i).sum()
        if gold_count[i] != 0:
            prec += match_count / gold_count[i]
            pcnt += 1
        if pred_count[i] != 0:
            recall += match_count / pred_count[i]
            rcnt += 1
    prec /= pcnt
    recall /= rcnt
    print(f"pcnt={pcnt}, rcnt={rcnt}")
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1
