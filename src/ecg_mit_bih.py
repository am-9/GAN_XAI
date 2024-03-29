"""Preprocessing the MIT-BIH dataset"""

import patient, heartbeat_types
import pandas as pd
import logging

DATA_DIR = '/Users/alainamahalanabis/Downloads/mit-bih-arrhythmia-database-1.0.0/'

train_set = [101]  # DS1
train_set = [str(x) for x in train_set]
test_set = [100]  # DS2
test_set = [str(x) for x in test_set]

train_set = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
             223, 230]  # DS1
train_set = [str(x) for x in train_set]
test_set = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
            233, 234]  # DS2
test_set = [str(x) for x in test_set]

class ECGMitBihDataset(object):
    def __init__(self):
        self.train_patients = [patient.Patient(p) for p in train_set]
        self.test_patients = [patient.Patient(p) for p in test_set]

        self.train_heartbeats = self.concat_heartbeats(self.train_patients)
        self.test_heartbeats = self.concat_heartbeats(self.test_patients)

    @staticmethod
    def concat_heartbeats(patients):
        heartbeats = []
        for p in patients:
            heartbeats += p.heartbeats
        return heartbeats

    def get_heartbeats_of_type(self, aami_label_str, partition):
        """

        :param aami_label_str:
        :return:
        """
        if partition == 'train':
            return [hb for hb in self.train_heartbeats if hb['aami_label_str'] == aami_label_str]
        elif partition == 'test':
            return [hb for hb in self.test_heartbeats if hb['aami_label_str'] == aami_label_str]
        else:
            raise ValueError("Undefined partition: {}".format(partition))

    def num_heartbeats(self, aami_label_str, partition):
        """

        :param aami_label_str:
        :return:
        """
        return len(self.get_heartbeats_of_type(aami_label_str, partition))

    def heartbeats_summaries(self, partition):
        """Create summaries:

        :return:
        """
        heartbeat_summaries = []
        total_beats = 0
        for hb_aami in heartbeat_types.AAMIHeartBeatTypes:
            hb_summary = {}
            hb_summary['heartbeat_aami_str'] = hb_aami.name
            num_hb = self.num_heartbeats(hb_aami.name, partition)
            hb_summary['number_of_beats'] = num_hb
            heartbeat_summaries.append(hb_summary)
            total_beats += hb_summary['number_of_beats']
        total_heartbeats = {}
        total_heartbeats['heartbeat_aami_str'] = 'ALL'
        if partition == 'train':
            total_heartbeats['number_of_beats'] = total_beats
        elif partition == 'test':
            total_heartbeats['number_of_beats'] = total_beats

        heartbeat_summaries.append(total_heartbeats)
        return pd.DataFrame(heartbeat_summaries)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ecg_ds = ECGMitBihDataset()

    print ("training data")
    print(ecg_ds.heartbeats_summaries('train'))

    print ("testing data")
    print(ecg_ds.heartbeats_summaries('test'))
