import numpy as np
import os
import logging
# from matplotlib import pyplot as plt
# from bokeh.io import output_file, show
# from bokeh.layouts import row
# from bokeh.plotting import figure

import heartbeat_types
import wfdb
import pandas as pd

DATA_DIR = '/Users/alainamahalanabis/Documents/'+ 'ecg_pytorch/ecg_pytorch/data_reader/text_files/'
dat_dir = '/Users/alainamahalanabis/Downloads/mit-bih-arrhythmia-database-1.0.0/'

train_set = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
             223, 230]  # DS1
train_set = [str(x) for x in train_set]
test_set = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
            233, 234]  # DS2
test_set = [str(x) for x in test_set]


class Patient(object):
    """Patient object represents a patient from the MIT-BIH AR database.
    Attributes:
    """

    def __init__(self, patient_number):
        """Init Patient object from corresponding text file.

        :param patient_number: string which represents the patient number.
        """
        logging.info("Creating patient {}...".format(patient_number))
        self.patient_number = patient_number
        self.signals, self.additional_fields = self.get_raw_signals()
        self.mit_bih_labels_str, self.labels_locations, self.labels_descriptions = self.get_annotations()
        self.heartbeats = self.slice_heartbeats()
        logging.info("Completed patient {}.\n\n".format(patient_number))

    @DeprecationWarning
    def read_raw_data(self):
        """Read patient's data file.

        :return:
        """
        print('READING RAW DATA')
        dat_file = os.path.join(DATA_DIR, self.patient_number + '.txt')
        if not os.path.exists(dat_file):
            raise AssertionError("{} doesn't exist.".format(dat_file))
        time = []
        voltage1 = []
        voltage2 = []
        with open(dat_file, 'r') as fd:
            for line in fd:
                line = line.split()
                time.append(line[0])
                voltage1.append(float(line[1]))
                voltage2.append(float(line[2]))

        tags_file = os.path.join(DATA_DIR, self.patient_number + '_tag.txt')
        if not os.path.exists(dat_file):
            raise AssertionError("{} doesn't exist.".format(tags_file))
        tags_time = []
        tags = []
        r_peaks_indexes = []
        with open(tags_file, 'r') as fd:
            for line in fd:
                line = line.split()
                tags_time.append(line[0])
                tags.append(line[2])
                r_peaks_indexes.append(int(line[1]))
        return time, voltage1, voltage2, tags_time, tags, r_peaks_indexes

    def get_raw_signals(self):
        """Get raw signal using the wfdb package.

        :return: signals : numpy array
                    A 2d numpy array storing the physical signals from the record.
                fields : dict
                    A dictionary containing several key attributes of the read
                    record:
                      - fs: The sampling frequency of the record
                      - units: The units for each channel
                      - sig_name: The signal name for each channel
                      - comments: Any comments written in the header
        """
        #print ("getting raw signal")
        dat_file = os.path.join(dat_dir, self.patient_number)
        #print(dat_file)
        signals, fields = wfdb.rdsamp(dat_file, warn_empty=True)
        #print (signals)
        logging.info("Patient {} additional info: {}".format(self.patient_number, fields))
        return signals, fields

    def get_annotations(self):
        """Get signal annotation using the wfdb package.

        :return:
        """

        dat_file = os.path.join(dat_dir, self.patient_number)
        ann = wfdb.rdann(dat_file, 'atr', return_label_elements=['symbol', 'label_store',
                                                                                            'description'],
                         summarize_labels=True)

        mit_bih_labels_str = ann.symbol

        labels_locations = ann.sample

        labels_description = ann.description

        return mit_bih_labels_str, labels_locations, labels_description

    def slice_heartbeats(self):
        """Slice heartbeats from the raw signal.

        :return:
        """
        sampling_rate = self.additional_fields['fs']  # 360 samples per second
        logging.info("Sampling rate: {}".format(sampling_rate))
        assert sampling_rate == 360
        before = 0.2  # 0.2 seconds == 0.2 * 10^3 miliseconds == 200 ms
        after = 0.4  # --> 400 ms

        #
        # Find lead 2 position:
        #
        lead_pos = None
        for i, lead in enumerate(self.additional_fields['sig_name']):
            if lead == 'MLII':
                lead_pos = i
        if lead_pos is None:
            raise AssertionError("Didn't find lead 2 position. LEADS: {}".format(self.additional_fields['sig_name']))
        logging.info("LEAD 2 position: {}".format(lead_pos))
        ecg_signal = self.signals[:, lead_pos]
        r_peak_locations = self.labels_locations

        # convert seconds to samples
        before = int(before * sampling_rate)  # Number of samples per 200 ms.
        after = int(after * sampling_rate)  # number of samples per 400 ms.

        len_of_signal = len(ecg_signal)

        heart_beats = []

        for ind, r_peak in enumerate(r_peak_locations):
            start = r_peak - before
            if start < 0:
                logging.info("Skipping beat {}".format(ind))
                continue
            end = r_peak + after
            if end > len_of_signal - 1:
                logging.info("Skipping beat {}".format(ind))
                break
            heart_beats_dict = {}
            heart_beat = np.array(ecg_signal[start:end])
            heart_beats_dict['patient_number'] = self.patient_number
            heart_beats_dict['cardiac_cycle'] = heart_beat
            aami_label_str = heartbeat_types.convert_heartbeat_mit_bih_to_aami(self.mit_bih_labels_str[ind])
            aami_label_ind = heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class(self.mit_bih_labels_str[ind])
            heart_beats_dict['mit_bih_label_str'] = self.mit_bih_labels_str[ind]
            heart_beats_dict['aami_label_str'] = aami_label_str
            heart_beats_dict['aami_label_ind'] = aami_label_ind
            heart_beats_dict['aami_label_one_hot'] = heartbeat_types.convert_to_one_hot(aami_label_ind)
            heart_beats_dict['beat_ind'] = ind
            heart_beats_dict['lead'] = 'MLII'
            heart_beats.append(heart_beats_dict)
        #print (len(heart_beats))
        return heart_beats

    def get_heartbeats_of_type(self, aami_label_str):
        """

        :param aami_label_str:
        :return:
        """
        return [hb for hb in self.heartbeats if hb['aami_label_str'] == aami_label_str]

    def num_heartbeats(self, aami_label_str):
        """

        :param aami_label_str:
        :return:
        """
        return len(self.get_heartbeats_of_type(aami_label_str))

    def heartbeats_summaries(self):
        """Create summaries:


        :return:
        """
        heartbeat_summaries = []
        for hb_aami in heartbeat_types.AAMIHeartBeatTypes:
            hb_summary = {}
            hb_summary['heartbeat_aami_str'] = hb_aami.name
            num_hb = self.num_heartbeats(hb_aami.name)
            hb_summary['number_of_beats'] = num_hb
            heartbeat_summaries.append(hb_summary)
        total_summary = {}
        total_summary['heartbeat_aami_str'] = 'ALL'
        total_summary['number_of_beats'] = len(self.heartbeats)
        heartbeat_summaries.append(total_summary)
        return pd.DataFrame(heartbeat_summaries)

    def get_patient_df(self):
        """Get data frame with patient details per heartbeat.

        :return: pandas dataframe.
        """
        df = pd.DataFrame(self.heartbeats)
        df.drop(columns=['cardiac_cycle'], inplace=True)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p_100 = Patient('114')
    df = p_100.get_patient_df()
    # print(df)
    print(p_100.heartbeats_summaries())
    # heartbeats = p_100.heartbeats
    #
    # logging.info("Total number of heartbeats: {}\t #N: {}\t #S: {}\t, #V: {}, #F: {}\t #Q: {}"
    #              .format(len(heartbeats), p_100.num_heartbeats('N'), p_100.num_heartbeats('S'), p_100.num_heartbeats('V'),
    #                      p_100.num_heartbeats('F'), p_100.num_heartbeats('Q')))

    # time = list(range(216))
    # for i in range(100):
    #     p = figure(x_axis_label='Sample number (360 Hz)', y_axis_label='Voltage[mV]')
    #     p.line(time, N_b[i], line_width=2, line_color="green")
    #     output_file("N_{}_real.html".format(i))
    #     show(p)
    # plt.plot(N_b[i])
    # plt.show()
