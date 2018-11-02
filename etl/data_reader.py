import logging
import os
import glob2
import json
from collections import Counter

log = logging.getLogger(__name__)


class QR:
    def __init__(self, qrcode):
        self.qrcode = qrcode
        self.timestamps = []

    def add_timestamp(self, ts, jpg_paths, pcd_paths):
        if len(pcd_paths) > 0:
            self.timestamps.append({
                'timestamp': ts,
                'jpg_paths': jpg_paths,
                'pcd_paths': pcd_paths
            })

    def get_matching_measurements(self,
                                  timestamp,
                                  threshold=int(60 * 60 * 24 * 1000)):
        # TODO : only returning one match
        for data in self.timestamps:
            ts = data.get('timestamp')
            difference = abs(int(timestamp) - int(ts))
            if difference < threshold:
                return data
        return None


class DataReader:
    """
    provide methods to read all data
    has understanding of paths etc
    """

    def __init__(self, dataset_path, output_targets):
        self.dataset_path = dataset_path
        self.output_targets = output_targets
        self.qr_storage_dict = {}
        self.person_id_qr_dict = {}
        self.initialize()

    def initialize(self):
        self.process_storage()
        self.process_person_id_qrcde()
        log.info("Total possible qr codes %s" % len(self.person_id_qr_dict))

    def _extract_targets(self, json_data_measure):
        """
        Extracts a list of targets from JSON.
        """

        targets = []
        for output_target in self.output_targets:
            value = json_data_measure[output_target]["value"]
            targets.append(value)
        return targets

    def process_storage(self):
        # loop through storage
        #qrcode_path_1 = os.path.join(self.dataset_path, 'storage', 'person', '[A-Z][A-Z][A-Z]-*')
        #qrcode_path_2 = os.path.join(self.dataset_path, 'storage', 'person', '[A-Z][A-Z][A-Z]_*')
        qrcode_path = os.path.join(self.dataset_path, 'storage', 'person')

        for qr_code in os.listdir(qrcode_path):
            log.info("Processing qr code %s" % qr_code)
            if 'test' in qr_code.lower():
                log.info("Ignoring test qr code %s" % qr_code)
                continue
            # process each timestamp inisde and get all the jpg and pcd paths
            measurement_path = os.path.join(qrcode_path, qr_code,
                                            'measurements')
            if not os.path.exists(measurement_path):
                log.warning(
                    "Ignoring qrcode without measurements path %s" % qr_code)
                continue
            code = QR(qr_code)
            for ts in os.listdir(measurement_path):
                pcd_pattern = os.path.join(measurement_path, ts, "**/*.pcd")
                jpg_pattern = os.path.join(measurement_path, ts, "**/*.jpg")
                pcd_paths = list(glob2.glob(pcd_pattern))
                jpg_paths = list(glob2.glob(jpg_pattern))

                code.add_timestamp(ts, jpg_paths, pcd_paths)

            self.qr_storage_dict[qr_code] = code
            log.info("Processed storage data for qrcode %s" % qr_code)
        log.info("Completed processing storage data")

    def process_person_id_qrcde(self):
        log.info("Process process_person_id_qrcde")
        path = os.path.join(self.dataset_path, 'db', 'persons', '*/*.json')

        for pid in glob2.glob(path):
            try:
                data = json.load(open(pid))
                qrcode = data['qrcode']['value']
                person_id = data['id']['value']
                self.person_id_qr_dict[person_id] = qrcode
            except Exception as e:
                log.exception("Error in loading file %s" % pid)

    def is_measure_manual(self, measure_data):
        measure_type = measure_data["type"]["value"]
        return measure_type == "manual"

    def get_measure_timestamp(self, measure_data):
        ts = measure_data['timestamp']['value']
        return int(ts)

    def get_qr_code(self, measure_data):
        person_id = measure_data["personId"]["value"]
        qrcode = self.person_id_qr_dict.get(person_id)
        return qrcode

    def find_matching_files(self, qrcode, timestamp):
        if qrcode not in self.qr_storage_dict:
            return None
        code = self.qr_storage_dict[qrcode]
        matching_files = code.get_matching_measurements(timestamp)
        return matching_files

    def process_measure_files(self):
        # for each measure file.
        # check if measurement is manual
        # if so, get qr code
        # get timestamp
        # get pcd path and jpg path for the combination:
        # qrcode & matching timestamp
        process_counter = Counter()

        qrcodes_dictionary = {}
        mpath = os.path.join(self.dataset_path, 'db', 'persons',
                             '**/measures/*/**.json')
        measure_files = list(glob2.glob(mpath))

        for mfile in measure_files:
            process_counter['measure_file'] += 1
            log.info("Processing json path measure file %s" % str(mfile))
            json_data_measure = json.load(open(mfile))

            qrcode = self.get_qr_code(json_data_measure)
            if qrcode is None:
                process_counter['ignored_qr_not_found'] += 1
                log.warning(
                    'measure file %s Ignored without matching qr code' % mfile)
                continue

            if not self.is_measure_manual(json_data_measure):
                process_counter['ignored_measure_not_manual'] += 1
                log.warning(
                    "QR code %s measure file %s Ignored with measure != manual "
                    % (qrcode, mfile))
                continue

            measure_timestamp = self.get_measure_timestamp(json_data_measure)
            targets = self._extract_targets(json_data_measure)

            # get matching code by matching timestamp
            matching_files = self.find_matching_files(qrcode,
                                                      measure_timestamp)
            if matching_files is None:
                process_counter['ignored_pc_not_matched'] += 1
                log.warning(
                    "QR code %s measure file %s Ignored Without matching pc files "
                    % (qrcode, mfile))
                continue
            jpg_paths = matching_files['jpg_paths']
            pcd_paths = matching_files['pcd_paths']
            log.info("Adding measurement from file %s with qr code %s" %
                     (mfile, qrcode))

            if qrcode not in qrcodes_dictionary.keys():
                process_counter['added_qr_code'] += 1
                qrcodes_dictionary[qrcode] = []

            qrcodes_dictionary[qrcode].append((targets, jpg_paths, pcd_paths,
                                               measure_timestamp))
            process_counter['added_measure_file'] += 1

        log.info("Total number of qr codes %d " % len(qrcodes_dictionary))
        log.info("Total processing counters %s " % str(process_counter))
        return qrcodes_dictionary
