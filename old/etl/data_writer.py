import os
import numpy as np
import h5py
import logging
import csv
import shutil


log = logging.getLogger(__name__)


class DataWriter:
    
    def __init__(self, config, run_id, simulate=True):
        base_dir = config.get('output', 'base_dir')
        self.base_dir = base_dir
        self.run_dir = os.path.join(base_dir, run_id)
        self.run_id = run_id
        self.simulate = simulate
        self.initialize()

        
    def initialize(self):
        # create directory
        self.makedirs_if_not_exists(self.run_dir)

        
    def write(self, qrcode, y_output, timestamp, pcd_paths, jpg_paths):
        # qr code is the name of the file
        # xinput is ndarray
        # output is the target values
        qrcode_dir = os.path.join(self.run_dir, qrcode)
        self.makedirs_if_not_exists(qrcode_dir)
        subdir = os.path.join(qrcode_dir, str(timestamp))
        self.makedirs_if_not_exists(subdir)
        
        # target filename
        targetfilename = os.path.join(subdir, 'target.txt')
        self.write_target(y_output, targetfilename)

        # copy over the pcd paths
        pcd_dir = os.path.join(subdir, 'pcd')
        self.makedirs_if_not_exists(pcd_dir)
        log.debug("Copying pcd files for qrcode %s" % qrcode)
        for pcd_path in pcd_paths:
            fname = os.path.basename(pcd_path)
            dst = os.path.join(pcd_dir, fname)
            self.copyfile(pcd_path, dst)

        # copy over the pcd paths
        jpg_dir = os.path.join(subdir, 'jpg')
        self.makedirs_if_not_exists(jpg_dir)
        log.debug("Copying jpg files for qrcode %s" % qrcode)
        for jpg_path in jpg_paths:
            fname = os.path.basename(jpg_path)
            dst = os.path.join(jpg_dir, fname)
            self.copyfile(jpg_path, dst)

            
    def wrapup(self):
        if self.simulate == False:
            # write the readme file
            # zip and create simlink
            log.info("Going to zip the directory %s" % self.run_dir)
            zipfile = os.path.join(self.base_dir, self.run_id)
            shutil.make_archive(zipfile, 'zip', self.run_dir)

            # check existing simlink
            latestfilename = os.path.join(self.base_dir, 'latest.zip')
            if os.path.exists(latestfilename):
                os.unlink(latestfilename)

            # create a simlink
            simlinkfile = "%s.zip" % zipfile
            os.symlink(simlinkfile, latestfilename)
        else:
            print("Simulating wrapup")
            
    def makedirs_if_not_exists(self, path):
        if self.simulate == False:
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            print("Simulating makedirs " + path)


    def write_target(self, target, path):
        if self.simulate == False:
            with open(path, "w") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(target)
        else:
            print("Simulating write target" + path)

    def copyfile(self, source, destination):
        if self.simulate == False:
            shutil.copyfile(source, destination)
        else:
            print("Simulating copy from", source, "to", destination)
