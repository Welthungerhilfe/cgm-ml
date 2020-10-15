import numpy as np


def get_timestamps_from_rgb(rgb_paths):
    #connector1 = dbutils.connect_to_main_database()

    # get all artifacts of a certain unique qr code
#     sql_statement  = "SELECT '{}'".format(mount_path + '/') + " || storage_path, dataformat, session_timestamp FROM artifact "
#     sql_statement += " WHERE qr_code = '{}'".format(qr_code)
#     sql_statement += " AND dataformat = 'rgb'"

#     all_rgb = connector1.execute(sql_statement, fetch_all=True)
    
    
    
    #timestamps = [x[2] for x in all_rgb]
    path       = [x for x in rgb_paths]
    
    timestamps = []

    for p in path:
        
        filename = p.split('/')[-1]
        value = filename.split('_')[-1]
        timestamp_value = value.replace('.jpg', '')
        timestamps.append(float(timestamp_value))

    
    if( len(timestamps) == 0): 
        error = np.array([])
        return [error, path]
    
    timestamps      = np.asarray(timestamps)
    return [timestamps, path]


def get_timestamp_from_pcd(pcd_path): 
    filename  = str(pcd_path)
    #print("mount_path",mount_path)
    #filename=str(mount_path)+"/"+filename_
    #print("filenaaaaaame",filename)
    infile    = open(filename, 'r')
    firstLine = infile.readline()

    # get the time from the header of the pcd file
    import re
    timestamp = re.findall("\d+\.\d+", firstLine)

    # check if a timestamp is parsed from the header of the pcd file
    try: 
        return_timestamp = float(timestamp[0])
    except IndexError:
        return_timestamp = []

    return return_timestamp  # index error? IndexError

def get_timestamps_from_pcd(pcd_paths): 
#     connector2 = dbutils.connect_to_main_database()
    
#     sql_statement  = "SELECT '{}'".format(mount_path + '/') + " || storage_path FROM artifact "
#     sql_statement += " WHERE qr_code = '{}'".format(unique_qr_codes[0])
#     sql_statement += " AND dataformat = 'pcd'"

#     sql_statement  = "SELECT storage_path FROM artifact "
#     sql_statement += " WHERE qr_code = '{}'".format(qr_code)
#     sql_statement += " AND dataformat = 'pcd'" 
    #path = connector2.execute(sql_statement, fetch_all=True)
    timestamps = np.array([])
    path       = [x for x in pcd_paths]

    #iterate over all paths pointing to pcds
    for p in path: 
        
        try: 
            stamp = get_timestamp_from_pcd(p)
            timestamps = np.append(timestamps, stamp)
        except IndexError: 
            error = np.array([])
            logging.error("Error with timestamp in pcd")
            return [error, p]

    
    #print("path",path)

    if( len(timestamps) == 0): 
        error = np.array([])
        return [error, path]
    
    return [timestamps, path]




