import cv2
import pandas as pd
import time
import os


def init(proto_txt, model_type):
    net = cv2.dnn.readNetFromCaffe(proto_txt, model_type)
    print('cv2 dnn readNetFromCaffe')
    return net


def set_pose_details(dataset_type):
    default_dataset_type = 'default-dataset'

    if dataset_type == 'COCO':
        body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        pose_pairs = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                      ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                      ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                      ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
    elif dataset_type == 'MPI':
        body_parts = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                      "Background": 15}

        pose_pairs = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                      ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                      ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
    else:
        dataset_type = default_dataset_type
        body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5, "LElbow": 6,
                      "LWrist": 7, "MidHip": 8, "RHip": 9, "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13,
                      "LAnkle": 14, "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19, "LSmallToe": 20,
                      "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24, "Background": 25}

        pose_pairs = [["Neck", "MidHip"], ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["MidHip", "RHip"],
                      ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["MidHip", "LHip"], ["LHip", "LKnee"],
                      ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"], ["Nose", "LEye"],
                      ["LEye", "LEar"],
                      ["RShoulder", "REar"], ["LShoulder", "LEar"], ["LAnkle", "LBigToe"], ["LBigToe", "LSmallToe"],
                      ["LAnkle", "LHeel"], ["RAnkle", "RBigToe"], ["RBigToe", "RSmallToe"], ["RAnkle", "RHeel"]]

    dataset_type_and_model = dataset_type + "-caffe_model"
    return dataset_type_and_model, body_parts, pose_pairs


def add_columns_to_dataframe(body_parts, pose_pairs, df):
    pair_cols = []

    for i in range(len(pose_pairs)):
        part_from = pose_pairs[i][0]
        part_to = pose_pairs[i][1]
        assert (part_from in body_parts)
        assert (part_to in body_parts)

        id_from = body_parts[part_from]
        id_to = body_parts[part_to]
        col_name = "P" + str(id_from) + str(id_to)
        pair_cols.append(col_name)

    # add other columns to the dataframe 
    df = pd.DataFrame(columns=df.columns.tolist() + pair_cols)

    return df, pair_cols


def pose_estimate(image_path, net, body_parts, pose_pairs, threshold=0.1, width=368, height=368):
    points = []
    start_t = time.time()

    try:
        # print('image_path - ', image_path)
        img_read = cv2.imread(image_path)
        frame = cv2.rotate(img_read, cv2.cv2.ROTATE_90_CLOCKWISE)
        pts = []
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height),
                                    (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp)
        # do the forward-pass
        out = net.forward()

        print(f"time is {time.time() - start_t}")

        for i in range(len(body_parts)):
            # Slice heatmap of corresponding body part.
            heat_map = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heat_map)
            x = (frame_width * point[0]) / out.shape[3]
            y = (frame_height * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            pts.append((int(x), int(y)) if conf > threshold else None)

        for pair in pose_pairs:
            part_from = pair[0]
            part_to = pair[1]
            assert (part_from in body_parts)
            assert (part_to in body_parts)

            id_from = body_parts[part_from]
            id_to = body_parts[part_to]
            if pts[id_from] and pts[id_to]:
                # so we keep the points (belonging to pairs together)
                curr_set_of_points = [pts[id_from], pts[id_to]]

                # NOTE: uncomment for DEBUG purposes
                # print('curr_set_of_points for ', id_from, id_to)
                # print('curr_set_of_points ', curr_set_of_points)

                points.append(curr_set_of_points)
                # print('points ', points)

                # the last pairs, are
                # - right ankle to right big toe 
                # - left ankle to left big toe
            else:
                # append [] if no pose points estimated
                points.append([])

    except Exception:
        filename = 'outputs/not_processed.txt'

        if os.path.exists(filename):
            append_write = 'a'
        else:
            append_write = 'w'

        exp_write = open(filename, append_write)
        exp_write.write(f"File: {image_path}\n")
        exp_write.close()
    return points
