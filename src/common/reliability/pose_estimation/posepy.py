import cv2
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display


def pose_estimation_from_model_and_dataset(net, body_parts, pose_pairs, dataset_type_and_model,
                                           image_path, image_filename, get_title_box_and_resize_frame,
                                           threshold=0.1, width=368, height=368, write_image=False):
    img_read = cv2.imread(image_path)
    frame = cv2.rotate(img_read, cv2.cv2.ROTATE_90_CLOCKWISE)
    # print('image_filename: ', image_filename)

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height),
                                (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    start_t = time.time()
    out = net.forward()

    print(f"time is {time.time() - start_t}")
    # assert(len(body_parts) == out.shape[1])

    points = []
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
        points.append((int(x), int(y)) if conf > threshold else None)

    for pair in pose_pairs:
        part_from = pair[0]
        part_to = pair[1]
        assert (part_from in body_parts)
        assert (part_to in body_parts)

        id_from = body_parts[part_from]
        id_to = body_parts[part_to]
        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (255, 74, 0), 3)
            cv2.ellipse(frame, points[id_from], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(frame, points[id_to], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, str(id_from), points[id_from], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, str(id_to), points[id_to], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                        cv2.LINE_AA)

    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # set up title box to show some description etc. of the result
    # and also modify the outer edge dims of the frame
    frame_w_border, title_box = get_title_box_and_resize_frame(frame, image_filename,
                                                               additionalTitleText=dataset_type_and_model)
    # use below for combining the title_box and frame_w_border
    im_frame = cv2.vconcat((title_box, frame_w_border))
    # set Pose Estimation file name
    im_path = f"PoseEst-{image_filename}"
    if write_image:
        # create folder if need be
        if not (os.path.exists('output')):
            os.makedirs('output', mode=0o777, exist_ok=False)
        # write the file
        cv2.imwrite(f"./output/{im_path}", im_frame)

    # display image
    plt.figure()
    # Converting BGR to RGB
    img = cv2.cvtColor(im_frame, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(img))

    return points
