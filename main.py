import cv2
import numpy as np 
import os
from vo import MonocularVO

dataset_img_path = '/home/alexlin/Developer/Monocular_VO/dataset/images/00/image_0/'
dataset_pose_path = '/home/alexlin/Developer/Monocular_VO/dataset/poses/00.txt'
dataset_color_path = '/home/alexlin/Developer/Monocular_VO/dataset/color/00/image_2/'

focal = 718.8560
pp = (607.1928, 185.2157)

lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

fast_detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

vo = MonocularVO(focal, pp, fast_detector, lk_params, dataset_pose_path, 100)

traj = np.zeros(shape=(1200, 1200, 3))

MSE_sum = 0.0

while(vo.frame_id < len(os.listdir(dataset_img_path))):
    old_frame = cv2.imread(dataset_img_path + str(vo.frame_id).zfill(6)+'.png', 0)
    new_frame = cv2.imread(dataset_img_path + str(vo.frame_id+1).zfill(6)+'.png', 0)
    color_frame = cv2.imread(dataset_color_path + str(vo.frame_id).zfill(6)+'.png')

    if(old_frame is not None and new_frame is not None):
        cv2.imshow('frame', color_frame)
        vo.process_frame(old_frame, new_frame, color_frame)

        cv2.imshow('lk_frame', vo.lk_img)
        cv2.imshow('kp_frame', vo.features_img)

        coord = vo.get_predicted_coords()
        true_coord = vo.get_true_coords()

        MSE_sum += np.linalg.norm(coord - true_coord)
        print("MSE Error: ", np.linalg.norm(coord - true_coord))
        print("Average MSE Error: ", MSE_sum / (vo.frame_id - vo.start_id))

        draw_x, draw_y, draw_z = [int(round(x)) for x in coord]
        true_x, true_y, true_z = [int(round(x)) for x in true_coord]
        traj = cv2.circle(traj, (true_x + 500, true_z + 500), 1, list((0, 0, 255)), 4)
        traj = cv2.circle(traj, (draw_x + 500, draw_z + 500), 1, list((0, 255, 0)), 4)
        
        cv2.imshow('trajectory', traj)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    vo.frame_id += 1

cv2.destroyAllWindows()