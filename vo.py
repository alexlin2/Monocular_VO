import cv2
import numpy as np 
import os

color = np.random.randint(0, 255, (100, 3))
class MonocularVO():
    
    def __init__(self, focal, pp, detector, lk_params, pose_file_path, start_id):
        self.focal = focal
        self.pp = pp
        self.detector = detector
        self.lk_params = lk_params
        self.features_img = None
        self.lk_img = None
        self.frame_id = start_id
        self.start_id = start_id
        self.min_features = 0
        self.R = np.zeros(shape=(3, 3), dtype=np.float32)
        self.t = np.zeros(shape=(3, 1), dtype=np.float32)
        with open(pose_file_path) as f:
            self.pose = f.readlines()

        self.get_initial_pose()
        print(self.t)

    def get_initial_pose(self):
        start_pose = self.pose[self.start_id].strip().split()
        x = float(start_pose[3])
        y = float(start_pose[7])
        z = float(start_pose[11])
        self.t[0] = -x
        self.t[1] = -y
        self.t[2] = -z
        self.R[0, :] = start_pose[:3]
        self.R[1, :] = start_pose[4:7]
        self.R[2, :] = start_pose[8:11]

    def detect_features(self, frame):

        kp = self.detector.detect(frame)
        return np.array([x.pt for x in kp], dtype=np.float32).reshape(-1, 1, 2)

    def lk_optical_flow(self, p0, old_frame, current_frame):

        self.p1, st, _ = cv2.calcOpticalFlowPyrLK(old_frame, current_frame, self.p0, None, **self.lk_params)

        self.good_old = self.p0[st == 1].reshape(-1, 1, 2)
        self.good_new = self.p1[st == 1].reshape(-1, 1, 2)

        mask = np.zeros_like(self.lk_img)

        for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i % 100].tolist(), 2)
            lk_img = cv2.circle(self.lk_img, (int(a), int(b)), 5, color[i % 100].tolist(), -1)
        self.lk_img = cv2.add(lk_img, mask)
        
    def get_absolute_scale(self):

        pose = self.pose[self.frame_id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[self.frame_id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])

        true_vect = np.array([[x], [y], [z]])
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])

        self.true_coord = true_vect
        
        return np.linalg.norm(true_vect - prev_vect)

    def process_frame(self, old_frame, current_frame, color_frame):

        self.features_img = color_frame.copy()
        self.lk_img = color_frame.copy()
        
        if self.min_features < 2000:
            self.p0 = self.detect_features(old_frame)

        
        self.lk_optical_flow(self.p0, old_frame, current_frame)

        kp = self.detector.detect(old_frame)
        cv2.drawKeypoints(self.features_img, kp, self.features_img, color=(0,0,255))

        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)

        absolute_scale = self.get_absolute_scale()

  
        _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, focal=self.focal, pp=self.pp)
        
        if (absolute_scale > 0.1 and absolute_scale < 10):
            self.t = self.t + absolute_scale * self.R.dot(t)
            self.R = R.dot(self.R)

        self.p0 = self.good_new.reshape(-1, 1, 2)

    def get_predicted_coords(self):
        diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()

    def get_true_coords(self):
        return self.true_coord.flatten()

