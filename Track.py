from Kalman_Filter import Kalman_Filter
from Hungarian import get_optim_assignment
import numpy as np

class TrackingElem(object):
    def __init__(self, pos, id, dt):
        self.id = id
        self.KF = Kalman_Filter(dt)
        self.frames_skipped = 0
        self.trace = []
        self.pos = pos

class TrackingList(object):
    def __init__(self, distance_thresh, max_frames_can_skip, max_trace_to_store, dt = 0.1):
        self.distance_thresh = distance_thresh
        self.max_frames_can_skip = max_frames_can_skip
        self.max_trace_to_store = max_trace_to_store
        self.tracks = []
        self.trackIdCount = 0
        self.dt = dt

    def add_new_track(self, track_pos):
        pos = np.array([[track_pos[0]], [0.0], [track_pos[1]], [0.0]])
        track = TrackingElem(pos, self.trackIdCount, self.dt)
        self.trackIdCount += 1
        self.tracks.append(track)

    def update(self, detects):
        if (len(self.tracks) == 0):
            for i in range(len(detects)):
                self.add_new_track(detects[i])

        tracks_no = len(self.tracks)
        detects_no = len(detects)
        cost_matrix = np.zeros((tracks_no, detects_no))
        for i in range(tracks_no):
            for j in range(detects_no):
                diff_x = self.tracks[i].pos[0] - detects[j][0]
                diff_y = self.tracks[i].pos[2] - detects[j][1]
                cost_matrix[i][j] = np.sqrt(diff_x ** 2 + diff_y ** 2)
                # print(cost_matrix[i][j])
        row_inds, col_inds = get_optim_assignment(cost_matrix)
        # assign assigned rows
        assigns = [-1 for i in range(tracks_no)]
        for i in range(len(row_inds)):
            assigns[row_inds[i]] = col_inds[i]

        # filter tracks
        for row_ind,col_ind in enumerate(assigns):
            if col_ind == -1:
                self.tracks[row_ind].frames_skipped += 1
            elif cost_matrix[row_ind][col_ind] <= self.distance_thresh:
                self.tracks[row_ind].frames_skipped = 0
            else:
                assigns[row_ind] = -1

        # create new tracks
        for i in range(detects_no):
            if i not in assigns:
                self.add_new_track(detects[i])

        # utilize kalman filter
        for i in range(len(assigns)):
            old_pos = self.tracks[i].KF.x.copy()
            pred_pos = self.tracks[i].KF.predict()
            if assigns[i] != -1:
                detect_id = assigns[i]
                # print('------------look---------')
                # if len(self.tracks[i].trace) == 0:
                #     vel_x = 0
                #     vel_y = 0
                # else:
                    # print(detects[detect_id][0])
                    # print(old_pos[0])
                    # print(detects[detect_id][1])
                    # print(old_pos[2])
                    # vel_x = (detects[detect_id][0] - old_pos[0]) / self.dt
                    # vel_y = (detects[detect_id][1] - old_pos[2]) / self.dt
                # print(detects[detect_id])
                obs = np.asarray([[detects[detect_id][0]],[0],[detects[detect_id][1]],[0]])
                self.tracks[i].pos = self.tracks[i].KF.correct(obs, flag=True)
            else:
                obs = np.zeros((4,1))
                self.tracks[i].pos = self.tracks[i].KF.correct(obs, flag=False)

            if len(self.tracks[i].trace) > self.max_trace_to_store:
                for j in range(len(self.tracks[i].trace) - self.max_trace_to_store):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].pos)

        self.tracks = [x for x in self.tracks if x.frames_skipped <= self.max_frames_can_skip]

        # print("--------------start---------------")

        # for i in range(len(self.tracks)):
        #     print(self.tracks[i].trace)
        #     print("")

        # print("--------------end---------------")