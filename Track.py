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
        track = TrackingElem(track_pos, self.trackIdCount, self.dt)
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
                diff = self.tracks[i].pos - detects[j]
                cost_matrix[i][j] = np.sqrt(diff[0][0] ** 2 + diff[1][0] ** 2)

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
                vel_x = (detects[i][0] - old_pos[i][0]) / self.dt
                vel_y = (detects[i][1] - old_pos[i][1]) / self.dt

                obs = np.asarray([detects[i][0],vel_x,detects[i][1],vel_y])
                self.tracks[i].pos = self.tracks[i].KF.correct(obs, isFirst = False)
            else:
                obs = np.zeros((4,1))
                self.tracks[i].pos = self.tracks[i].KF.correct(obs, isFirst = True)

            if len(self.tracks[i].trace) > self.max_trace_to_store:
                for j in range(len(self.tracks[i].trace) - self.max_trace_to_store):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].pos)

        # del tracks exceeding max allowed frames
        for track_ind in self.tracks:
            if self.tracks[track_ind].frames_skipped > self.max_frames_can_skip:
                del self.tracks[track_ind]
