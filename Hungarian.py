import numpy as np

def get_optim_assignment(matrix):

    matrix = np.asarray(matrix)
    if matrix.shape[1] < matrix.shape[0]:
        matrix = matrix.T
        transposed = True
    else:
        transposed = False

    hungarian = Hungarian(matrix)
    assignments = hungarian.calculate()

    if transposed:
        assignments = assignments.T

    return np.where(assignments == 1)

class Hungarian(object):
    def __init__(self, matrix):
        self.ori_matrix = matrix.copy()
        self.cost_matrix = np.pad(matrix, [(0, matrix.shape[1] - matrix.shape[0]), (0, 0)], mode='constant')
        n,m = self.cost_matrix.shape
        self.assignments = np.zeros((n,m),dtype=int)
        self.row_marked = np.zeros(n, dtype=bool)
        self.col_marked = np.zeros(m, dtype=bool)
        self.row_no = n
        self.col_no = m

    def calculate(self):
        self.step_1()
        assignments = self.assignments[0:self.ori_matrix.shape[0],:]
        return assignments

    def get_total_cost(self):
        total_cost = 0
        assigns = np.where(self.assignments == 1)

        for i in range(assigns[0].shape[0]):
            total_cost += self.ori_matrix[i][assigns[1][i]]
        return total_cost

    def reset_assignments(self):
        self.row_marked[:] = False
        self.col_marked[:] = False
        self.assignments[:,:] = 0

    def step_1(self):
        self.cost_matrix -= self.cost_matrix.min(axis=1)[:, np.newaxis]

        if self.cost_matrix.shape[0] == 1:
            self.step_3()
        else:
            self.step_2()

    def step_2(self):
        self.cost_matrix -= self.cost_matrix.min(axis=0)[np.newaxis, :]
        self.step_3()

    def step_3(self):
        self.reset_assignments()

        # 1. Assign as many tasks as possible
        crossed = self.cost_matrix.copy()
        prev_count = np.count_nonzero(crossed == 0)
        while True:
            for i in range(self.row_no):
                cost_row = crossed[i]
                if np.count_nonzero(cost_row == 0) == 1:
                    j = np.where(cost_row == 0)[0][0]
                    self.assignments[i][j] = 1
                    crossed[i, :] = 1
                    crossed[:, j] = 1

            for j in range(self.col_no):
                cost_col = crossed[:, j]
                if np.count_nonzero(cost_col == 0) == 1:
                    i = np.where(cost_col == 0)[0][0]
                    self.assignments[i][j] = 1
                    crossed[i, :] = 1
                    crossed[:, j] = 1

            if np.count_nonzero(crossed == 0) == 0:
                break
            elif prev_count == np.count_nonzero(crossed == 0):
                indices = np.where(crossed == 0)
                i = indices[0][0]
                j = indices[1][0]
                self.assignments[i][j] = 1
                crossed[i, :] = 1
                crossed[:, j] = 1

            prev_count = np.count_nonzero(crossed == 0)

        # 2. Draw as few lines as possible
        new_row_marked = np.all(self.assignments==0,axis=1)
        self.row_marked = new_row_marked.copy()

        while True:
            new_col_marked = np.any(self.cost_matrix[new_row_marked,:]==0, axis=0)
            new_col_marked[self.col_marked == True] = False
            new_row_marked = np.any(self.assignments[:, new_col_marked] == 1, axis=1)
            self.col_marked = np.logical_or(self.col_marked, new_col_marked)
            self.row_marked = np.logical_or(self.row_marked, new_row_marked)

            if np.all(new_col_marked==False):
                break

        self.row_marked = ~self.row_marked

        no_lines = np.count_nonzero(self.col_marked) + np.count_nonzero(self.row_marked)

        if no_lines < self.cost_matrix.shape[0]:
            self.step_4()

        else:
            self.step_5()

    def step_4(self):
        temp_cost_matrix = self.cost_matrix[~self.row_marked, :]
        min_val_left = temp_cost_matrix[:, ~self.col_marked].min()
        self.cost_matrix[~self.row_marked,:] -= min_val_left
        self.cost_matrix[:,self.col_marked] += min_val_left

        self.step_3()

    def step_5(self):
        self.reset_assignments()
        self._step_5(0)

    def _step_5(self, row):
        if row == self.row_no:
            return True

        for col in range(self.col_no):
            if self.cost_matrix[row, col] == 0 and not self.col_marked[col]:
                self.assignments[row][col] = 1
                self.col_marked[col] = True
                if (self._step_5(row+1)):
                    return True
                self.assignments[row][col] = 0
                self.col_marked[col] = False
        return False
