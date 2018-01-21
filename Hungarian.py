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

    def clear_cover(self):
        self.row_marked[:] = False
        self.col_marked[:] = False

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
        for i, j in zip(*np.where(self.cost_matrix == 0)):
            if not self.row_marked[i] and not self.col_marked[j]:
                self.assignments[i, j] = 1
                self.row_marked[i] = True
                self.col_marked[j] = True

        self.clear_cover()

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

# cost_matrix = np.asarray([[10, 19, 8, 15, 19],[10, 18, 7, 17, 19],[13, 16, 9, 14, 19],[12, 19, 8, 18, 19],[14, 17, 10, 19, 19]])
# cost_matrix = np.asarray([[236.53012493,234.64121548,224.94554897]])
cost_matrix = [[225.35860312,222.4707846, 213.39283025],
 [ 30.08321791,  5.85234996, 10.        ],
 [ 50.14229751, 25.93260496, 19.47434209],
 [ 60.01666435, 35.85038354, 26.40075756],
 [ 32.00390601, 10.12422837,  3.04138127]]
# cost_matrix = [[  7,  53, 183, 439, 863],
#              [497, 383, 563,  79, 973],
#              [287,  63, 343, 169, 583],
#              [627, 343, 773, 959, 943],
#              [767, 473, 103, 699, 303]]
# print(cost_matrix.T)
print(get_optim_assignment(cost_matrix))

# import hungarian_new
#
# hungarian_new.minimize(cost_matrix)
