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
        print(repr(self.cost_matrix))
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
# cost_matrix = [[225.35860312,222.4707846, 213.39283025],
#  [ 30.08321791,  5.85234996, 10.        ],
#  [ 50.14229751, 25.93260496, 19.47434209],
#  [ 60.01666435, 35.85038354, 26.40075756],
#  [ 32.00390601, 10.12422837,  3.04138127]]
cost_matrix = [[118.80340904, 102.20812101,  93.15041599,  78.20805585 , 58.83026432,
   79.30952024, 111.93078218,   8.60232527,   1.58113883 ,  7.        ],
 [ 53.5023364 ,  36.22499137,  26.856098  ,  11.88486432 ,  7.63216876,
   16.97792685,  52.53808143,  70.22997935,  66.11542936 , 70.4006392 ],
 [104.20292702,  87.09190548,  77.73351915,  62.76941931 , 43.36473221,
   65.36436338,  99.72462083,  19.55760722,  17.20465053 , 24.42334948],
 [109.74629834,  92.85741758,  83.63013811,  68.64765109 , 49.1934955,
   70.51950085, 104.1561328 ,  13.60147051,   9.92471662 , 17.08800749],
 [ 17.39252713,   7.56637298,  13.82931669,  28.57008925 , 47.94006675,
   30.30264015,  32.33032632, 110.46379497, 106.42485612 ,110.52714599],
 [ 23.54782368,   6.02079729,   4.03112887,  19.00657781 , 38.46101923,
   20.52437575,  31.5634282 , 101.10019782,  96.88782173 ,100.82286447],
 [ 33.36540124,  18.50675552,  13.15294644,  13.20984481 , 29.06888371,
    6.32455532,  30.00833218,  90.44335244,  85.51315688 , 88.54942123],
 [ 29.34706118,  12.08304597,   3.80788655,  12.36931688 , 31.6622804,
   13.20984481,  32.64965543,  94.27884174,  89.89438247 , 93.65094767],
 [ 11.40175425,   6.57647322,  16.00781059,  30.31913587 , 49.48989796,
   28.57008925,  22.20923231, 111.97432741, 107.38365797 ,110.77567422],
 [ 29.15475947,  33.90058997,  39.37321425,  47.46841055 , 61.89709202,
   39.42397748,   7.43303437, 119.01365468, 113.10724999 ,114.3558044 ],
 [ 43.50287347,  33.72313746,  31.004032  ,  28.21790212 , 35.33058165,
   18.11767093,  29.75315109,  89.06317982,  83.10385069 , 84.41711912],
 [ 45.54393483,  47.67074575,  51.03920062,  55.37598758 , 65.7951366,
   45.80392996,  23.54782368, 117.00427343, 110.52827692 ,110.54863183]]
# cost_matrix = [[  7,  53, 183, 439, 863],
#              [497, 383, 563,  79, 973],
#              [287,  63, 343, 169, 583],
#              [627, 343, 773, 959, 943],
#              [767, 473, 103, 699, 303]]
# print(cost_matrix.T)
print(get_optim_assignment(cost_matrix))
