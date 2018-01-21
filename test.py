import numpy as np

# cost_matrix = [[1,0,0,0],[0,0,1,1],[0,1,1,1],[0,1,1,1]]
cost_matrix = [[1,1,0],[0,0,0],[0,0,0]]
cost_matrix = np.asarray(cost_matrix)

row_no = 3
col_no = 3

row_marked = np.zeros(3, dtype=bool)
col_marked = np.zeros(3, dtype=bool)
assignments = np.zeros((3,3),dtype=int)

def new():
    crossed = cost_matrix.copy()
    # while True:
    #     for i in range(row_no):
    #         cost_row = crossed[i]
    #         if np.count_nonzero(cost_row==0) == 1:
    #             j = np.where(cost_row == 0)[0][0]
    #             assignments[i][j] = 1
    #             crossed[i,:] = 1
    #             crossed[:,j] = 1
    #
    #     for j in range(col_no):
    #         cost_col = crossed[:,j]
    #         if np.count_nonzero(cost_col==0) == 1:
    #             i = np.where(cost_col == 0)[0][0]
    #             assignments[i][j] = 1
    #             crossed[i, :] = 1
    #             crossed[:, j] = 1
    #
    #     if np.count_nonzero(crossed==0) == 0:
    #         break

    prev_count = np.count_nonzero(crossed == 0)
    while True:
        for i in range(row_no):
            cost_row = crossed[i]
            if np.count_nonzero(cost_row == 0) == 1:
                j = np.where(cost_row == 0)[0][0]
                assignments[i][j] = 1
                crossed[i, :] = 1
                crossed[:, j] = 1

        for j in range(col_no):
            cost_col = crossed[:, j]
            if np.count_nonzero(cost_col == 0) == 1:
                i = np.where(cost_col == 0)[0][0]
                assignments[i][j] = 1
                crossed[i, :] = 1
                crossed[:, j] = 1

        if np.count_nonzero(crossed == 0) == 0:
            break
        elif prev_count == np.count_nonzero(crossed == 0):
            indices = np.where(crossed == 0)
            i = indices[0][0]
            j = indices[1][0]
            assignments[i][j] = 1
            crossed[i, :] = 1
            crossed[:, j] = 1

        prev_count = np.count_nonzero(crossed == 0)

def old():
    for i, j in zip(*np.where(cost_matrix == 0)):
        if not row_marked[i] and not col_marked[j]:
            assignments[i, j] = 1
            row_marked[i] = True
            col_marked[j] = True

new()
# step_5()
print(assignments)