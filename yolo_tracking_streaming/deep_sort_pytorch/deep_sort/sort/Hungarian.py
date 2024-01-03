import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

def Hungarian_(cost_matrix:np.ndarray):
	row_indices, col_indices = linear_assignment(cost_matrix)
	return row_indices, col_indices
    

if __name__ == '__main__':
    arr = [
        [7, 6, 2, 9, 2],
        [6, 2, 1, 3, 9],
        [5, 6, 8, 9, 5],
        [6, 8, 5, 8, 6],
        [9, 5, 6, 4, 7],
    ]

    arr = np.array(arr)
    Hungarian_(arr.copy())
	
	# total = 0
	# for row, column in indexes:
	# 	value = cost_matrix[row][column]
	# 	total += value
	# 	print(f'({row}, {column}) -> {value}')
			
	# print(f'total cost: {total}')
    