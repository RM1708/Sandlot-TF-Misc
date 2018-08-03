# -*- coding: utf-8 -*-
"""
Created on 12Jul2018

@author: RM

The snippet of code below is from the function images_to_sprite() in the file 
embedder.py. 

The experiment here is to figure out the transformations done on the array 

    [
        [
            [    <--3--->  D[0:3]
                [1, 2, 3],    ^
                [4, 5, 6],    3  C[0:3]    ^
                [7, 8, 9]     |            | 
            ],                V            |
            [                              |
                [10, 20, 30],              |
                [40, 50, 60],              3  B[0:3]    ^
                [70, 80, 90]               |            |
            ],                             |            |
            [                              |            |
                [100, 200, 300],           |            |
                [400, 500, 600],           V            |
                [700, 800, 900]
            ]                                           |
        ], -------------------------------------------  |
        [
            [
                [110, 120, 130],
                [140, 150, 160],                        |
                [170, 180, 190]                         |
            ],
            [                                           |
                [210, 220, 230],                        
                [240, 250, 260],                        3 A[0:3]
                [270, 280, 290]
            ],                                          |
            [
                [310, 320, 330],                        |
                [340, 350, 360],                        |
                [370, 380, 390]
            ]
        ], -------------------------------------------- |
        [
            [
                [1100, 1200, 1300],                     |
                [1400, 1500, 1600],                     |
                [1700, 1800, 1900]
            ],
            [
                [2100, 2200, 2300],                     |
                [2400, 2500, 2600],                     V
                [2700, 2800, 2900]
            ],
            [
                [3100, 3200, 3300],
                [3400, 3500, 3600],
                [3700, 3800, 3900]
            ]
        ]
    ]

ipdb> data[:, 0, 0, 0].astype(int)
array([   1,  110, 1100]) #A0B0C0D0, A1B0C0D0, A2B0C0D0

ipdb> data_transpose[0,0,0,:].astype(int)
array([   1,  110, 1100])

ipdb> data[:, :, 0, 0].astype(int)
array([[   1,   10,  100],  #A0B0C0D0, A0B1C0D0, A0B2C0D0
       [ 110,  210,  310],  #A1B0C0D0, A1B1C0D0, A1B2C0D0
       [1100, 2100, 3100]]) #A2B0C0D0, A2B1C0D0, A2B2C0D0

#The next is a transpose of the one prior
ipdb> data_transpose[:,0,0,:].astype(int)
array([[   1,  110, 1100],
       [  10,  210, 2100],
       [ 100,  310, 3100]])

ipdb> data[:, :, :, 0].astype(int)
array([[[   1,    4,    7], A0B0C0D0, A0B0C1D0, A0B0C2D0
        [  10,   40,   70], A0B1C0D0, A0B1C1D0, A0B1C2D0
        [ 100,  400,  700]],A0B2C0D0, A0B2C1D0, A0B2C2D0

       [[ 110,  140,  170], A1B0C0D0, A1B0C1D0, A1B0C2D0
        [ 210,  240,  270], A1B1C0D0, A1B1C1D0, A1B1C2D0
        [ 310,  340,  370]],A1B2C0D0, A1B2C1D0, A1B2C2D0

       [[1100, 1400, 1700],  A2B0C0D0, A2B0C1D0, A2B0C2D0
        [2100, 2400, 2700],  A2B1C0D0, A2B1C1D0, A2B1C2D0
        [3100, 3400, 3700]]])A2B2C0D0, A2B2C1D0, A2B2C2D0

ipdb> data[0, :, :, :].astype(int)
array([[[  1,   2,   3], A0B0C0D0, A0B0C0D1, A0B0C0D2
        [  4,   5,   6], .....
        [  7,   8,   9]],A0B0C2D0, A0B0C2D1, A0B0C2D2

       [[ 10,  20,  30], A0B1C0D0, A0B1C0D1, A0B1C0D2
        [ 40,  50,  60], ...
        [ 70,  80,  90]],A0B1C2D0, A0B1C2D1, A0B1C2D2

       [[100, 200, 300],   A0B2C0D0, A0B2C0D1, A0B2C0D2
        [400, 500, 600],
        [700, 800, 900]]]) A0B2C2D0, A0B2C2D1, A0B2C2D2
        
ipdb> data_transpose[0,:,:,:].astype(int)
        1.The outer-most dimension is the slowest changing. For data_transpose
        matrix, it is identical to the dimension B of data matrix. 
        It is held at index 0. As can be seen below B0 is present in all 
        27 element.
        2. The inner-most dimension is the fastest changing dimension.
        For the data_transpose matrix it identical to Dimension A of the 
        data matrix. A0, A1, A2 are present, in order, in all rows.
        3. In the data_transpose matrix the 2nd fastest changing dimension
        is identical to dimension D of data matrix. So it should remain 
        constant for a given row (unlike the fastest changing dimension) and 
        change along rows taking the values D[0:3]. That is true for each of 
        the 3x3 matrices.
        4. Since the outer-most dimension of data_transpose matrix is held 
        constant, the actual slowest-changing dimension of data_transpose matrix
        is identical to dimension C of the data matrix. It will thus be constant
        over a 3x3 matrix. This is so below
array([[[   1,  110, 1100], A0B0C0D0, A1B0C0D0, A2B0C0D0,
        [   2,  120, 1200], A0B0C0D1, A1B0C0D1, A2B0C0D1,
        [   3,  130, 1300]],A0B0C0D2, A1B0C0D2, A2B0C0D2

       [[   4,  140, 1400], A0B0C1D0, A1B0C1D0, A2B0C1D0
        [   5,  150, 1500], A0B0C1D1, A1B0C1D1, A2B0C1D1
        [   6,  160, 1600]],A0B0C1D2, A1B0C1D2, A2B0C1D2

       [[   7,  170, 1700],   A0B0C2D0, A1B0C2D0, A2B0C2D0,
        [   8,  180, 1800],   A0B0C2D1, A1B0C2D1, A2B0C2D1
        [   9,  190, 1900]]]) A0B0C2D2, A1B0C2D2, A2B0C2D2

ipdb> data_transpose[0,0,:,:].astype(int)
array([[   1,  110, 1100],
       [   2,  120, 1200],
       [   3,  130, 1300]])
        
TREE REPRESENTATION OF 3 x 3 x 3 x 3 SPACE

M--------A0-----B0----C0-----D0
     |       |      |     |--D1
     |       |      |     |--D2
     |       |      |-C1-----D0
     |       |      |     |--D1
     |       |      |     |--D2
     |       |      |-C2-----D0
     |       |            |--D1
     |       |             |--D2
     |       ---B1----C0-----D0
     |       |      |     |--D1
     |       |      |     |--D2
     |       |      |-C1-----D0
     |       |      |     |--D1
     |       |      |     |--D2
     |       |      |-C2-----D0
     |       |            |--D1
     |       |            |--D2
     |       ---B2----C0-----D0
     |              |     |--D1
     |              |     |--D2
     |              |-C1-----D0
     |              |     |--D1
     |              |     |--D2
     |              |-C2-----D0
     |                    |--D1
     |                    |--D2
     |---A1-----B0----C0-----D0
     |       |      |     |--D1
     |       |      |     |--D2
     |       |      |-C1-----D0
     |       |      |     |--D1
     |       |      |     |--D2
     |       |      |-C2-----D0
     |       |            |--D1
     |       |             |--D2
     |       ---B1----C0-----D0
     |       |      |     |--D1
     |       |      |     |--D2
     |       |      |-C1-----D0
     |       |      |     |--D1
     |       |      |     |--D2
     |       |      |-C2-----D0
     |       |            |--D1
     |       |            |--D2
     |       ---B2----C0-----D0
     |              |     |--D1
     |              |     |--D2
     |              |-C1-----D0
     |              |     |--D1
     |              |     |--D2
     |              |-C2-----D0
     |                    |--D1
     |                    |--D2
     |----A2-----B0----C0-----D0
             |      |     |--D1
             |      |     |--D2
             |      |-C1-----D0
             |      |     |--D1
             |      |     |--D2
             |      |-C2-----D0
             |            |--D1
             |             |--D2
             ---B1----C0-----D0
             |      |     |--D1
             |      |     |--D2
             |      |-C1-----D0
             |      |     |--D1
             |      |     |--D2
             |      |-C2-----D0
             |            |--D1
             |            |--D2
             ---B2----C0-----D0
                    |     |--D1
                    |     |--D2
                    |-C1-----D0
                    |     |--D1
                    |     |--D2
                    |-C2-----D0
                          |--D1
                          |--D2

"""

import numpy as np
#import tensorflow as tf


data = \
    np.asarray(
                    [
                        [
                            [
                                [1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]
                            ],
                            [
                                [10, 20, 30],
                                [40, 50, 60],
                                [70, 80, 90]
                            ],
                            [
                                [100, 200, 300],
                                [400, 500, 600],
                                [700, 800, 900]
                            ]
                        ],
                        [
                            [
                                [110, 120, 130],
                                [140, 150, 160],
                                [170, 180, 190]
                            ],
                            [
                                [210, 220, 230],
                                [240, 250, 260],
                                [270, 280, 290]
                            ],
                            [
                                [310, 320, 330],
                                [340, 350, 360],
                                [370, 380, 390]
                            ]
                        ],
                        [
                            [
                                [1100, 1200, 1300],
                                [1400, 1500, 1600],
                                [1700, 1800, 1900]
                            ],
                            [
                                [2100, 2200, 2300],
                                [2400, 2500, 2600],
                                [2700, 2800, 2900]
                            ],
                            [
                                [3100, 3200, 3300],
                                [3400, 3500, 3600],
                                [3700, 3800, 3900]
                            ]
                        ]
                    ]
                )

data = data.astype(np.float32)
assert((3, 3, 3, 3) == data.shape)
data_2D = data.reshape((data.shape[0], -1))
assert((3, 27) == data_2D.shape)
min_data = np.min(data_2D, axis=1)
assert((data[0, 0, 0, 0],\
        data[1][0][0][0],\
        data[2][0][0][0]) == (min_data[0], \
                                min_data[1], \
                                min_data[2]))

data_transpose = data.transpose(1, 2, 3, 0)
data_shifted = data_transpose - min_data
data_shifted_transposed = (data_shifted).transpose(3, 0, 1, 2)
shape = (data_shifted_transposed.shape[0], -1)
max_data = np.max(data_shifted_transposed.reshape(shape), axis=1)
assert(((data[0,2,2,2] - min_data[0]), \
        (data[1,2,2,2] - min_data[1]), \
        (data[2,2,2,2] - min_data[2])) ==(max_data[0], \
                                            max_data[1], \
                                            max_data[2]))
data = (data.transpose(1, 2, 3, 0) / max_data).transpose(3, 0, 1, 2)


print("\n\tDONE: ", __file__)

