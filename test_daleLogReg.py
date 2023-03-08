import numpy as np
import pandas as pd
import DALE_ML
import pytest 

# defining good and bad dependent variables
good_dv = [0,1,1,1,0,0,0,0,1,0,1,0,0,0,1,1,0,1,0,1,0,1,1]
badLevel_dv = [0,4,2,7,10,8,2]
badShape_dv = [[3,4,5,9],
              [1,2,3,4],
              [1,1,1,1]]
good_ivs = [[2,2,2,2],
            [3,3,3,3]]
good_dv = np.asarray(good_dv)
badLevel_dv = np.asarray(badLevel_dv)
badShape_dv = np.asarray(badShape_dv)

# testing if function raises exception if...
#...multi-dimensional dependent variable is entered as input
def test_load_multidim_dv():
    with pytest.raises(TypeError):
        DALE_ML.ML_model(badShape_dv, good_ivs)
    
# testing if function raises exceptoin if...
#...non-binary dependent variable is entered as input
def test_load_nonbinary_dv():
    with pytest.raises(TypeError):
        DALE_ML.ML_model(badLevel_dv, good_ivs)