#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

min = np.array([
    [9.37500000e-01, -7.08333333e-01,  1.70000000e+02],
    [0.00000000e+00, -4.58984375e-01,  3.00000000e+02],
    [-0.00000000e+00, -2.21354167e-03,  1.00000000e+00]
],dtype=np.float32)


a = np.array([16,262,0])

print(min@a)

