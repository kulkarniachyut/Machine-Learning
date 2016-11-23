
import numpy as np
from numpy import linalg as la
a = [[91.43 , 171.92 , 297.99],[171.92 , 393.92 , 545.21], [297.99 , 545.21 , 1297.26]]

print la.eigvals(a)
print la.eig(a)

