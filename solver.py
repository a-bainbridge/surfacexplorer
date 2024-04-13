from math import sqrt
import time
import scipy
import scipy.integrate
import sympy.diffgeom
from sympy.diffgeom.rn import R2
from sympy.diffgeom import TensorProduct
import numpy as np
TP = TensorProduct
x,y=sympy.symbols('x y')
# christoffelx, christoffely = sympy.diffgeom.metric_to_Christoffel_2nd(R2.y*TP(R2.dx,R2.dx) + R2.x*TP(R2.dy,R2.dy))
# metric = 4*(TP(R2.dx,R2.dx) + TP(R2.dy,R2.dy))/(1-R2.x**2-R2.y**2)**2
# metric = (TP(R2.dx,R2.dx) + TP(R2.dy,R2.dy))/R2.y**2
metric = TP(R2.dx,R2.dx) + TP(R2.dy,R2.dy)
christoffelx, christoffely = sympy.diffgeom.metric_to_Christoffel_2nd(metric)
# print(christoffelx[0][0])
precomputedcx = [sympy.diff(christoffelx[0][0].subs({R2.x:x,R2.y:y}),x), sympy.diff(christoffelx[0][1].subs({R2.x:x,R2.y:y}),x), sympy.diff(christoffelx[1][1].subs({R2.x:x,R2.y:y}),x), \
  sympy.diff(christoffelx[0][0].subs({R2.x:x,R2.y:y}),y), sympy.diff(christoffelx[0][1].subs({R2.x:x,R2.y:y}),y), sympy.diff(christoffelx[1][1].subs({R2.x:x,R2.y:y}),y)]
precomputedcy = [sympy.diff(christoffely[0][0].subs({R2.x:x,R2.y:y}),x), sympy.diff(christoffely[0][1].subs({R2.x:x,R2.y:y}),x), sympy.diff(christoffely[1][1].subs({R2.x:x,R2.y:y}),x), \
  sympy.diff(christoffely[0][0].subs({R2.x:x,R2.y:y}),y), sympy.diff(christoffely[0][1].subs({R2.x:x,R2.y:y}),y), sympy.diff(christoffely[1][1].subs({R2.x:x,R2.y:y}),y)]
def jac(t,w):
  a1 = christoffelx[0][0].evalf(subs={R2.x:w[0],R2.y:w[1]})
  a2 = christoffelx[0][1].evalf(subs={R2.x:w[0],R2.y:w[1]})
  a3 = christoffelx[1][1].evalf(subs={R2.x:w[0],R2.y:w[1]})
  b1 = christoffelx[0][0].evalf(subs={R2.x:w[0],R2.y:w[1]})
  b2 = christoffelx[0][1].evalf(subs={R2.x:w[0],R2.y:w[1]})
  b3 = christoffelx[1][1].evalf(subs={R2.x:w[0],R2.y:w[1]})
  return [[0,0,1,0],[0,0,0,1], \
  [-(precomputedcx[0].evalf(subs={x:w[0],y:w[1]}) * w[2]**2 + 2*precomputedcx[1].evalf(subs={x:w[0],y:w[1]})*w[2]*w[3] + precomputedcx[2].evalf(subs={x:w[0],y:w[1]}) * w[3]**2), \
    -(precomputedcx[3].evalf(subs={x:w[0],y:w[1]}) * w[2]**2 + 2*precomputedcx[4].evalf(subs={x:w[0],y:w[1]})*w[2]*w[3] + precomputedcx[5].evalf(subs={x:w[0],y:w[1]}) * w[3]**2),
    -(2* a1* w[2] + 2*a2 * w[3]),\
    -(2*a2*w[1] + 2*a3*w[3])
    ],
     [-(precomputedcy[0].evalf(subs={x:w[0],y:w[1]}) * w[2]**2 + 2*precomputedcy[1].evalf(subs={x:w[0],y:w[1]})*w[2]*w[3] + precomputedcy[2].evalf(subs={x:w[0],y:w[1]}) * w[3]**2), \
    -(precomputedcy[3].evalf(subs={x:w[0],y:w[1]}) * w[2]**2 + 2*precomputedcy[4].evalf(subs={x:w[0],y:w[1]})*w[2]*w[3] + precomputedcy[5].evalf(subs={x:w[0],y:w[1]}) * w[3]**2),
    -(2* b1* w[2] + 2*b2 * w[3]),\
    -(2*b2*w[1] + 2*b3*w[3])
    ]]

#u1,v1,u2,v2
def model(t,w):
  return np.array([w[2], \
  w[3], \
  -(christoffelx[0][0].evalf(subs={R2.x:w[0],R2.y:w[1]}) * w[2]**2 + 2*(christoffelx[0][1]).evalf(subs={R2.x:w[0],R2.y:w[1]}) * w[2]*w[3] + christoffelx[1][1].evalf(subs={R2.x:w[0],R2.y:w[1]}) * w[3]**2), \
  -(christoffely[0][0].evalf(subs={R2.x:w[0],R2.y:w[1]}) * w[2]**2 + 2*(christoffely[0][1]).evalf(subs={R2.x:w[0],R2.y:w[1]}) * w[2]*w[3] + christoffely[1][1].evalf(subs={R2.x:w[0],R2.y:w[1]}) * w[3]**2),\
  ])


# print(model(0,[0.5,0.5,1,0]))
tim = time.time()
integrator = scipy.integrate.LSODA(fun=model, t0=0, y0=[0.5,0.5,1/2,sqrt(3)/2], t_bound=2, max_step=0.05, jac=jac)
while integrator.status == "running":
  integrator.step()
  print(f"{integrator.y[0]},{integrator.y[1]}")
print(time.time()-tim)