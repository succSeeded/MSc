import numpy as np;
import time;
from utils import progress_bar;

def der1(func, point, h=1e-9):
return (func(point+h*0.5)-func(point-h*0.5))/h;

def partial(func, point, n_arg=1, h=1e-9):
step = np.zeros(point.shape[0]);
step[n_arg-1] = h;
return (func(point+step*0.5)-func(point-step*0.5))/h;

def der2(func, point, h=1e-9):
return (func(point+h) - 2.0*func(h) + func(point-h))/(h**2);

def find_zero(func, interval, method='linear', j=0, time_this=False):   

if time_this:
start = time.time();

match method.lower():
case 'linear':
zero = interval[0]-(interval[1]-interval[0])*loss/(loss_next-loss);
case 'halley':
zero = (interval[1]+interval[0])*0.5;
i = j;
while True:
i += 1;
zero_prev = zero;
zero = zero - func(zero)/(der1(func, zero) - 0.5*func(zero)*der2(func,zero)/der1(func,zero));
if np.abs(func(zero)) < 1e-6 or i > 1e+5:
break;
case 'golden_section':
a = interval[0];
b = interval[1];
eps = 1e-7;
t = (1 + np.sqrt(5))*0.5;
x1 = a + (1 - 1/t) * (b - a);
x2 = a + 1/t * (b - a);
l = b - a;
f1 = f(x1);
f2 = f(x2);
while (l > eps): 
if (f1 > f2):
a = x1;
x1 = x2;
f1 = f2;
x2 = b - (b - a)/(t+1);
f2 = f(x2);
else:
b = x2;
x2 = x1;
f2 = f1;
x2 = a + (b - a)/(t+1);
f1 = f(x1);
l = b - a;
x = (a + b)*0.5;
return x;
if time_this:
end = time.time();
return (zero, end-start);
return (zero, None);

def zero_localizer(func, interval, d=1e-1, k=0):
""" 
@param func (callable): Function for which zeros are localized. 
@param interval (ArrayLike): Interval in which zeros are localized. 
@param d (float): Diameter of interval division.
@param k (int): number of recursive function calls. 
This function finds zeros by continuously searching for sign changes on nodes of a division of interval and then subdivide it until a zero is found.
"""
zeros = [];
zero_times = [];
points = np.arange(interval[0], interval[1], d);
target = func(points);
for i in range(len(points)-1):
if k == 0 and i%1e+5==0:
progress_bar(i, len(points)-2);
if (target[i]*target[i+1] < 0) and (np.abs(d) > 1e-6):
zeros, zero_times = zero_localizer(func, points[i:i+2], d=d*1e-1, k=k+1);
elif (target[i]*target[i+1] < 0) and (np.abs(d) <= 1e-6):
zero, zero_time = find_zero(func, points[i:i+2], method='halley', time_this=True);
zeros += [zero];
zero_times += [zero_time];
if k == 0:
progress_bar(i, len(points)-2);
return (zeros, zero_times);