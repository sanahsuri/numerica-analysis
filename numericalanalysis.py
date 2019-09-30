import numpy as np

# bisection method
# find some interval where f(a) f(b) < 0 .. compute midpoint = c of [a, b] .. check if c is root .. check if ac < 0

def funky(x):
    return np.power(x, 2) - (2 * x) + 1;

def bisection_method(f, a, b, e):

    if f(a) * f(b) > 0:
        return "error"

    while ((b-a)/2 > e):
        c = (a+b)/2
        if f(c) == 0.0:
             return c
        elif f(a) * f(c) < 0.0:
            b = c;
        elif f(b) * f(c) < 0.0:
            a = c;
    return c

print(bisection_method(funky, 0.5, 3, 0.0005))
#print(funky(2))
