from rich.console import Console
import sympy

def getIntervalEndsRoot(expr, x, a, b):
  f_a = expr.subs(x, a)
  f_b = expr.subs(x, b)
  if (f_a == 0):
    return f_a
  if (f_b == 0):
    return f_b
  return None

def dichotomy(expr, x, a, b, epsilon):
  f_a = expr.subs(x, a)
  f_b = expr.subs(x, b)
  i = 0
  if (f_a * f_b > 0):
    return None, None
  if (f_a == 0):
    return f_a, i
  if (f_b == 0):
    return f_b, i
  currentA = a
  currentB = b
  while currentB - currentA > epsilon: 
    i += 1
    c = (currentA + currentB) / 2
    f_a = expr.subs(x, currentA)
    f_b = expr.subs(x, currentB)
    f_c = expr.subs(x, c)
    if (f_a * f_c < 0):
      currentB = c
    elif (f_b * f_c < 0):
      currentA = c
    elif (f_c == 0):
      return c, i
  return (currentA + currentB) / 2, i

def relaxation(expr, x, a, b, lambd, epsilon):
  if lambd < 0 or lambd > 1: 
    return None, None
  i = 0

  intervalEndsRoot = getIntervalEndsRoot(expr, x, a, b)
  if (intervalEndsRoot is not None):
    return intervalEndsRoot, i

  x_i = (a + b) / 2
  der = sympy.diff(expr, x)
  der_val = der.subs(x, x_i)
  if (der_val == 0):
    return None, None
  alpha = (lambd - 1) / der_val
  x_ii = alpha * expr.subs(x, x_i) + x_i 
  while abs(x_ii - x_i) > epsilon:
    x_i = x_ii
    x_ii = alpha * expr.subs(x, x_ii) + x_ii 
    i += 1
  return x_ii, i

# |alpha * f'(x) + 1| <= lambda
#  -lambda <= alpha * f'(x) + 1 <= lambda
#  -lambda <= alpha * f'(x) + 1 <= lambda
# (-lambda - 1) / f'(x) <= alpha <= (lambda - 1) / f'(x)
# alpha = ((-lambda - 1) / f'(x) + (lambda - 1) / f'(x)) / 2
# alpha = - 1 / f'(x)
def newton(expr, x, a, b, epsilon):
  i = 0

  intervalEndsRoot = getIntervalEndsRoot(expr, x, a, b)
  if (intervalEndsRoot is not None):
    return intervalEndsRoot, i

  x_i = (a + b) / 2
  der = sympy.diff(expr, x)
  der_val = der.subs(x, x_i)
  if (der_val == 0):
    return None, None
  alpha =  -1 / der_val
  x_ii = alpha * expr.subs(x, x_i) + x_i 
  while abs(x_ii - x_i) > epsilon:
    der_val = der.subs(x, x_i)
    if (der_val == 0):
      return None, None
    alpha = -1 / der_val
    x_i = x_ii
    x_ii = alpha * expr.subs(x, x_ii) + x_ii 
    i += 1
  return x_ii, i

def printResult(methodName, res, iterations, expr, x):
  console.print(methodName, style="red")
  console.print("value:", res)
  console.print("function value:", expr.subs(x, res))
  console.print("iterations:", iterations)

if __name__ == '__main__':
  sympy.init_printing() 

  console = Console()

  x = sympy.symbols('x')

  # expr = x ** 2 - 4
  expr = sympy.sin(x) - 0.5
  # expr = x - sympy.Rational(1, 2)

  a = -1
  b = 1.1
  epsilon = 10e-13

  dichotomyRes = dichotomy(expr, x, a, b, epsilon)
  printResult("DICHOTOMY:", *dichotomyRes, expr, x)

  relaxationLambda = 0.2
  relaxationRes = relaxation(expr, x, a, b, relaxationLambda, epsilon)
  printResult("RELAXATION:", *relaxationRes, expr, x)

  newtonRes = newton(expr, x, a, b, epsilon)
  printResult("NEWTON:", *newtonRes, expr, x)
