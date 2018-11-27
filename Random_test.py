# Test the Powell optimization method
# Goal: input any functions and return any good solutions
import math
import numpy as np

def bracket(f, x1, h):

    # Find the search range only based on the x1, h and f (function)
    # Find the bracket within this function
    c = 1.618033989

    # print("The f inside the function is ", f)
    f1 = f(x1)
    x2 = x1 + h
    f2 = f(x2)

    # Determine the downhill and change sign if needed
    if f2 > f1:
        h = -h
        x2 = x1 + h
        f2 = f(x2)

        # check if minimum between x1 - h and x1 + h
        if f2 > f1:
            return x2, x1 - h

    for i in range(100):  # maximum 100 times
        h = c * h
        x3 = x2 + h
        f3 = f(x3)
        if f3 > f2:
            return x1, x3
        x1 = x2
        x2 = x3
        f1 = f2
        f2 = f3

def Powell(F, x, h = 0.1, tol = 1.0e-6):

    # Define a new function
    def f(s):
        return F(x + s * v)

    n = len(x)          # number of design variables
    df = np.zeros(n)    # Decreases of F stored here
    u = np.identity(n)  # Initial vectors here by rows

    for j in range(30):

        # Allow for only 30 cycles (loops) - maximum n times -- no less then 20 is good enough
        # j is the number of iteration -- this is a good idea

        xOld = x.copy()  # The input x is usually the xStart point
        fOld = F(xOld)

        # First n line searches record decreases of F -- followed by the last line search algorithm
        for i in range(n):

            # The initial direction on v -- This is im as well
            v = u[i]

            # problem with this cas
            a, b = bracket(f, 0.0, h)

            # For the line search only
            s, fMin = search(f, a, b, tol = 1.0e-9)
            df[i] = fOld - fMin
            fOld = fMin
            x = x + s * v

        # Last line search in the cycle -- this is im -- why this works and how we can prove that?
        v = x - xOld

        # Problem with this sentence
        a, b = bracket(f, 0.0, h)
        s, fLast = search(f, a, b, tol = 1.0e-9)
        x = x + s * v

        # Check for convergence
        if math.sqrt(np.dot(x - xOld, x - xOld) / n) < tol:
            return x, j + 1

        # Identify biggest decrease
        iMax = np.argmax(df)

        # update search directions
        for i in range(iMax, n - 1):
            u[i] = u[i + 1]

        u[n - 1] = v

    print("Powell did not converge")

def search(f, a, b, tol = 1.0e-9):

    # Initial position
    # Calculate the total iteration numbers
    nIter = int(math.ceil(-2.078087*math.log(tol/abs(b-a))))
    R = 0.618033989
    C = 1.0 - R

    # First telescoping -- Define the golden length
    x1 = R * a + C * b
    x2 = C * a + R * b
    f1 = f(x1)
    f2 = f(x2)

    # Main loop
    for i in range(nIter):

        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = C * a + R * b
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = R * a + C * b
            f1 = f(x1)
    if f1 < f2:
        return x1, f1
    else:
        return x2, f2

def test_1():

    # Test the golden line function
    def f(x, x_test, v_test):
        lam = 1.0
        c = min(0.0, x)
        return 1.6 * x ** 3 + 3.0 * x ** 2 - 2.0 * x + lam * c ** 2

    x_test = 0
    v_test = 0

    xStart = 1.0
    h = 0.01
    x1, x2 = bracket(f, xStart, h, x_test, v_test)
    x, fMin = search(f, x1, x2, x_test, v_test)
    print("x =", x)
    print("f(x) =", fMin)
    input("\nPress return to exit")

def test_2():

    def F(x):
        return x[0] * x[0] + 2 * x[1] * x[1] - 4 * x[0] - 2 * x[0] * x[1]

    xStart = np.array([1.0, 1.0])
    xMin, nIter = Powell(F, xStart)

    print("x =", xMin)
    print("F(x) =", F(xMin))
    print("Number of cycles =", nIter)
    input("Press return to exit")

if __name__ == "__main__":

    test_2()