#Punkt 6.1
#Punkt 6.2
import numpy as np

arr = np.array([1,2,3,4,5])
print(arr)

A = np.array([[1,2,3],[7,8,9]])
print(A)
A = np.array([[1,2,3],
              [7,8,9]])
print(A)
A = np.array([[1,2,
              3],
             [7,8,9]])
print(A)

v = np.arange(1,7)
print(v,"\n")
v = np.arange(-2,7)
print(v,"\n")
v = np.arange(1,10,3)
print(v,"\n")
v = np.arange(1,11,3)
print(v,"\n")
v = np.arange(1,2,0.1)
print(v,"\n")

v = np.linspace(1,3,4)
print(v)
v = np.linspace(1,10,4)
print(v)

X = np.ones((2,3))
Y = np.zeros((2,3,4))
Z = np.eye(2)
Q = np.random.rand(2,5)
print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)

#U = np.block([[A],[X,Z]])
#print(U) #Wypisuje Error:all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 2 and the array at index 1 has size 1

V = np.block([[
np.block([
np.block([[np.linspace(1,3,3)],
[np.zeros((2,3))]]) ,
np.ones((3,1))])
],
[np.array([100, 3, 1/2, 0.333])]] )
print(V)

print( V[0,2] )
print( V[3,0] )
print( V[3,3] )
print( V[-1,-1] )
print( V[-4,-3] )
print( V[3,:] )
print( V[:,2] )
print( V[3,0:3] )
print( V[np.ix_([0,2,3],[0,-1])] )
print( V[3] )

Q = np.delete(V,2,0)
print(Q)
Q = np.delete(V,2,1)
print(Q)

v = np.arange(1,7)
print(np.delete(v,3,0))

nsiv = np.size(v) #Dodalem printy
print(nsiv)
nshv = np.shape(v)
print(nshv)
nsiV = np.size(V)
print(nsiV)
nshV = np.shape(V)
print(nshV)

A = np.array([[1, 0, 0],
[2, 3, -1],
[0, 7, 2]] )
B = np.array([[1, 2, 3],
[-1, 5, 2],
[2, 2, 2]] )
print( A+B )
print( A-B )
print( A+2 )
print( 2*A )

MM1 = A@B
print(MM1)
MM2 = B@A
print(MM2)

MT1 = A*B
print(MT1)
MT2 = B*A
print(MT2)

DT1 = A/B
print(DT1)

C = np.linalg.solve(A,MM1)
print(C)
x = np.ones((3,1))
b = A@x
y = np.linalg.solve(A,b)
print(y)

PM = np.linalg.matrix_power(A,2)
PT = A**2

A.T
A.transpose()
A.conj()
A.conj().transpose()

A == B
A != B
2 < A
A > B
A < B
A >= B
A <= B
np.logical_not(A)
np.logical_and(A, B)
np.logical_or(A, B)
np.logical_xor(A, B)
print( np.all(A) )
print( np.any(A) )
print( v > 4 )
print( np.logical_or(v>4, v<2))
print( np.nonzero(v>4) )
print( v[np.nonzero(v>4) ] )

print(np.max(A))
print(np.min(A))
print(np.max(A,0))
print(np.max(A,1))
print( A.flatten() )

import matplotlib.pyplot as plt
x = [1,2,3]
y = [4,6,5]
plt.plot(x,y)
plt.show()

x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y)
plt.show()

x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y,'r:',linewidth=6)
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Nasz pierwszy wykres')
plt.grid(True)
plt.show()

x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
plt.plot(x,y1,'r:',x,y2,'g')
plt.legend(('dane y1','dane y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)
plt.show()

x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
y = y1*y2
l1, = plt.plot(x,y,'b')
l2,l3 = plt.plot(x,y1,'r:',x,y2,'g')
plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)
plt.show()

#Punkt 6.3

A1 = np.array([np.arange(1,6)])
A2 = np.array([np.arange(-5,0)*(-1)])
A3 = np.zeros((2,2))
A4 = np.ones((2,3))*2
A5 = np.ones((5,1))*10
A6 = np.array([0, 0, -90, -80, -70])

A12 = np.vstack((A1,A2))
A34 = np.hstack((A3,A4))
A14 = np.vstack((A12,A34))
A146 = np.vstack((A14,A6))
A = np.hstack((A146,A5))
print(A)

import numpy as np
import matplotlib.pyplot as plt
import funzad15


#Punkt 6.4
print("\n#Zadanie 4\n")

B = A[1, :] + A[3, :]
print(B)

# Zadanie 5
print("\n#Zadanie 5\n")

C = np.array([])
for i in range(np.shape(A)[1]):
    C = np.append(C, max(A[:, i]))
print(C)

# Zadanie 6
print("\n#Zadanie 6\n")
D = np.delete(B, [0, 5])
print(D)

# Zadanie 7
print("\n#Zadanie 7\n")
D[D == 4] = 0
print(D)

# Zadanie 8
print("\n#Zadanie 8\n")
E = C[C > min(C)]
E = E[E < max(E)]
print(E)

# Zadanie 9
print("\n#Zadanie 9\n")
j = np.array([])
for i in range(np.shape(A)[0]):
    if np.isin(np.max(A), A[i, :]) and np.isin(np.min(A), A[i, :]):
        print(A[i, :])

# Zadanie 10
print("\n#Zadanie 10\n")
print("Mnożenie tablicowe: ")
print(D * E)
print("Mnożenie wektorowe: ")
print(D @ E)

# Zadanie 11
print("\n#Zadanie 11\n")


def fun11(n):
    matrix = np.random.randint(0, 11, [n, n])
    return matrix, np.trace(matrix)


print(fun11(3))

# Zadanie 12
print("\n#Zadanie 12\n")


def fun12(matrix):
    size = np.shape(matrix)
    matrix = matrix * (1 - np.eye(size[0], size[0]))
    matrix = matrix * (1 - np.fliplr(np.eye(size[0], size[0])))
    return matrix


tempMatrix = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 150, 11, 12],
                       [13, 14, 15, 16]])

print(fun12(tempMatrix))

# Zadanie 13
print("\n#Zadanie 13\n")


def fun13(matrix):
    s = 0
    size = np.shape(matrix)
    for i in range(size[0]):
        if i % 2 == 1:
            s = s + np.sum(matrix[i, :])
    return s


print(fun13(tempMatrix))

# Zadanie 14
print("\n#Zadanie 14\n")
y = lambda x: np.cos(2 * x)
x_points = np.linspace(-10, 10, 201)
plt.plot(x_points, y(x_points), color='red', dashes=[2, 2])
# plt.show()

# Zadanie 15
print("\n#Zadanie 15\n")
arr = np.array([])
for i in x_points:
    arr = np.append(arr, funzad15.y2(i))

plt.plot(x_points, arr, '+g')
# plt.show()

# Zadanie 16
print("\n#Zadanie 16\n")
y16 = lambda x: np.sin(x) if x<0 else np.sqrt(x)
print(y16(-np.pi))
print(y16(9))

# Zadanie 17
print("\n#Zadanie 17\n")
plt.plot(x_points, 3 * y(x_points) + arr, '*b')
plt.show()

# Zadanie 18
print("\n#Zadanie 18\n")
matrix = np.array([[10, 5, 1, 7],
                   [10, 9, 5, 5],
                   [1, 6, 7, 3],
                   [10, 0, 1, 5]])
right_matrix = np.array([[34],
                         [44],
                         [25],
                         [27]])
x_matrix = np.linalg.solve(matrix, right_matrix)
print(x_matrix)


# Zadanie 19
print("\n#Zadanie 19\n")
x = np.linspace(0, 2 * np.pi, 1000000)
y = np.sin(x)
integral = np.sum(2 * np.pi / 1000000 * y)
print(integral)