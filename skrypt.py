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

#Punkt 6.4

PB = np.zeros((6,1))
B = np.zeros((6,1))
for i in range (0,5):
    for j in range (0,6):
        if(i == 3):
           PB[j] = A[i,j]
print(PB)

for i in range (0,5):
    for j in range (0,6):
       if(i == 1):
            B[j] = A[i,j] + PB[j]          
print(np.transpose(B))

#Punkt 6.5
K = np.zeros((5,1))
C = np.zeros((6,1))
AT = np.transpose(A)
for i in range (0,6):
    for j in range (0,5):
        K[j] = AT[i,j]
        C[i] = max(K)
print(C)

#Punkt 6.6
DB1=np.delete(B,5)
D=np.delete(DB1,0)
print(D)

