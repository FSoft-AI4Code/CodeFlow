from math import sqrt
for i in range(sqrt(n),0,-1):
  if n%i==0:
    j=n//i
    print(i+j-2)
    break