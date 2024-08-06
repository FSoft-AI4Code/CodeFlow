import math
m=N+1
for j in range(2,math.sqrt(N)):
    if N%j==0:
        a=j
        b=N//j
        if a>b:
            break
        else:
            m=min(m,(a+b))
print(m-2)