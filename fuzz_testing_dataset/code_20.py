a = [1, 1, 2, 2, 2]
t=a[0]
s=0
Ans=0
for i in range(1,n):
  if t==a[i]:
    s+=1
  else:
    Ans+=s//2
    s=0
  t=a[i]
print(Ans)