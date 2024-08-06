a = [1, 2, 3, 4, 5, 6]
memo = sum(a)
a=a[0]
b=memo-a[0]
ans = abs(a-b)
for i in range(1,n-1):
    a += a[i]
    b -= a[i]
    ans = min(ans,abs(a-b))
print(ans)