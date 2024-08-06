ans = 10**12
for k in range(1,(n+1)**0.5):
    if n%k == 0 :
        m = n//k + k - 2
        if ans > m:
            ans = m
        else:
            print(ans)
            sys.exit()
print(ans)