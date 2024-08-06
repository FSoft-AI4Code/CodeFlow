if a == b & b == c:
    ans = 0
else:
    for count in range(n):
        tmp = 0
        if a[count] == b[count]:
            if a[count] != c[count]:
                tmp += 1
        else:
            tmp += 1
            if (a[count] != c[count]) & (b[count] != c[count]):
                tmp += 1
        ans += tmp
print(ans)