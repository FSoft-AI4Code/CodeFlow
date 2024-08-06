from collections import defaultdict
d = defaultdict(str)
no_flag = False
for i in range(len(S)):
    if S[i] != T[i]:
        if d[S[i]] == T[i]:
            S[i] = T[i]
            pass
        elif d[S[i]] == "" and d[T[i]] == "":
            d[S[i]] = T[i]
            d[T[i]] = S[i]
        else:
            pass
S = sorted(S)
T = sorted(T)
if S == T:
    print("Yes")
else:
    print("No")