k = '4'
flag = True
for i in range(k):
    if s[i] == '1':
        print(s[i])
        flag = False
if flag:
    print(s[1])