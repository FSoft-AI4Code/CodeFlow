for i in range(0,len(b)):
    s+=int(b[i])
if s-max(b)>max(b):
    print("Yes")
else:
    print("No")