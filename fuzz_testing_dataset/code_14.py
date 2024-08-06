cont_str = _in[0] * _in[1]
cont_num = int(cont_str)
sqrt_flag = False
for i in range(4, 100):
    sqrt = i * i
    if cont_num == sqrt:
        sqrt_flag = True
        break
if sqrt_flag:
    print('Yes')
else:
    print('No')