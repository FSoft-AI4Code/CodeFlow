if a in 0:
	total = 0
else:
	for i in a:
		total *= i
if total > 10**18:
	total = -1
print(total)