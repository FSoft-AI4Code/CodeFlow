err = []
for i in range(len(s) - 3):
  err.append(abs(int(s[i:i+3]), 753))
print(min(err))