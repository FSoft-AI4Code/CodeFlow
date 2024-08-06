for i in A:
  if A%2==0:
    if not A%3==0 or A%5==0:
      print("DENIED")
      x=1
      break
  else:
   	continue
if x !=0:
	print("APPROVED")