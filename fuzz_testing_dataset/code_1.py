ans=0
cur=0
ACGT=set("A","C","G","T")
for ss in s:
  if ss in ACGT:
    cur+=1
  else:
    ans=max(cur,ans)
    cur=0
print(max(ans,cur))