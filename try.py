import re

fin = open("features.txt", "rt")
fout = open("out.txt", "wt")

for line in fin:
	fout.write(re.sub('\s+',',',line))
	
fin.close()
fout.close()
