def hour2cat(x):
	if (x>=6) & (x<= 12): #早上
		y= 0
	elif (x>12) & (x< 18): #下午
		y= 1
	else:
		y= 2 #晚上
	return y
