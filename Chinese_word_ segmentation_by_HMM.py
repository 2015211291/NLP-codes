import numpy as np


# 消除词性标注,传入列表,返回列表
def eliminate_tag(all_data)
	temp = []
	for sr in all_data:
		for i, c in enumerate(sr):
			if c == '/':
				sr = sr[:i]
				temp.append(sr)
	return temp


# 分组，边界为a；若80%为训练集，20%为测试集，a=0.8
# 传入一个列表和边界，返回两个列表
def grouping(temp, a):
	train_set = []
	test_set = []
	length = len(temp)
	for i, sr in enumerate(temp):
		if i < a * length:
			train_set.append(sr)
		else:
			test_set.append(sr)
	return  train_set, test_set


# 建立训练集的状态序列
# Q={S,B,M,E} S:单独的字;B:开头的字;M:中间的字;E:末尾的字
# 传入列表，返回对应的状态序列列表
def get_state_set(train_set):
	train_state_set = []
	states = 'B'
	for sr in train_set:
		if len(sr) == 1:
			train_state_set.append('S')
		else:
			for i in range(len(sr)):
				if i == 0:
					states = 'B'
				else:
					if i == len(sr) - 1:
						states += 'E'
					else:
						states += 'M' 
			train_state_set.append(states)
	return train_state_set


# 传入统计完该状态下字符出现次数的字典
# 修改S为发射概率矩阵
def get_EmitMatrix(S):
	keys = S.values()# 获得全部键值的列表
	S_num = sum(keys)# 求得该状态下字符出现次数的总和
	for key in S:
		S[key] = S[key] / S_num


# 将列表中已消除词性标注的字符串，连接为一个字符串用来分词
def connect(temp):
	s = temp[0]
	for i in range(1, len(temp)):
		s = s + temp[i]
	return s



# 读取文件
f = open("p.txt", 'r', encoding='gbk')
all_data = f.read().split()


# 消除词性标注
temp = eliminate_tag(all_data)


# 分组，80%为训练集，20%为测试集
train_set, test_set = grouping(temp, 0.8)


# 获得状态序列
train_state_set = get_state_set(train_set)


# 计算初始概率分布initStatus
# 统计每个句子第一个状态为B和S的数目,从而求概率
initStatus = []
B_num = 0
S_num = 0
for st in train_state_set:
	if st[0] == 'B':
		B_num = B_num + 1
	else:
		S_num = S_num + 1

S_prob = S_num / (S_num+B_num)
initStatus.append(S_prob)

B_prob = B_num / (S_num+B_num)
initStatus.append(B_prob)

M_prob = 0
initStatus.append(M_prob)

E_prob = 0
initStatus.append(E_prob)


# 计算转移概率矩阵transMatrix (4*4)
# S->M,E B->B M->S,B E->M,E都是不可能发生的，置为0
transMatrix = np.zeros((4, 4))
transMatrix[0][2] = transMatrix[0][3] = 0
transMatrix[1][1] = 0
transMatrix[2][0] = transMatrix[2][1] = 0
transMatrix[3][2] = transMatrix[3][3] = 0

# 统计各状态转移次数
transNum = np.zeros((4, 4))
prev_st = 'S'
st = 'B'
for sr in train_state_set:
	for c in sr:
		st = c
		s_trans = prev_st + st
		if s_trans == 'SS':
			transNum[0][0] = transNum[0][0] + 1
		else:
			if s_trans == 'SB':
				transNum[0][1] = transNum[0][1] + 1
			else:
				if s_trans == 'BS':
					transNum[1][0] = transNum[1][0] + 1
				else:
					if s_trans == 'BM':
						transNum[1][2] = transNum[1][2] + 1
					else:
						if s_trans == 'BE':
							transNum[1][3] = transNum[1][3] + 1
						else:
							if s_trans == 'MM':
								transNum[2][2] = transNum[2][2] + 1
							else:
								if s_trans == 'ME':
									transNum[2][3] = transNum[2][3] + 1
								else:
									if s_trans == 'ES':
										transNum[3][0] = transNum[3][0] + 1
									else:
										if s_trans == 'EB':
											transNum[3][1] = transNum[3][1] + 1
		prev_st = c

# 根据转移次数计算概率，并填入矩阵
rows_sum = np.sum(transNum, axis=1)
transMatrix[0][0] = transNum[0][0] / rows_sum[0]
transMatrix[0][1] = transNum[0][1] / rows_sum[0]
transMatrix[1][0] = transNum[1][0] / rows_sum[1]
transMatrix[1][2] = transNum[1][2] / rows_sum[1]
transMatrix[1][3] = transNum[1][3] / rows_sum[1]
transMatrix[2][2] = transNum[2][2] / rows_sum[2]
transMatrix[2][3] = transNum[2][3] / rows_sum[2]
transMatrix[3][0] = transNum[3][0] / rows_sum[3]
transMatrix[3][1] = transNum[3][1] / rows_sum[3]


# 计算发射概率矩阵
# 遍历训练集，统计每个状态下不同字符的出现次数
S = {}
B = {}
M = {}
E = {}
for (sr, st) in zip(train_set, train_state_set):
	for i in range(len(sr)):
		if st[i] == 'S':
			if sr[i] in S:
				S[sr[i]] = S[sr[i]] + 1 
			else:
				S[sr[i]] = 1 
		else:
			if st[i] == 'B':
				if sr[i] in B:
					B[sr[i]] = B[sr[i]] + 1 
				else:
					B[sr[i]] = 1 
			else:
				if st[i] == 'M':
					if sr[i] in M:
						M[sr[i]] = M[sr[i]] + 1
					else:
						M[sr[i]] = 1 
				else:
					if st[i] == 'E':
						if sr[i] in E:
							E[sr[i]] = E[sr[i]] + 1 
						else:
							E[sr[i]] = 1 

# 根据统计次数得到发射概率矩阵
get_EmitMatrix(S)
get_EmitMatrix(B)
get_EmitMatrix(M)
get_EmitMatrix(E)


# 
