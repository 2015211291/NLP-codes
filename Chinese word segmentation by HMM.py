import numpy as np


# 消除词性标注,传入列表,返回列表
def eliminate_tag(all_data):
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
	length = len(temp)
	train_set = temp[: int(length*a)]
	test_set = temp[int(length*a):]
	
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


# 将列表中已分过词的字符串元素，根据标点符号，合并成更大的句子，作为列表的新元素
def connect(temp):
	new_temp = []
	s = ''
	for i in range(len(temp)):
		s = s + temp[i]
		if (temp[i] == '，') or (temp[i] == '。') or (temp[i] == '！') or (temp[i] == '？'):
			new_temp.append(s)
			s = ''
	return new_temp


# Viterbi算法
# all_obs是按序存放各汉字，用于在发射矩阵中索引
def Viterbi(all_obs, obs, states, initStatus, transMatrix, emitMatrix):
	# max_prob[state][obs]表示在状态state下，观测到obs的最大概率
	max_prob = np.zeros((len(states), len(obs)))

	# path[state][obs]表示max_prob[state][obs]取到最大时，前一个状态，用于回溯
	path = np.zeros((len(states), len(obs)))

	# 初始化
	index = all_obs.index(obs[0])
	for i in range(len(states)):
		max_prob[i][0] = initStatus[i] * emitMatrix[i][index]
		path[i][0] = -1

	# 迭代
	# 遍历给定的观测序列
	for obs_count in range(1, len(obs)):
		index = all_obs.index(obs[obs_count])

		# 对每个观测结果，遍历所有状态找最优
		for st_count in range(len(states)):
			prob = -1
			path[st_count][obs_count] = -1

			# 对当前每个状态，遍历所有前一个字可能的状态
			for prev_st_count in range(len(states)):
				nprob = max_prob[prev_st_count][obs_count-1] * transMatrix[prev_st_count][st_count] * emitMatrix[st_count][index]

				if nprob > prob:
					prob = nprob
					max_prob[st_count][obs_count] = prob
					path[st_count][obs_count] = prev_st_count
		
	
	# 找到最大概率的路径终点
	max_pro = -1
	path_state = 0
	for st_count in range(len(states)):
		if max_prob[st_count][len(obs)-1] > max_pro:
			max_pro = max_prob[st_count][len(obs)-1]
			path_state = st_count

	# 回溯
	L_path = []
	L_path.append(int(path_state))
	for i in range(1, len(obs)):
		path_state = path[int(path_state)][len(obs)-i]
		L_path.append(int(path_state))

	return L_path[::-1]


# 在字符串sr的pos处插入c
def str_insert(sr, pos, c):
	sr = sr[:pos] + c + sr[pos:]
	return sr



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

# 计算转移概率矩阵transMatrix (4*4)
# S->M,E B->B M->S,B E->M,E都是不可能发生的，置为0
transMatrix = np.zeros((4, 4))

# 根据转移次数计算概率，并填入矩阵
rows_sum = np.sum(transNum, axis=1)
rows_sum = rows_sum.reshape((4,1))
transMatrix = transNum / rows_sum


# 计算发射概率矩阵
# 遍历训练集，统计每个状态下不同字符的出现次数
all_obs = []

# 先统计所有不同的字符，并存入all_obs，用于矩阵的索引
for sr in temp:
	for c in sr:
		if c not in all_obs:
			all_obs.append(c)

# 统计各状态下字符出现的次数
states = ['S', 'B', 'M', 'E']
emitMatrix = np.zeros((len(states), len(all_obs)))
for (sr, st) in zip(train_set, train_state_set):
	for i in range(len(sr)):
		index = all_obs.index(sr[i])
		if st[i] == 'S':
			emitMatrix[0][index] = emitMatrix[0][index] + 1 
		else:
			if st[i] == 'B':
				emitMatrix[1][index] = emitMatrix[1][index] + 1 
			else:
				if st[i] == 'M':
					emitMatrix[2][index] = emitMatrix[2][index] + 1 
				else:
					if st[i] == 'E':
						emitMatrix[3][index] = emitMatrix[3][index] + 1 

# 根据统计次数得到发射概率矩阵
rows_sum = emitMatrix.sum(axis=1)
rows_sum = rows_sum.reshape((4, 1))
emitMatrix = emitMatrix / rows_sum


# 分词,得到状态序列
s_temp = connect(test_set)
path_all = []
for sr in s_temp:
	path = Viterbi(all_obs, sr, states, initStatus, transMatrix, emitMatrix)
	path_all.append(path)


# 评估，求准确率，召回率，F值
test_state_set = get_state_set(test_set)
set_temp = []

# 先把两个列表统一格式
for sr in test_state_set:
	for c in sr:
		if c == 'S':
			set_temp.append(0)
		elif c == 'B':
			set_temp.append(1)
		elif c == 'M':
			set_temp.append(2)
		elif c == 'E':
			set_temp.append(3)
test_state_set = set_temp

set_temp = []
for sr in path_all:
	for c in sr:
		set_temp.append(int(c))
path_all = set_temp

# 一个个比较,统计状态正确的数目
num = 0
for (i, j) in zip(test_state_set, path_all):
	if i==j:
		num += 1

# 准确率
Precision = num / len(path_all)

# 召回率
all_state_set = get_state_set(temp)
set_temp = []
for sr in all_state_set:
	for c in sr:
		if c == 'S':
			set_temp.append(0)
		elif c == 'B':
			set_temp.append(1)
		elif c == 'M':
			set_temp.append(2)
		elif c == 'E':
			set_temp.append(3)
all_state_set = set_temp
Recall = num / len(all_state_set)

# F值
F = (2*Precision*Recall) / (Precision+Recall)

print("Precision: " + str(Precision))
print("Recall: " + str(Recall))
print("F: " + str(F))





