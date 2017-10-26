import numpy as np
import matplotlib.pyplot as plt

class Fisher:
	def __init__(self):
		self.data_Set1 = []
		self.data_Set2 = []
		self.m_1 = np.mat([0., 0., 0.]).T
		self.m_2 = np.mat([0., 0., 0.]).T
		self.S_1 = np.mat([[0., 0., 0.], [0., 0., 0.,], [0., 0., 0.]])
		self.S_2 = np.mat([[0., 0., 0.], [0., 0., 0.,], [0., 0., 0.]])
		self.w = np.mat(np.random.random(4)).T
		self.b = 0
		self.load_data()


	def load_data(self):
		with open("data.txt") as file:
			datas = file.readlines()
			for data in datas:
				if data[-1] == '\n':
					xdata = data[:-1].split(' ')
				else:
					xdata = data.split(' ')
				while xdata.count('') > 0:
					xdata.remove('')
				X = np.mat(list(map(float, xdata[:-1]))).T
				label = int(xdata[-1])
				if label == 1:
					self.data_Set1.append(X)
				else:
					self.data_Set2.append(X)

	def test_load(self):
		print(self.m_1)
		print(self.m_2)
		print(self.S_1)
		print(self.S_2)

	def get_para(self):
		for data in self.data_Set1:
			self.m_1 += data
		self.m_1 /= len(self.data_Set1)
		for data in self.data_Set2:
			self.m_2 += data
		self.m_2 /= len(self.data_Set2)

		for data in self.data_Set1:
			self.S_1 += np.mat(data - self.m_1) * np.mat(data - self.m_1).T
		for data in self.data_Set2:
			self.S_2 += np.mat(data - self.m_2) * np.mat(data - self.m_2).T

	def get_weight(self):
		return (self.S_1 + self.S_2) ** (-1) * np.mat(self.m_1 - self.m_2)

	def get_bias(self):
		return - 1/2 * self.w.T * np.mat(self.m_1 + self.m_2)

	def run(self):
		self.get_para()
		self.w = self.get_weight()
		self.b = self.get_bias()
		print(self.w)
		print(self.b)

	def test(self):
		corrent = 0
		for data in self.data_Set1:
			label = self.w.T * data + self.b
			if label > 0:
				corrent += 1
		for data in self.data_Set2:
			label = self.w.T * data + self.b
			if label < 0:
				corrent += 1
		print(corrent / (len(self.data_Set1 + self.data_Set2)))

def main():
	model = Fisher()
	model.run()
	model.test()


if __name__ == "__main__":
	main()
