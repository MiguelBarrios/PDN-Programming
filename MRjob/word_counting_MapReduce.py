from mrjob.job import MRJob
import re

WORD_RE = re.compile(r"[\w']+")

class MRWordCounting(MRJob):
	def mapper(self, _, line):
		count = [0] * 26
		line = line.lower()
		x = set()
		word_dictionary = {}
		for word in WORD_RE.findall(line):
			letter = word[0]
			if letter >= 'a' and letter <= 'z' :
				if letter in word_dictionary:
					word_dictionary[letter].append(word)
				else:
					word_dictionary[letter] = [word]
		for (key, local_value) in word_dictionary.items():
			yield (key, local_value)

	def reducer(self, letter, words):
		x = set()
		combined_list = []
		lists = [x for x in words]
		for l in lists:
			for word in l:
				combined_list.append(word)
		combined_list.sort(key = len, reverse=True)
		maxLen = len(combined_list[0])
		output = []
		for i in combined_list:
			if len(i) == maxLen:
				if i not in x:
					output.append(i)
			else:
				break
			x.add(i)
		yield(letter, output)

if __name__ == '__main__':
    MRWordCounting.run()

