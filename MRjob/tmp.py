from mrjob.job import MRJob
import re

WORD_RE = re.compile(r"[\w']+")

count = [0] * 26

f = open("1342-0.txt", "r")
line = f.read()

line = line.lower()
x = set()

word_dictionary = {}
for word in WORD_RE.findall(line):
	letter = word[0]
	if letter >= 'a' and letter <= 'z' :
		key = 97 - ord(word[0])
		if letter in word_dictionary:
			if word not in x:
				if len(word) > count[key]:
					word_dictionary[letter] = [word]
					count[key] = len(word)
				elif len(word) == count[key]:
					word_dictionary[letter].append(word)
		else:
			word_dictionary[letter] = [word]
			count[key] = len(word)
		x.add(word)


"""
word_dictionary = {}
for word in WORD_RE.findall(line):
	letter = word[0]
	if letter >= 'a' and letter <= 'z' :
		key = 97 - ord(word[0])
		if key in word_dictionary:
			if word not in x:
				if len(word) > count[key]:
					word_dictionary[key] = [word]
					count[key] = len(word)
				elif len(word) == count[key]:
					word_dictionary[key].append(word)
		else:
			word_dictionary[key] = [word]
			count[key] = len(word)
		x.add(word)
"""

for k, v in word_dictionary.items():
	print("{} : {}".format(k,v))
