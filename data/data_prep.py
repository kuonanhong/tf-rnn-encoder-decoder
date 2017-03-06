import pickle
import zipfile
import collections
import re
import numpy as np

num_batches = 1000
k = 150

def read_data(filename):
    z = zipfile.ZipFile(filename, 'r')
    lines = list()
    line_count = 0
    with z.open(z.namelist()[0]) as f:
        for line in f:
            line_count += 1
            line = line.decode('utf-8')
            lines.append(line.replace(".", "").replace("?", "").replace("!", ""))
            if line_count == num_batches:
                break
    z.close()
    return lines

french = read_data('fr.zip')
english = read_data('en.zip')
print('Data size %d %d' % (len(english), len(french)))

def build_dataset(sentences):
  count = [['UNK', -1]]
  count.extend(collections.Counter(re.findall(r'\w+', ' '.join(sentences).lower())).most_common(k - 1))
  indeces = dict()
  for word, _ in count:
    indeces[word] = len(indeces)
  data = list()
  unk_count = 0
  for sentence in sentences:
    words = re.findall(r'[\w]+', sentence.lower())
    sentence_arr = np.zeros((k, 0))
    for word in words:
        if word in indeces:
            index = indeces[word]
        else:
            index = 0
            unk_count += 1
        vector = np.zeros((k, 1))
        vector[index] = 1
        sentence_arr = np.concatenate((sentence_arr, vector), axis=1)
    data.append(sentence_arr)
  count[0][1] = unk_count
  return data, count, indeces

fr_data, fr_count, fr_indeces = build_dataset(french)
print('Most common words (+UNK)', fr_count[:5])
pickle.dump(fr_data, open("fr_data.p", "wb"))
pickle.dump(fr_count, open("fr_count.p", "wb"))
pickle.dump(fr_indeces, open("fr_indeces.p", "wb"))

en_data, en_count, en_indeces = build_dataset(english)
print('Most common words (+UNK)', en_count[:5])
del english, french  # Hint to reduce memory.
pickle.dump(en_data, open("en_data.p", "wb"))
pickle.dump(en_count, open("en_count.p", "wb"))
pickle.dump(en_indeces, open("en_indeces.p", "wb"))