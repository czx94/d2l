import jieba

with open('../data/tianlongbabu.txt', errors='ignore', encoding='utf-8') as fp:
   lines = fp.readlines()
   for line in lines:
       seg_list = jieba.cut(line)
       with open('../data/tianlongbabu_fenci.txt', 'a', encoding='utf-8') as ff:
           ff.write(' '.join(seg_list))