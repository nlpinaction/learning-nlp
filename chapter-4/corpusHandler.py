#coding=utf8


def tag_line(words, mark):
    chars = []
    tags = []
    temp_word = '' #用于合并组合词
    for word in words:
        word = word.strip('\t ')
        if temp_word == '':
            bracket_pos = word.find('[')
            w, h = word.split('/')
            if bracket_pos == -1:
                if len(w) == 0: continue
                chars.extend(w)
                if h == 'ns':
                    tags += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
                else:
                    tags += ['O'] * len(w)
            else:
                w = w[bracket_pos+1:]
                temp_word += w
        else:
            bracket_pos = word.find(']')
            w, h = word.split('/')
            if bracket_pos == -1:
                temp_word += w
            else:
                w = temp_word + w
                h = word[bracket_pos+1:]
                temp_word = ''
                if len(w) == 0: continue
                chars.extend(w)
                if h == 'ns':
                    tags += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
                else:
                    tags += ['O'] * len(w)

    assert temp_word == ''
    return (chars, tags)

def corpusHandler(corpusPath):
    import os
    root = os.path.dirname(corpusPath)
    with open(corpusPath) as corpus_f, \
        open(os.path.join(root, 'train.txt'), 'w') as train_f, \
        open(os.path.join(root, 'test.txt'), 'w') as test_f:

        pos = 0
        for line in  corpus_f:
            line = line.strip('\r\n\t ')
            if line == '': continue
            isTest = True if pos % 5 == 0 else False  # 抽样20%作为测试集使用
            words = line.split()[1:]
            if len(words) == 0: continue
            line_chars, line_tags = tag_line(words, pos)
            saveObj = test_f if isTest else train_f
            for k, v in enumerate(line_chars):
                saveObj.write(v + '\t' + line_tags[k] + '\n')
            saveObj.write('\n')
            pos += 1

if __name__ == '__main__':
    corpusHandler('./data/people-daily.txt')