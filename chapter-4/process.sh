crf_learn -f 4 -p 8 -c 3 template ./data/train.txt model>./data/train.log
crf_test -m model ./data/test.txt>./data/test.rst

