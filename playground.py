import sys
sys.path.append('../')



from codes.data_reader import *
from transformers import AlbertTokenizer

processor = jade_processor(AlbertTokenizer, args.max_seq_length)
reader = jade_reader(args.batch_size)
