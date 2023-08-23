import pandas as pd
import numpy as np

from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.config import Config
from medcat.cdb_maker import CDBMaker
from medcat.cat import CAT

vocab = Vocab()
vocab.add_words('/Users/Maxhi/Python_Tools_AG_Lux/OCR_Reader/ocr_agl/dictionaries/vocab.txt', replace=True)
vocab.make_unigram_table()
vocab.save("vocab.dat")
config = Config()
config.general['spacy_model'] = 'de_core_news_sm'
maker = CDBMaker(config)
# Create an array containing CSV files that will be used to build our CDB
csv_path = ['/Users/Maxhi/Python_Tools_AG_Lux/OCR_Reader/ocr_agl/dictionaries/output_cdb.csv']

# Create your CDB
cdb = maker.prepare_csvs(csv_path, full_build=True)
cdb.save("cdb.dat")
cdb.config.ner['min_name_len'] = 2
cdb.config.ner['upper_case_limit_len'] = 3
cdb.config.general['spell_check'] = True
cdb.config.linking['train_count_threshold'] = 10
cdb.config.linking['similarity_threshold'] = 0.3
cdb.config.linking['train'] = True
cdb.config.linking['disamb_length_limit'] = 5
cdb.config.general['full_unlink'] = True