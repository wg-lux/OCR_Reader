import pandas as pd
import numpy as np

from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.config import Config
from medcat.cdb_maker import CDBMaker
from medcat.cat import CAT
import pandas as pd
import numpy as np
import pickle
import seaborn as sns

from matplotlib import pyplot as plt
from medcat.cat import CAT

! pip install medcat==1.5.0
try:
    from medcat.cat import CAT
except:
    print("WARNING: Runtime will restart automatically and please run other cells thereafter.")
    exit()

cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)
cat.create_model_pack("agl_medcat_modelpack")

