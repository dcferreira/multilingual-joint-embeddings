from src.utils import files
from src import config
import time

def build_data(data, method_name, dataset, args=()):
    """
    Builds matrices representing the data
    """
    try:
        method = files.import_file('src.word_reps.datasets.' + dataset + '.' + str(method_name))
    except:
        raise NotImplementedError(str(method_name) + ' is unknown (not implemented) method!')
    if data.EN is None:
        init = time.clock()
        method.build_reps(data, *args)
        if config.PRINT_TIMES:
            print str(method_name) + '.build_reps() took:', time.clock() - init
    init = time.clock()
    method.build_data(data, *args)
    if config.PRINT_TIMES:
        print str(method_name) + '.build_data() took:', time.clock() - init

    data.build_reps.append([method_name, dataset, args])