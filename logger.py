import logging
import datetime

now = datetime.datetime.now().strftime('%m%d%H%M%S')

logging.basicConfig(filename="log/{}.log".format(now), level=logging.DEBUG)
