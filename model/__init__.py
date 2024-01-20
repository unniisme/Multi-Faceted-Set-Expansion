import logging
from time import time
logfile = f"log/model.{time()}.log"
FORMAT = "[%(levelname)-8s][%(asctime)s][%(filename)s:%(lineno)s - %(funcName)13s() ] %(message)s"
logging.basicConfig(format=FORMAT, filename=logfile, encoding='utf-8', level=logging.DEBUG)