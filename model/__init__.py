import logging
FORMAT = "[%(levelname)-8s][%(asctime)s][%(filename)s:%(lineno)s - %(funcName)13s() ] %(message)s"
logging.basicConfig(format=FORMAT, filename='log/model.log', encoding='utf-8', level=logging.DEBUG)