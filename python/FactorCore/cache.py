import redis
import pyarrow
import uuid
import logging

from typing import List

from . import stock_data
from .utils import MeasureTime

logger = logging.getLogger(__name__)

class DataCache():

    def __init__(self, host):
        port = 6379
        self.redis_cache = redis.Redis(host=host, port=port, decode_responses=False)
        self.redis_cache_decode = redis.Redis(host=host, port=port, decode_responses=True)

    def flush(self):
        self.redis_cache.flushdb()

    def clear_results(self):
        self.redis_cache.unlink("latestcombination")
        self.redis_cache.unlink(self.opt_score_sorted_set_key)
        self.redis_cache.unlink(self.st_score_sorted_set_key)
        self.redis_cache.unlink(self.mean_score_sorted_set_key)
        self.redis_cache.unlink(self.score_comparability_sorted_set_key)
        for key in self.redis_cache.keys("metrics:*"):
            self.redis_cache.unlink(key)
        for key in self.redis_cache.keys("timescale_scores:*"):
            self.redis_cache.unlink(key)

    def set_stock_data(self, data):
        self.redis_cache.hmset("stockdata", data.serialize())

        pyarrow_context = pyarrow.default_serialization_context()
        self.redis_cache.set("factornames", pyarrow_context.serialize(data.get_factor_names()).to_buffer().to_pybytes())
        self.redis_cache.set("daterange", pyarrow_context.serialize(data.get_date_range()).to_buffer().to_pybytes())

    def get_stock_data(self):

        data = stock_data.StockData()

        serialization = self.redis_cache.hgetall("stockdata")
        serialization = { key.decode('utf-8') : value for key, value in serialization.items() }
        if len(serialization) == 0:
            return data

        data.deserialize(serialization)
        return data

    def get_factor_names(self):
        pyarrow_context = pyarrow.default_serialization_context()
        serialization = self.redis_cache.get("factornames")
        if serialization is None:
            return None
        return pyarrow_context.deserialize(serialization)

    def get_date_range(self):
        pyarrow_context = pyarrow.default_serialization_context()
        serialization = self.redis_cache.get("daterange")
        if serialization is None:
            return None
        return pyarrow_context.deserialize(serialization)

    def set_data_filename(self, filename):
        self.redis_cache.set("data_filename", filename)

    def get_data_filename(self):
        return self.redis_cache_decode.get("data_filename")

    def set_all_settings(self, settings):
        self.redis_cache.hmset("settings", settings)

    def get_all_settings(self):
        return self.redis_cache_decode.hgetall("settings")

    def set_setting(self, key, value):
        self.redis_cache.hset("settings", key, value)
