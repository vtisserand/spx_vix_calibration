import logging

NB_TRADING_HOURS_PER_DAY = 7
NB_SAMPLE_PER_HOUR = 60 / 5  # 5mins ticks for vol estimates
NB_DAYS_PER_YEAR = 252

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger('qHeston')