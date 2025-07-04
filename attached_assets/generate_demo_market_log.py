import logging
import random
from datetime import datetime
import time

logging.basicConfig(filename='market_data_debug.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

symbols = [
    ('^NSEI', 'NIFTY'),
    ('^NSEBANK', 'BANKNIFTY'),
    ('RELIANCE.NS', 'RELIANCE'),
    ('TCS.NS', 'TCS'),
    ('HDFCBANK.NS', 'HDFC'),
    ('INFY.NS', 'INFOSYS'),
    ('ICICIBANK.NS', 'ICICI'),
]

def generate_random_ohlc(base):
    ltp = round(base + random.uniform(-100, 100), 2)
    high = ltp + random.uniform(0, 50)
    low = ltp - random.uniform(0, 50)
    vol = random.randint(10000, 1000000)
    return ltp, round(high, 2), round(low, 2), vol

base_prices = {
    '^NSEI': 24500,
    '^NSEBANK': 52000,
    'RELIANCE.NS': 2900,
    'TCS.NS': 3900,
    'HDFCBANK.NS': 1700,
    'INFY.NS': 1500,
    'ICICIBANK.NS': 1200,
}

if __name__ == '__main__':
    for _ in range(5):  # Generate 5 rounds of data
        for symbol, _ in symbols:
            ltp, high, low, vol = generate_random_ohlc(base_prices[symbol])
            logging.debug(f'{symbol}: LTP={ltp}, HIGH={high}, LOW={low}, VOL={vol}')
        time.sleep(1)
    print('Demo market_data_debug.log generated!') 