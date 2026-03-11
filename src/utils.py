def load_tickers(path):

    tickers = []

    with open(path) as f:

        for line in f:

            t = line.strip()

            if t:
                tickers.append(t)

    return tickers
