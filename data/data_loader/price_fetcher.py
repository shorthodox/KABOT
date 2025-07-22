import yfinance as yf

def get_live_price(stock_symbol):
    try:
        # Try raw ticker first
        ticker = yf.Ticker(stock_symbol)
        price = ticker.info.get("regularMarketPrice", None)

        # If price is None, try with '.NS'
        if price is None:
            ticker = yf.Ticker(f"{stock_symbol}.NS")
            price = ticker.info.get("regularMarketPrice", None)

        if price is None:
            raise ValueError("'regularMarketPrice' not found.")

        return round(price, 2)

    except Exception as e:
        print(f"[ERROR] Failed to fetch price for {stock_symbol}: {e}")
        return None

