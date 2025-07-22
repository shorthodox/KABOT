from src.simulator import TradeSimulator 
from src.predictor import MockPredictor
from src.notifier import Notifier
import yaml
import time
import os

CONFIG_PATH = os.path.join("config", "trading_config.yaml")

def load_watchlist():
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            if not config or "watchlist" not in config:
                raise ValueError("Watchlist not found or config is empty")

            if not isinstance(config, dict):
                raise ValueError("Config is not a dictionary")

            watchlist = config.get("watchlist", [])
            if isinstance(watchlist, dict):
                return watchlist.get("penny_stocks", []) + watchlist.get("midrange_stocks", [])
            elif isinstance(watchlist, list):
                return watchlist
            else:
                raise ValueError("Invalid format for watchlist")
    except Exception as e:
        raise RuntimeError(f"Failed to load watchlist: {str(e)}")

def run_trading_day():
    print("=== Welcome to Viren Trada Lite: Auto Run Mode (Single Scan) ===")
    
    try:
        simulator = TradeSimulator()
        predictor = MockPredictor()
        notifier = Notifier()

        # Select mode
        while True:
            mode = input("Select mode (base/moderate/beast): ").strip().lower()
            if mode in ['base', 'moderate', 'beast']:
                break
            print("Invalid mode. Choose base/moderate/beast.")

        # Notify mode switch
        simulator.set_mode(mode)
        try:
            notifier.notify_mode_switch(mode, simulator.daily_target)
        except Exception as e:
            print(f"[NOTIFY WARNING] Mode switch failed: {str(e)}")

        # Enter capital
        while True:
            try:
                capital = float(input("Enter your current total capital (₹): "))
                if capital > 0:
                    break
                print("Capital must be positive.")
            except ValueError:
                print("Enter a valid number.")

        # Config from selected mode
        mode_config = simulator.config["modes"].get(mode, {})
        profit_percent = mode_config.get("profit_percent", 0.02)
        max_trades = mode_config.get("max_trades_per_day", 5)
        per_trade_capital = capital / max_trades

        stocks = load_watchlist()
        print(f"\n🧾 Running scan on {len(stocks)} stocks in '{mode}' mode...\n")

        trades_made = 0

        for stock in stocks:
            if trades_made >= max_trades:
                print("\n[INFO] Trade limit reached for the day.")
                break

            print(f"[CHECKING] {stock}")
            try:
                # Try to fetch stock price with retries
                price = None
                for attempt in range(3):
                    try:
                        price = predictor.get_current_price(stock)
                        if price is not None:
                            break
                    except Exception as e:
                        print(f"[PRICE WARNING] Attempt {attempt + 1}: {str(e)}")
                        time.sleep(1)

                if price is None:
                    print(f"[WARNING] Could not fetch price for {stock}. Skipping...\n")
                    continue

                prediction = predictor.predict(stock)
                signal = prediction.get("signal", "HOLD")
                confidence = prediction.get("confidence", 0)
                print(f"Signal: {signal} | Confidence: {confidence}%")

                quantity = max(int(per_trade_capital // price), 1)
                total = quantity * price

                if signal == "BUY":
                    stoploss = round(price * 0.98, 2)
                    target = round(price * (1 + profit_percent), 2)

                    print(f"[ENTRY] BUY {quantity} shares of {stock} at ₹{price:.2f} EACH (₹{total:.2f})")
                    print(f"[STOPLOSS] ₹{stoploss:.2f} | [TARGET] ₹{target:.2f}\n")

                    try:
                        notifier.notify_trade(
                            stock=stock,
                            price=price,
                            quantity=quantity,
                            stoploss=stoploss,
                            target=target,
                            trade_type="BUY"
                        )
                    except Exception as e:
                        print(f"[NOTIFY WARNING] Trade alert failed: {str(e)}")

                    simulator.suggest_buy(stock, price)
                    trades_made += 1

                elif signal == "SELL":
                    stoploss = round(price * 1.02, 2)
                    target = round(price * (1 - profit_percent), 2)

                    print(f"[ENTRY] SELL {quantity} shares of {stock} at ₹{price:.2f} EACH (₹{total:.2f})")
                    print(f"[STOPLOSS] ₹{stoploss:.2f} | [TARGET] ₹{target:.2f}\n")

                    try:
                        notifier.notify_trade(
                            stock=stock,
                            price=price,
                            quantity=quantity,
                            stoploss=stoploss,
                            target=target,
                            trade_type="SELL"
                        )
                    except Exception as e:
                        print(f"[NOTIFY WARNING] Trade alert failed: {str(e)}")

                    simulator.record_sell(stock, price)
                    trades_made += 1

                else:
                    print(f"[INFO] HOLD signal for {stock}\n")

                time.sleep(1)

            except Exception as e:
                print(f"[ERROR] {stock}: {str(e)}\n")
                continue

        print(f"\n✅ Scan complete. {trades_made} trade(s) executed.")
        try:
            notifier.notify_completion(trades_made, simulator.capital)
        except Exception as e:
            print(f"[NOTIFY WARNING] Completion alert failed: {str(e)}")

    except Exception as e:
        error_msg = f"Trading day ended: {str(e)}"
        print(f"[CRITICAL ERROR] {error_msg}")
        try:
            notifier.notify_error(error_msg)
        except Exception as notify_err:
            print(f"[NOTIFY ERROR] Couldn't send error notification: {str(notify_err)}")

if __name__ == "__main__":
    run_trading_day()



