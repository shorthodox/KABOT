import yaml
import os
import datetime
import json
from src.notifier import Notifier 

CONFIG_PATH = os.path.join("config", "trading_config.yaml")
LOG_PATH = os.path.join("logs", "trade_log.txt")

class TradeSimulator:
    def __init__(self):
        self.config: dict = {}
        self.load_config()
        self.capital = self.config["initial_capital"]
        self.daily_target = 0
        self.current_mode = "base"
        self.trade_count = 0
        self.trades_today = []
        self.notifier = Notifier()

    def load_config(self):
        with open(CONFIG_PATH, "r") as f:
            loaded = yaml.safe_load(f)
            if not isinstance(loaded, dict):
                raise ValueError("Config file must contain a dictionary at the top level.")
            self.config = loaded

    def set_mode(self, mode: str):
        if mode not in self.config["modes"]:
            raise ValueError("Invalid mode. Choose from base, moderate, beast.")
        self.current_mode = mode
        self.daily_target = (
            self.capital * self.config["modes"][mode]["daily_target_percent"] / 100
        )
        self.trade_count = 0
        self.trades_today = []
        print(f"[INFO] Mode set to {mode.upper()} | Target ₹{self.daily_target:.2f}")
        self.notifier.notify_mode_switch(mode, self.daily_target)

    def calculate_trade_amount(self) -> float:
        risk_percent = self.config["risk_per_trade_percent"]
        return self.capital * risk_percent / 100

    def log_trade(self, trade: dict):
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def suggest_buy(self, stock: str, current_price: float):
        if self.trade_count >= self.config["modes"][self.current_mode]["max_trades_per_day"]:
            print("[INFO] Max trades reached for the day.")
            return

        amount_to_invest = self.calculate_trade_amount()
        quantity = int(amount_to_invest / current_price)

        if quantity == 0:
            print(f"[WARN] Not enough capital to buy even 1 share of {stock}.")
            return

        stoploss = current_price - (current_price * self.config["default_stoploss_percent"] / 100)
        target = current_price + (self.daily_target / quantity)

        trade = {
            "timestamp": str(datetime.datetime.now()),
            "mode": self.current_mode,
            "type": "BUY",
            "stock": stock,
            "price": current_price,
            "quantity": quantity,
            "stoploss": round(stoploss, 2),
            "target_price": round(target, 2),
            "capital_used": round(quantity * current_price, 2)
        }

        self.trades_today.append(trade)
        self.trade_count += 1
        self.log_trade(trade)

        # Console notification
        print(f"\n[TRADE SUGGESTION]")
        print(f"BUY {quantity} shares of {stock} at ₹{current_price}")
        print(f"Target: ₹{target:.2f} | Stoploss: ₹{stoploss:.2f}")
        print(f"Capital Used: ₹{quantity * current_price:.2f}")

        # Notifier
        self.notifier.notify_trade(
            stock=stock,
            price=current_price,
            quantity=quantity,
            stoploss=round(stoploss, 2),
            target=round(target, 2),
            trade_type="BUY"
        )

    def record_sell(self, stock: str, sell_price: float):
        for trade in self.trades_today:
            if trade["stock"] == stock and trade["type"] == "BUY":
                pnl = (sell_price - trade["price"]) * trade["quantity"]
                self.capital += pnl

                print(f"\n[SELL RECORDED] {stock} @ ₹{sell_price} | P&L: ₹{pnl:.2f}")
                print(f"New Capital: ₹{self.capital:.2f}")

                # Log SELL manually
                self.log_trade({
                    "timestamp": str(datetime.datetime.now()),
                    "mode": self.current_mode,
                    "type": "SELL",
                    "stock": stock,
                    "price": sell_price,
                    "quantity": trade["quantity"],
                    "pnl": round(pnl, 2),
                    "new_capital": round(self.capital, 2)
                })

                return
        print("[WARN] No matching BUY found for this stock.")





