import pyttsx3
import winsound
import ctypes

class Notifier:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.user32 = ctypes.windll.user32

    def notify_mode_switch(self, mode: str, target: float):
        message = f"Switched to {mode.upper()} mode. Daily target ₹{target:.2f}"
        self._show_alert("Mode Changed", message)
        self._speak(message)

    def notify_trade(self, stock: str, price: float, quantity: int, stoploss: float, target: float, trade_type: str):
        message = (
            f"{trade_type} {quantity} shares of {stock} at ₹{price:.2f}\n"
            f"Target: ₹{target:.2f} | Stoploss: ₹{stoploss:.2f}"
        )
        self._show_alert(f"{trade_type} Signal", message)
        self._speak(f"{trade_type} signal for {stock} at price {price:.2f}")

    def notify_completion(self, trades_made: int, capital: float):
        message = f"Completed {trades_made} trades. Current capital ₹{capital:.2f}"
        self._show_alert("Trading Complete", message)
        self._speak(message)

    def notify_error(self, error_msg: str):
        self._show_alert("Trading Error", error_msg)
        self._speak(f"Error occurred: {error_msg}")

    def _show_alert(self, title: str, message: str):
        try:
            # Simple beep and taskbar flash
            winsound.Beep(1000, 300)
            self.user32.FlashWindow(self.user32.GetForegroundWindow(), True)
        except Exception as e:
            print(f"[NOTIFY ERROR] Failed to show alert: {e}")

    def _speak(self, text: str):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[TTS ERROR] {e}")



