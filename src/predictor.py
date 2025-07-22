import random
import yfinance as yf
from typing import Optional

class MockPredictor:
    def __init__(self):
        self.modes = ["BUY", "SELL", "HOLD"]
        
    def get_current_price(self, stock: str) -> Optional[float]:
        """Get real market price with fallback to mock data"""
        try:
            # Ensure proper NSE symbol format
            if not stock.endswith('.NS'):
                stock += '.NS'
                
            ticker = yf.Ticker(stock)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                return round(data['Close'].iloc[-1], 2)
            
            # Fallback to mock price if API fails
            return round(random.uniform(50, 2000), 2)
            
        except Exception:
            # Fallback to mock price if any error occurs
            return round(random.uniform(50, 2000), 2)

    def predict(self, stock: str, features=None) -> dict:
        """Generate mock prediction"""
        decision = random.choices(
            population=self.modes,
            weights=[0.4, 0.3, 0.3],  # Slight bias toward BUY
            k=1
        )[0]
        
        return {
            "stock": stock,
            "signal": decision,
            "confidence": round(random.uniform(60, 95), 2)
        }
   