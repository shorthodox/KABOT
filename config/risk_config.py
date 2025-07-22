# config/risk_config.py
RISK_CONFIG = {
    'base': {
        'risk_per_trade': 0.01,  # 1% of capital
        'daily_target': 0.005,    # 0.5% daily target
        'max_trades': 5
    },
    'moderate': {
        'risk_per_trade': 0.02,   # 2% of capital
        'daily_target': 0.01,     # 1% daily target
        'max_trades': 8
    },
    'beast': {
        'risk_per_trade': 0.03,   # 3% of capital
        'daily_target': 0.02,     # 2% daily target
        'max_trades': 12
    }
}

