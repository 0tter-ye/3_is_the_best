import MetaTrader5 as mt5

print(f"MetaTrader5 version: {mt5.__version__}")
print(f"Module file: {mt5.__file__}")
print(f"Has account_balance: {hasattr(mt5, 'account_balance')}")
print(f"Has account_info: {hasattr(mt5, 'account_info')}")
print(f"Dir(mt5): {dir(mt5)[:10]}")

if mt5.initialize():
    print("MT5 initialized successfully")
    if hasattr(mt5, 'account_info'):
        account_info = mt5.account_info()
        if account_info:
            print(f"Account balance via account_info: {account_info.balance}")
        else:
            print("Failed to get account_info")
    else:
        print("account_info not available")
else:
    print("MT5 initialization failed")