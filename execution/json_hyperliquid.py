import json
import os
from decimal import Decimal
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.core.data_type.common import OrderType, PositionMode

class JsonHyperliquid(ScriptStrategyBase):
    # --- Configuration ---
    # 1. Path to signal file (inside Docker)
    SIGNAL_FILE = "/signals/signal.json"
    
    # 2. Connector: use TESTNET first.
    #    (Change to "hyperliquid_perpetual" later for real money)
    EXCHANGE = "hyperliquid_perpetual_testnet"
    
    # 3. Markets to check (Hyperliquid uses "USD", not "USDT")
    #    Map the JSON name (ATOM-USDT) to the Exchange name (ATOM-USD)
    markets = {
        EXCHANGE: {}
    }

    last_processed_timestamp = None

    def on_tick(self):
        if not os.path.exists(self.SIGNAL_FILE):
            return

        try:
            with open(self.SIGNAL_FILE, 'r') as f:
                data = json.load(f)

            signal_ts = data.get("timestamp")
            
            # Only process if haven't seen this timestamp before
            if signal_ts != self.last_processed_timestamp:
                self.logger().info(f"🆕 NEW SIGNAL: {signal_ts}")
                self.process_signal(data)
                self.last_processed_timestamp = signal_ts

        except Exception as e:
            # Ignore errors while reading (e.g. if file is being saved)
            pass

    def process_signal(self, data):
        notional_usd = Decimal(str(data.get("notional_usd", 100)))
        weights = data["weights"]
        market_map = data["markets"]

        for asset_key, weight in weights.items():
            # 1. Get pair from JSON (e.g. "ATOM-USDT")
            raw_pair = market_map.get(asset_key)
            if not raw_pair: continue

            # 2. Convert to Hyperliquid format (USDT -> USD)
            hb_pair = raw_pair.replace("USDT", "USD")

            if hb_pair not in self.markets[self.EXCHANGE]:
                continue

            # 3. Calculate Trade
            weight_dec = Decimal(str(weight))
            is_buy = weight_dec > 0
            target_value = notional_usd * abs(weight_dec)
            
            # 4. Get Price
            price = self.connectors[self.EXCHANGE].get_price(hb_pair, is_buy)
            if price is None:
                self.logger().warning(f"❌ No price for {hb_pair}")
                continue

            amount = target_value / price

            # 5. Execute Order
            action = "BUY" if is_buy else "SELL"
            self.logger().info(f"🚀 EXECUTING: {action} {hb_pair} (${target_value:.2f})")
            
            if is_buy:
                self.buy(self.EXCHANGE, hb_pair, amount, OrderType.MARKET)
            else:
                self.sell(self.EXCHANGE, hb_pair, amount, OrderType.MARKET)