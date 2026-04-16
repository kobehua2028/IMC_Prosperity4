import string
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

STATIC_PRODUCT = "ASH_COATED_OSMIUM"
DYNAMIC_PRODUCT = "INTARIAN_PEPPER_ROOT"

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out
logger = Logger()

class BaseTrader:
    def __init__(self, name: str, state: TradingState, trader_data: str, prints: dict, print_logs: bool = True, product=None):
        self.orders = []

        self.name = name
        self.state = state
        self.print_logs = print_logs
        self.trader_data = trader_data
        self.product = name if not product else product
        self.prints = prints
        self.product_group = name

        self.position_limit = 100
        self.initial_position = self.state.position.get(self.product, 0)
        self.expected_position = self.initial_position

        self.mk_buy_orders, self.mk_sell_orders = self.get_order_book()
        self.bid_wall, self.mid_wall, self.ask_wall = self.get_walls()
        self.total_buy_volume, self.total_sell_volume = self.total_trading_volume()
        self.max_buy_size, self.max_sell_size = self.max_trade_size()

    # splits the order book into buy and sell orders for current product
    def get_order_book(self):
        order_depth, buy_orders, sell_orders = {}, {}, {}

        try: order_depth: OrderDepth = self.state.order_depths[self.name]
        except: pass
        try: buy_orders = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
        except: pass
        try: sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except: pass

        return buy_orders, sell_orders

    # Find the spread of the order book and the bid and ask walls
    def get_walls(self):
        bid_wall = max(self.mk_buy_orders.keys()) if self.mk_buy_orders else None
        ask_wall = min(self.mk_sell_orders.keys()) if self.mk_sell_orders else None
        mid_wall = (bid_wall + ask_wall) / 2 if bid_wall and ask_wall else None
        return bid_wall, mid_wall, ask_wall

    def total_trading_volume(self):
        total_buy_volume = 0
        total_sell_volume = 0
        for quantity in self.mk_buy_orders.values():
            total_buy_volume += quantity
        for quantity in self.mk_sell_orders.values():
            total_sell_volume += quantity
        return total_buy_volume, total_sell_volume

    def max_trade_size(self):
        max_buy_size = self.position_limit - self.initial_position
        max_sell_size = self.position_limit + self.initial_position
        return max_buy_size, max_sell_size

    def log(self, kind, message, product_group=None):
        if product_group is None: product_group = self.product_group

        if product_group == 'ORDERS':
            group = self.prints.get(product_group, [])
            group.append({kind: message})
        else:
            group = self.prints.get(product_group, {})
            group[kind] = message

        self.prints[product_group] = group

    def bid(self, price: int, quantity: int):
        quantity = min(quantity, self.max_buy_size)
        self.orders.append(Order(self.product, price, abs(quantity)))
        self.log("BID", f"{self.name} bids {quantity} at {price}")
        self.max_buy_size -= abs(quantity)
        self.expected_position += abs(quantity)

    def ask(self, price: int, quantity: int):
        quantity = min(quantity, self.max_sell_size)
        self.orders.append(Order(self.product, price, -1 * abs(quantity)))
        self.log("ASK", f"{self.name} asks {quantity} at {price}")
        self.max_sell_size -= abs(quantity)
        self.expected_position -= abs(quantity)

    def get_orders(self):
        return []

class StaticTrader(BaseTrader):
    def __init__(self, name: str, state: TradingState, trader_data: str, prints: dict, print_logs: bool = True):
        super().__init__(name, state, trader_data, prints, print_logs, STATIC_PRODUCT)
    
    def get_orders(self):

        if self.mid_wall:
            # Taking
            for price, quantity in self.mk_sell_orders.items():
                if price <= self.mid_wall - 1:
                    self.bid(price, quantity)
                elif price <= self.mid_wall and self.initial_position < 0:
                    self.bid(price, min(abs(quantity), abs(self.initial_position)))
            
            for price, quantity in self.mk_buy_orders.items():
                if price >= self.mid_wall + 1:
                    self.ask(price, quantity)
                elif price >= self.mid_wall and self.initial_position > 0:
                    self.ask(price, min(abs(quantity), abs(self.initial_position)))

            # Making
            bid_price = int(self.bid_wall + 1)
            ask_price = int(self.ask_wall - 1)

            for price, quantity in self.mk_buy_orders.items():
                overbid = price + 1
                if overbid < self.mid_wall and self.initial_position > 0:
                    bid_price = max(bid_price, overbid)
                    break
                elif price < self.mid_wall:
                    bid_price = max(bid_price, price)
                    break
            for price, quantity in self.mk_sell_orders.items():
                underbid = price - 1
                if underbid > self.mid_wall and self.initial_position < 0:
                    ask_price = min(ask_price, underbid)
                    break
                elif price > self.mid_wall:
                    ask_price = min(ask_price, price)
                    break
            self.bid(bid_price, self.max_buy_size)
            self.ask(ask_price, self.max_sell_size)

        return {self.name: self.orders}

class DynamicTrader(BaseTrader):
    def __init__(self, name: str, state: TradingState, trader_data: str, prints: dict, print_logs: bool = True):
        super().__init__(name, state, trader_data, prints, print_logs, DYNAMIC_PRODUCT)
        self.direction = 1  # 1 for long, -1 for short
    
    def get_orders(self):
        # Seems like an assets that only moves in one direction, so we can just keep bidding until we hit the position limit
        if self.mid_wall:
            listed_sell_orders = list(self.mk_sell_orders.items())
            listed_buy_orders = list(self.mk_buy_orders.items())
            i = 0
            while self.max_buy_size > 0 and i < len(listed_sell_orders):
                buy_price, buy_quantity = listed_sell_orders[i]
                self.bid(buy_price, buy_quantity)
                i += 1
            
            j = 0
            while self.max_sell_size > 0 and j < len(listed_buy_orders):
                sell_price, sell_quantity = listed_buy_orders[j]
                if sell_price > self.mid_wall:
                    self.ask(sell_price, sell_quantity)
                j +=1

        return {self.name: self.orders}

class Trader:

    def run(self, state: TradingState):
        result:dict[str,list[Order]] = {}
        new_trader_data = {}
        prints = {
            "GENERAL": {
                "TIMESTAMP": state.timestamp,
                "POSITIONS": state.position
            },
        }

        product_traders = {
            STATIC_PRODUCT: StaticTrader,
            DYNAMIC_PRODUCT: DynamicTrader,
        }

        result, conversions = {}, 0
        for symbol, product_trader in product_traders.items():
            if symbol in state.order_depths:
                trader = product_trader(symbol, state, json.dumps(new_trader_data),
  prints)
                result.update(trader.get_orders())

        try: final_trader_data = json.dumps(new_trader_data)
        except: final_trader_data = ''
        logger.flush(state, result, conversions, final_trader_data)

        return result, conversions, final_trader_data