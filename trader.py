import string
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

STATIC_PRODUCT = "TOMATOES"
DYNAMIC_PRODUCT = "EMERALDS"

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
    def __init__(self, name: str, state: TradingState, print_logs: bool = True, trader_data: str, product=None):
        self.orders = []

        self.name = name
        self.state = state
        self.print_logs = print_logs
        self.trader_data = trader_data
        self.product = name if not product else product

        self.position_limit = 100
        self.initial_position = self.state.position.get(self.product, 0)
        self.expected_position = self.initial_position

        self.mk_buy_orders, self.mk_sell_orders = get_order_book()
        self.bid_wall, self.mid_wall, self.ask_wall = self.get_walls()
        self.total_buy_volume, self.total_sell_volume = self.total_trading_volume()
        self.max_buy_size, self.max_sell_size = self.max_trade_size()

    # splits the order book into buy and sell orders for current product
    def get_order_book(self):
        mk_buy_orders = self.state.order_depths[self.product].buy_orders
        mk_sell_orders = self.state.order_depths[self.product].sell_orders

        return mk_buy_orders, mk_sell_orders

    # Find the spread of the order book and the bid and ask walls
    def get_walls():
        bid_wall = min(self.mk_buy_orders.keys()) if self.mk_buy_orders else None
        ask_wall = max(self.mk_sell_orders.keys()) if self.mk_sell_orders else None
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

    def log(self, message: str):
        if self.print_logs:
            logger.print(f"{self.name}: {message}")

    def bid(self, price: int, quantity: int):
        quantity = min(quantity, self.max_buy_size)
        self.orders.append(Order(self.product, price, quantity))
        self.log(f"{self.name} bids {quantity} at {price}")
        self.max_buy_size -= quantity
        self.expected_position += quantity

    def ask(self, price: int, quantity: int):
        quantity = min(quantity, self.max_sell_size)
        self.orders.append(Order(self.product, price, quantity))
        self.log(f"{self.name} asks {quantity} at {price}")
        self.max_sell_size -= quantity
        self.expected_position -= quantity

    def get_orders(self):
        return []

class StaticTrader(BaseTrader):
    def __init__(self, name: str, state: TradingState, print_logs: bool = True, trader_data: str):
        super().__init__(name, state, print_logs, trader_data, STATIC_PRODUCT)
    
    def get_orders(self):
        # Taking
        for price, quantity in self.mk_sell_orders.items():
            if price < self.mid_wall and quantity < 0:
                self.bid(price, -quantity)
        
        for price, quantity in self.mk_buy_orders.items():
            if price > self.mid_wall and quantity > 0:
                self.ask(price, quantity)


        # Making


class Trader:

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        result = {}
        conversions = 0
        trader_data = ""

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10  # Participant should calculate this value
            logger.print("Acceptable price : " + str(acceptable_price))
            logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    logger.print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    logger.print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data