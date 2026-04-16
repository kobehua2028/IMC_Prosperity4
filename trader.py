import json
from math import ceil, floor
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

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
    def __init__(self, name: str, state: TradingState, trader_data: dict[str, Any], product=None):
        self.orders = []

        self.name = name
        self.state = state
        self.trader_data = trader_data
        self.product = name if not product else product
        self.product_group = name
        self.product_state = self.trader_data.get(self.product, {})
        if not isinstance(self.product_state, dict):
            self.product_state = {}
        self.trader_data[self.product] = self.product_state

        self.position_limit = 80
        self.initial_position = self.state.position.get(self.product, 0)
        self.expected_position = self.initial_position

        self.mk_buy_orders, self.mk_sell_orders = self.get_order_book()
        self.bid_wall, self.mid_wall, self.ask_wall = self.get_walls()
        self.total_buy_volume, self.total_sell_volume = self.total_trading_volume()
        self.max_buy_size, self.max_sell_size = self.max_trade_size()

    # splits the order book into buy and sell orders for current product
    def get_order_book(self):
        order_depth = self.state.order_depths.get(self.name)
        if order_depth is None:
            return {}, {}

        buy_orders = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
        sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}

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

    def bid(self, price: int, quantity: int):
        quantity = min(abs(quantity), self.max_buy_size)
        if quantity <= 0:
            return
        self.orders.append(Order(self.product, price, abs(quantity)))
        self.max_buy_size -= abs(quantity)
        self.expected_position += abs(quantity)

    def ask(self, price: int, quantity: int):
        quantity = min(abs(quantity), self.max_sell_size)
        if quantity <= 0:
            return
        self.orders.append(Order(self.product, price, -1 * abs(quantity)))
        self.max_sell_size -= abs(quantity)
        self.expected_position -= abs(quantity)

    def market_mid(self):
        if self.bid_wall is not None and self.ask_wall is not None:
            return (self.bid_wall + self.ask_wall) / 2
        if self.bid_wall is not None:
            return float(self.bid_wall)
        if self.ask_wall is not None:
            return float(self.ask_wall)
        return None

    def microprice(self):
        if self.bid_wall is None or self.ask_wall is None:
            return self.market_mid()

        bid_volume = self.mk_buy_orders.get(self.bid_wall, 0)
        ask_volume = self.mk_sell_orders.get(self.ask_wall, 0)
        if bid_volume + ask_volume == 0:
            return self.market_mid()

        return (self.bid_wall * ask_volume + self.ask_wall * bid_volume) / (bid_volume + ask_volume)

    def get_orders(self):
        return []

class OsmiumTrader(BaseTrader):
    FAIR_VALUE = 10000
    TAKE_EDGE = 1
    MAKE_EDGE = 2
    PASSIVE_SIZE = 20
    EWMA_ALPHA = 0.1
    EWMA_WEIGHT = 0.5
    INVENTORY_SKEW = 0.02

    def __init__(self, name: str, state: TradingState, trader_data: dict[str, Any]):
        super().__init__(name, state, trader_data, "ASH_COATED_OSMIUM")

    def fair_value(self):
        book_fair = self.microprice()
        if book_fair is None:
            return self.FAIR_VALUE - self.initial_position * self.INVENTORY_SKEW

        previous_ewma = self.product_state.get("ewma", book_fair)
        ewma = self.EWMA_ALPHA * book_fair + (1 - self.EWMA_ALPHA) * previous_ewma
        self.product_state["ewma"] = round(ewma, 4)

        model_fair = (1 - self.EWMA_WEIGHT) * self.FAIR_VALUE + self.EWMA_WEIGHT * ewma
        return model_fair - self.initial_position * self.INVENTORY_SKEW
    
    def get_orders(self):
        fair = self.fair_value()

        if self.bid_wall is None and self.ask_wall is None:
            logger.print("No orders in the book")
            return {self.name: self.orders}

        # Take clear dislocations around the stable 10,000 anchor.
        for price, quantity in self.mk_sell_orders.items():
            if price <= fair - self.TAKE_EDGE:
                self.bid(price, quantity)
            elif self.expected_position < 0 and price <= fair:
                self.bid(price, min(quantity, abs(self.expected_position)))

        for price, quantity in self.mk_buy_orders.items():
            if price >= fair + self.TAKE_EDGE:
                self.ask(price, quantity)
            elif self.expected_position > 0 and price >= fair:
                self.ask(price, min(quantity, self.expected_position))

        bid_ceiling = floor(fair - self.MAKE_EDGE)
        ask_floor = ceil(fair + self.MAKE_EDGE)

        if self.bid_wall is not None:
            bid_price = min(self.bid_wall + 1, bid_ceiling)
        else:
            bid_price = bid_ceiling

        if self.ask_wall is not None:
            ask_price = max(self.ask_wall - 1, ask_floor)
        else:
            ask_price = ask_floor

        if self.ask_wall is not None:
            bid_price = min(bid_price, self.ask_wall - 1)
        if self.bid_wall is not None:
            ask_price = max(ask_price, self.bid_wall + 1)

        if bid_price < ask_price:
            bid_size = min(self.max_buy_size, self.PASSIVE_SIZE + max(0, -self.expected_position))
            ask_size = min(self.max_sell_size, self.PASSIVE_SIZE + max(0, self.expected_position))
            self.bid(bid_price, bid_size)
            self.ask(ask_price, ask_size)

        return {self.name: self.orders}

class RootTrader(BaseTrader):
    DRIFT_PER_TIMESTAMP = 0.001
    ROUND_END_TIMESTAMP = 999900
    TAKE_EDGE = 2
    PASSIVE_LOOKAHEAD = 20

    def __init__(self, name: str, state: TradingState, trader_data: dict[str, Any]):
        super().__init__(name, state, trader_data, "INTARIAN_PEPPER_ROOT")
    
    def get_orders(self):
        current_fair = self.microprice()
        if current_fair is None:
            return {self.name: self.orders}

        remaining_drift = max(0, self.ROUND_END_TIMESTAMP - self.state.timestamp) * self.DRIFT_PER_TIMESTAMP
        exit_fair = current_fair + remaining_drift

        for price, quantity in self.mk_sell_orders.items():
            if price <= exit_fair - self.TAKE_EDGE:
                self.bid(price, quantity)
            elif self.expected_position < 0 and price <= current_fair + self.PASSIVE_LOOKAHEAD:
                self.bid(price, min(quantity, abs(self.expected_position)))

        for price, quantity in self.mk_buy_orders.items():
            if price >= exit_fair + self.TAKE_EDGE:
                self.ask(price, quantity)

        if self.max_buy_size > 0 and self.bid_wall is not None:
            bid_ceiling = floor(current_fair + min(self.PASSIVE_LOOKAHEAD, remaining_drift) - self.TAKE_EDGE)
            bid_price = min(self.bid_wall + 1, bid_ceiling)
            if self.ask_wall is not None:
                bid_price = min(bid_price, self.ask_wall - 1)
            if bid_price > 0:
                self.bid(bid_price, self.max_buy_size)

        return {self.name: self.orders}

class Trader:

    def run(self, state: TradingState):
        result:dict[str,list[Order]] = {}
        try:
            new_trader_data = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(new_trader_data, dict):
                new_trader_data = {}
        except Exception:
            new_trader_data = {}

        product_traders = {
            "ASH_COATED_OSMIUM": OsmiumTrader,
            "INTARIAN_PEPPER_ROOT": RootTrader,
        }

        result, conversions = {}, 0
        for symbol, product_trader in product_traders.items():
            if symbol in state.order_depths:
                trader = product_trader(symbol, state, new_trader_data)
                result.update(trader.get_orders())

        try: final_trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        except: final_trader_data = ''
        logger.flush(state, result, conversions, final_trader_data)

        return result, conversions, final_trader_data
