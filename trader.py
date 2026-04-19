import json
import random
from math import ceil, floor
from typing import Any

import jsonpickle

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# RootTrader history aggregation mode:
# - "avg": average of all price levels on that side at each timestamp
# - "max": highest price level on that side at each timestamp
# - "min": lowest price level on that side at each timestamp
ROOT_HISTORY_AGGREGATION_MODE = "avg"
ROOT_HISTORY_LOOKBACK = 100
ROOT_HISTORY_STATE_KEY = "order_depth_history"
ROOT_QUOTE_SPREAD = 5
ROOT_TAKE_THROUGH = 2
ROOT_EARLY_BUY_OFFSET = 10
ROOT_BUY_PROFIT_BUFFER = 1
ROOT_SELL_PROFIT_BUFFER = 1
ROOT_POSITION_LIMIT = 80
ROOT_EDGE_SHIFT = 10
ROOT_PRICE_OFFSET = 5
ROOT_TARGET_POSITION = 60
ROOT_POSITION_MIN = 40
ROOT_POSITION_MAX = 80
ROOT_MIN_ORDER_SIZE = 40
ROOT_MAX_ORDER_SIZE = 80
ROOT_ADAPT_WINDOW = 10
ROOT_PRICE_OFFSET_MIN = 0
ROOT_PRICE_OFFSET_MAX = 10
ROOT_EDGE_SHIFT_MIN = 5
ROOT_EDGE_SHIFT_MAX = 15
ROOT_TRADERDATA_MAX_CHARS = 45000
ROOT_HISTORY_MIN_SNAPSHOTS = 25
ROOT_HISTORY_MAX_SNAPSHOTS = 200


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

def _compress_order_depth(order_depth: OrderDepth) -> dict[str, list[list[int]]]:
    return {
        "buy_orders": [[price, quantity] for price, quantity in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)],
        "sell_orders": [[price, quantity] for price, quantity in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])],
    }

def _update_order_depth_history(state_data: dict[str, Any], timestamp: int, order_depths: dict[Symbol, OrderDepth]) -> None:
    snapshot = {
        "timestamp": timestamp,
        "order_depths": {
            symbol: _compress_order_depth(order_depth)
            for symbol, order_depth in order_depths.items()
        },
    }

    history = state_data.get(ROOT_HISTORY_STATE_KEY)
    if not isinstance(history, list):
        history = []
    history.append(snapshot)
    state_data[ROOT_HISTORY_STATE_KEY] = history


def _prune_history_for_traderdata(state_data: dict[str, Any]) -> None:
    history = state_data.get(ROOT_HISTORY_STATE_KEY)
    if not isinstance(history, list) or not history:
        return

    if len(history) > ROOT_HISTORY_MAX_SNAPSHOTS:
        history = history[-ROOT_HISTORY_MAX_SNAPSHOTS:]

    state_data[ROOT_HISTORY_STATE_KEY] = history

    if len(history) <= ROOT_HISTORY_MIN_SNAPSHOTS:
        return

    # Keep trimming a few oldest snapshots only if the serialized payload is still too large.
    while len(history) > ROOT_HISTORY_MIN_SNAPSHOTS and len(jsonpickle.encode(state_data)) > ROOT_TRADERDATA_MAX_CHARS:
        history = history[1:]
        state_data[ROOT_HISTORY_STATE_KEY] = history


def _aggregate_prices(prices: list[int]) -> float | None:
    if not prices:
        return None

    if ROOT_HISTORY_AGGREGATION_MODE == "max":
        return float(max(prices))
    if ROOT_HISTORY_AGGREGATION_MODE == "min":
        return float(min(prices))
    return float(sum(prices) / len(prices))


def _history_series(state_data: dict[str, Any], product: str, side: str, current_timestamp: int, limit: int = ROOT_HISTORY_LOOKBACK) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    history = state_data.get(ROOT_HISTORY_STATE_KEY, {})
    if not isinstance(history, list):
        return series

    for snapshot in reversed(history):
        try:
            timestamp = snapshot.get("timestamp")
            if not isinstance(timestamp, int) or timestamp > current_timestamp:
                continue
            order_depths = snapshot.get("order_depths", {})
            product_depth = order_depths.get(product)
            if not isinstance(product_depth, dict):
                continue

            levels = product_depth.get(side, [])
            if not isinstance(levels, list):
                continue

            prices = [level[0] for level in levels if isinstance(level, list) and len(level) >= 1 and isinstance(level[0], (int, float))]
            aggregated_price = _aggregate_prices([int(price) for price in prices])
            if aggregated_price is None:
                continue

            series.append((timestamp, aggregated_price))
            if len(series) >= limit:
                break
        except Exception:
            continue

    return list(reversed(series))


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
    def __init__(self, name: str, state: TradingState, trader_data: dict[str, Any]):
        super().__init__(name, state, trader_data, "INTARIAN_PEPPER_ROOT")
        self._init_adaptive_params()
        self.price_offset = self.adaptive_state["trial_params"]["price_offset"]
        self.edge_shift = self.adaptive_state["trial_params"]["edge_shift"]

    def _init_adaptive_params(self) -> None:
        adaptive = self.product_state.get("adaptive")
        if not isinstance(adaptive, dict):
            adaptive = {}

        if "trial_params" not in adaptive:
            adaptive["accepted_params"] = {
                "price_offset": ROOT_PRICE_OFFSET,
                "edge_shift": ROOT_EDGE_SHIFT,
            }
            adaptive["trial_params"] = dict(adaptive["accepted_params"])
            adaptive["accepted_score"] = None
            adaptive["trial_score"] = 0
            adaptive["trial_step_count"] = 0
            adaptive["step_size"] = 1.0
            adaptive["trial_index"] = 0

        self.product_state["adaptive"] = adaptive
        self.adaptive_state = adaptive
        self._advance_adaptive_trial()

    def _start_new_trial(self, base_params: dict[str, float], step_size: float) -> None:
        rng = random.Random(self.state.timestamp + self.adaptive_state.get("trial_index", 0) * 9973)

        offset_delta = rng.choice([-1, 0, 1]) * step_size
        shift_delta = rng.choice([-1, 0, 1]) * step_size

        trial_params = {
            "price_offset": _clamp(
                base_params["price_offset"] + offset_delta,
                ROOT_PRICE_OFFSET_MIN,
                ROOT_PRICE_OFFSET_MAX,
            ),
            "edge_shift": _clamp(
                base_params["edge_shift"] + shift_delta,
                ROOT_EDGE_SHIFT_MIN,
                ROOT_EDGE_SHIFT_MAX,
            ),
        }

        self.adaptive_state["trial_params"] = trial_params
        self.adaptive_state["trial_score"] = 0
        self.adaptive_state["trial_index"] = self.adaptive_state.get("trial_index", 0) + 1
        self.price_offset = trial_params["price_offset"]
        self.edge_shift = trial_params["edge_shift"]

    def _advance_adaptive_trial(self) -> None:
        adaptive = self.adaptive_state
        step_count = adaptive.get("trial_step_count", 0) + 1
        adaptive["trial_step_count"] = step_count

        trial_score = adaptive.get("trial_score", 0)

        if step_count < ROOT_ADAPT_WINDOW:
            trial_params = adaptive.get("trial_params", adaptive.get("accepted_params", {}))
            self.price_offset = trial_params.get("price_offset", ROOT_PRICE_OFFSET)
            self.edge_shift = trial_params.get("edge_shift", ROOT_EDGE_SHIFT)
            return

        accepted_params = adaptive.get("accepted_params", {
            "price_offset": ROOT_PRICE_OFFSET,
            "edge_shift": ROOT_EDGE_SHIFT,
        })
        accepted_score = adaptive.get("accepted_score")

        if accepted_score is None or trial_score >= accepted_score:
            adaptive["accepted_params"] = dict(adaptive.get("trial_params", accepted_params))
            adaptive["accepted_score"] = trial_score
            step_size = max(0.5, adaptive.get("step_size", 1.0) * 0.95)
        else:
            step_size = min(3.0, adaptive.get("step_size", 1.0) * 1.05)

        adaptive["step_size"] = step_size
        adaptive["trial_step_count"] = 0
        self._start_new_trial(adaptive["accepted_params"], step_size)
        self.price_offset = self.adaptive_state["trial_params"]["price_offset"]
        self.edge_shift = self.adaptive_state["trial_params"]["edge_shift"]

    def market_mid(self):
        if self.mid_wall is not None:
            return float(self.mid_wall)
        return super().market_mid()

    def microprice(self):
        return self.market_mid()

    def _inventory_shift(self) -> float:
        position = self.state.position.get(self.product, 0)
        clamped_position = max(ROOT_POSITION_MIN, min(ROOT_POSITION_MAX, position))
        shift = ((ROOT_TARGET_POSITION - clamped_position) / (ROOT_POSITION_MAX - ROOT_TARGET_POSITION)) * self.edge_shift
        return shift

    def _take_buy_limit(self, fair_value: float) -> int:
        return floor(fair_value - ROOT_BUY_PROFIT_BUFFER)

    def _take_sell_limit(self, fair_value: float) -> int:
        return ceil(fair_value + ROOT_SELL_PROFIT_BUFFER)

    def _make_buy_price(self, fair_value: float) -> int | None:
        profitable_bid = floor(fair_value - ROOT_BUY_PROFIT_BUFFER)
        if self.bid_wall is None:
            return profitable_bid

        bid_price = self.bid_wall + 1
        if self.ask_wall is not None:
            bid_price = min(bid_price, self.ask_wall - 1)

        bid_price = min(bid_price, profitable_bid)
        if self.bid_wall is not None and bid_price <= self.bid_wall:
            return None
        return bid_price

    def _make_sell_price(self, fair_value: float) -> int | None:
        profitable_ask = ceil(fair_value + ROOT_SELL_PROFIT_BUFFER)
        if self.ask_wall is None:
            return profitable_ask

        ask_price = self.ask_wall - 1
        if self.bid_wall is not None:
            ask_price = max(ask_price, self.bid_wall + 1)

        ask_price = max(ask_price, profitable_ask)
        if self.ask_wall is not None and ask_price >= self.ask_wall:
            return None
        return ask_price

    def _takeable_buy_volume(self, limit_price: int) -> int:
        volume = 0
        for price, quantity in sorted(self.mk_sell_orders.items(), key=lambda item: item[0]):
            if price > limit_price:
                break
            volume += quantity
        return volume

    def _takeable_sell_volume(self, limit_price: int) -> int:
        volume = 0
        for price, quantity in sorted(self.mk_buy_orders.items(), key=lambda item: item[0], reverse=True):
            if price < limit_price:
                break
            volume += quantity
        return volume

    def _quote_sizes(self) -> tuple[int, int]:
        position = self.state.position.get(self.product, 0)
        clamped_position = max(ROOT_POSITION_MIN, min(ROOT_POSITION_MAX, position))
        size_range = ROOT_MAX_ORDER_SIZE - ROOT_MIN_ORDER_SIZE

        buy_ratio = (ROOT_POSITION_MAX - clamped_position) / (ROOT_POSITION_MAX - ROOT_POSITION_MIN)
        sell_ratio = (clamped_position - ROOT_POSITION_MIN) / (ROOT_POSITION_MAX - ROOT_POSITION_MIN)

        bid_size = min(self.max_buy_size, max(0, round(ROOT_MIN_ORDER_SIZE + size_range * buy_ratio)))
        ask_size = min(self.max_sell_size, max(0, round(ROOT_MIN_ORDER_SIZE + size_range * sell_ratio)))
        return bid_size, ask_size
    
    def get_orders(self):
        current_fair = self.microprice()
        if current_fair is None:
            return {self.name: self.orders}

        self._advance_adaptive_trial()
        position = self.state.position.get(self.product, 0)
        bid_size, ask_size = self._quote_sizes()

        if position < ROOT_TARGET_POSITION:
            if position <= ROOT_POSITION_MIN:
                buy_limit = ceil(current_fair + ROOT_EARLY_BUY_OFFSET)
            else:
                buy_limit = self._take_buy_limit(current_fair)
            take_buy_size = min(self.max_buy_size, self._takeable_buy_volume(buy_limit))
            if take_buy_size > 0:
                self.bid(buy_limit, take_buy_size)

            make_buy_price = self._make_buy_price(current_fair)
            make_buy_size = min(self.max_buy_size, bid_size)
            if make_buy_price is not None and make_buy_size > 0:
                self.bid(make_buy_price, make_buy_size)
        elif position > ROOT_TARGET_POSITION:
            sell_limit = self._take_sell_limit(current_fair)
            take_sell_size = min(self.max_sell_size, self._takeable_sell_volume(sell_limit))
            if take_sell_size > 0:
                self.ask(sell_limit, take_sell_size)

            make_sell_price = self._make_sell_price(current_fair)
            make_sell_size = min(self.max_sell_size, ask_size)
            if make_sell_price is not None and make_sell_size > 0:
                self.ask(make_sell_price, make_sell_size)
        else:
            buy_limit = self._take_buy_limit(current_fair)
            sell_limit = self._take_sell_limit(current_fair)
            take_buy_size = min(self.max_buy_size, self._takeable_buy_volume(buy_limit))
            take_sell_size = min(self.max_sell_size, self._takeable_sell_volume(sell_limit))
            if take_buy_size > 0:
                self.bid(buy_limit, take_buy_size)
            if take_sell_size > 0:
                self.ask(sell_limit, take_sell_size)

            make_buy_price = self._make_buy_price(current_fair)
            make_sell_price = self._make_sell_price(current_fair)
            make_buy_size = min(self.max_buy_size, bid_size)
            make_sell_size = min(self.max_sell_size, ask_size)
            if make_buy_price is not None and make_buy_size > 0:
                self.bid(make_buy_price, make_buy_size)
            if make_sell_price is not None and make_sell_size > 0:
                self.ask(make_sell_price, make_sell_size)

        executed_volume = 0
        for trade in self.state.own_trades.get(self.name, []):
            executed_volume += abs(trade.quantity)
        self.adaptive_state["trial_score"] = self.adaptive_state.get("trial_score", 0) + executed_volume

        return {self.name: self.orders}

class Trader:

    def run(self, state: TradingState):
        result:dict[str,list[Order]] = {}
        try:
            new_trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}
            if not isinstance(new_trader_data, dict):
                new_trader_data = {}
        except Exception:
            new_trader_data = {}

        _update_order_depth_history(new_trader_data, state.timestamp, state.order_depths)
        _prune_history_for_traderdata(new_trader_data)

        product_traders = {
            "ASH_COATED_OSMIUM": OsmiumTrader,
            "INTARIAN_PEPPER_ROOT": RootTrader,
        }

        result, conversions = {}, 0
        for symbol, product_trader in product_traders.items():
            if symbol in state.order_depths:
                trader = product_trader(symbol, state, new_trader_data)
                result.update(trader.get_orders())

        try:
            final_trader_data = jsonpickle.encode(new_trader_data)
        except Exception:
            final_trader_data = ""
        logger.flush(state, result, conversions, final_trader_data)

        return result, conversions, final_trader_data
