import datetime
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.finance import candlestick_ohlc
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from gym import Env, spaces
from gym.utils import seeding

class ESTradingEnv_v2(Env):
    
    _actions = {
        'hold': 0,
        'buy': 1,
        'sell': 2
    }

    _positions = {
        'flat': 0,
        'long': 1,
        'short': -1
    }
    
    def __init__(self, history_length=64, episode_length=20*81, commission=2,
                order_penalty=0, time_penalty=0):
        

        """Initialisation function"""
        self._five_min_data = pd.read_feather('../data/processed/ES_5mintrading.feather')
        self._five_min_data = self._five_min_data[self._five_min_data['date']<'1-1-2018'] #Training Set
        self._history_length = history_length
        self._episode_length = episode_length
        
        self._commission = commission
        self._order_penalty = order_penalty
        self._time_penalty = time_penalty
        
        # We can take 3 actions 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        # Observation consists of history_length bars and 8 features
        # First of the 8 features is the position: 0=flat, 1=long, -1=short
        # Next 5 features are ohlc and the value of the 20 day ema
        # Last 2 features are sin_time and cos_time
        self.observation_space = spaces.Box(low=-9999, high=9999, 
                                            shape=(history_length,8), dtype=np.float32)
        
        self._first_render = True
        self._observation = self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        """Reset the trading environment. Reset rewards, data generator...

        Returns:
            observation (numpy.array): observation of the state
        """
        
        self._iteration = 0
        
        #Find indices of the first bars of each day
        i = self._five_min_data[(self._five_min_data['date'].dt.hour == 9) & \
                 (self._five_min_data['date'].dt.minute == 35)].index.tolist()
        
        #Randomly pick a day to start 
        self._start_index = random.choice(i[4:-math.ceil(self._episode_length/81)])
        #self._start_index = i[3]
        
        observation = self._get_observation(index=self._start_index, 
                                            history_length=self._history_length,
                                            position=0)
        
        self._action = self._actions['hold']
        self._position = self._positions['flat']
        self._working_order = None
        self._order_price = 0
        self._target_price = 0
        self._stop_price = 0
        
        self._total_reward = 0
        self._total_pnl = 0
        self._winning_trades = 0.0
        self._total_trades = 0.0
        self._win_rate = 0.0
        
        self._delayed_reward = 0
        self._delayed_reward_index = 0
        
        self._done = False
        self._first_render = True
        
        return observation
    
    
    def _get_observation(self, index, history_length, position=0):
        x_end = index + 1
        x_beg = x_end - history_length + 1
        
        df = self._five_min_data.loc[x_beg:x_end].copy()
        
        '''
        df['open'] = df['open'] / df['close'].iloc[-1]
        df['high'] = df['high'] / df['close'].iloc[-1]
        df['low'] = df['low'] / df['close'].iloc[-1]
        df['close'] = df['close'] / df['close'].iloc[-1]
        df['ema'] = df['ema'] / df['close'].iloc[-1]
        '''
        
        df = df.loc[:,['open','high','low','close','ema','sin_time','cos_time']]
        df.loc[:,'position'] = position
        
        return df.as_matrix()
        
        
    
    def step(self, action):
        """Take an action (buy/sell/hold) and computes the immediate reward.

        Args:
            action (numpy.array): Action to be taken, one-hot encoded.

        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """
        self._action = action
        pnl = 0
        reward = 0
        info = {}
        
        i = self._start_index + self._iteration
        
        
        if self._position == self._positions['flat']:
            #reward -= 1 ##Slight negative reward to force agent to trade
            if self._action == self._actions['buy']:
                reward -= self._order_penalty
                buy_r = self._five_min_data.loc[i, 'buy_r']
                buy_b = self._five_min_data.loc[i, 'buy_b']
                #Check to see if there is a reward and if it comes on this bar or in the future
                if buy_r != 0:
                    if buy_b == 0:
                        pnl += buy_r - 2 * self._commission
                        self._total_trades += 1
                    else:
                        self._position = self._positions['long']
                        self._delayed_reward = buy_r
                        self._delayed_reward_index = i + buy_b
                        pnl -= self._commission
            elif self._action == self._actions['sell']:
                reward -= self._order_penalty
                sell_r = self._five_min_data.loc[i, 'sell_r']
                sell_b = self._five_min_data.loc[i, 'sell_b']
                #Check to see if there is a reward and if it comes on this bar or in the future
                if sell_r != 0:
                    if sell_b == 0:
                        pnl += sell_r - 2 * self._commission
                        self._total_trades += 1
                    else:
                        self._position = self._positions['short']
                        self._delayed_reward = sell_r
                        self._delayed_reward_index = i + sell_b
                        pnl -= self._commission
        
        else: 
            reward -= self._time_penalty
            if i == self._delayed_reward_index:
                self._position = self._positions['flat']
                pnl += self._delayed_reward - self._commission
                self._total_trades += 1
                
            if not self._action == self._actions['hold']:
                #reward -= 500
                pass
        
        #Calculate win rate -- hope this is right
        if pnl > 0:
            self._winning_trades += 1
        if self._total_trades > 0:
            self._win_rate = (self._winning_trades / self._total_trades) * 100
       
        self._iteration += 1
          
        reward += pnl
        self._total_reward += reward
        self._total_pnl += pnl
        
        # End of episode logic
        if self._iteration >= self._episode_length:
            self._done = True
        elif self._total_pnl < -500:
            self._done = True
        
        
        observation = self._get_observation(index=self._start_index + self._iteration, 
                                            history_length=self._history_length,
                                            position=self._position)
        self._observation = observation
        
        return observation, reward, self._done, info
        
    
    def render(self):
        """Matlplotlib rendering of each step.
        """
        if self._first_render:
            self._f, self._ax = plt.subplots(figsize=(16,8))
            self._first_render = False
        
        #Format xaxis
        def format_hour(x, pos=None):
            thisind = np.clip(int(x + 0.5), self._start_index, self._start_index + len(self._five_min_data.index))
            return self._five_min_data['date'][thisind].strftime('%b %-d %I:%M')
        self._ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_hour))
        
        curr_index = self._start_index + self._iteration
        curr_bar = self._five_min_data.loc[curr_index]
        prev_bar = self._five_min_data.loc[curr_index-1]
        curr_bar_ohlc = curr_bar[['open','high','low','close']]
        candle_data = [(curr_index,) + tuple(curr_bar_ohlc.values)]
        candlestick_ohlc(self._ax, candle_data, width=.5, colorup='g', colordown='r', alpha=1)
        
        # Adjust axes
        i = curr_index - self._start_index
        i_min = max(0,i-100) + self._start_index
        i_max = i + self._start_index + 1 if i < 100 else i_min + 101
        self._ax.set_xlim(i_min - 0.5, i_max + 0.5)
        y_max = self._five_min_data.loc[i_min:i_max]['high'].max()
        y_min = self._five_min_data.loc[i_min:i_max]['low'].min()
        self._ax.set_ylim(y_min - 1, y_max + 1)
        
        # Plot vertical lines indicating new trading day
        ts = curr_bar['date']
        if (ts.hour == 9) and (ts.minute == 35):
            self._ax.axvline(curr_index - 0.5, color='black', lw=0.5)
        
        # Plot ema
        self._ax.plot(self._five_min_data.loc[i_min:i_max].index.tolist(), 
                self._five_min_data.loc[i_min:i_max]['ema'].tolist(), 
                color='blue', lw=0.5) 
        
        # Plot action
        if self._action == self._actions['buy']:
            #self._ax.scatter(curr_index + 1, curr_bar['low'], 
                             #color='lawngreen', marker='^', zorder=100)
            self._ax.plot([curr_index - 1, curr_index], [prev_bar['low'], prev_bar['low']],
                         color='black', ls="-", zorder=100)
        elif self._action == self._actions['sell']:
            self._ax.plot([curr_index - 1, curr_index], [prev_bar['high'], prev_bar['high']],
                         color='black', ls="-", zorder=100)
            
        # Plot stats
        plt.suptitle('Episode Length: ' + "%.0f" % self._iteration + ' ~ ' +
                     'Total Reward: ' + "%.2f" % self._total_reward + ' ~ ' +
                     'Total PnL: ' + "%.2f" % self._total_pnl + ' ~ ' +
                     'Total Trades: ' + "%.0f" % self._total_trades + ' ~ ' +
                     'Win Rate: ' + "%.2f" % self._win_rate + ' ~ ' +
                     'Position: ' + "%.0f" % self._position)
        
        plt.pause(.2)      


class ESTradingEnv(Env):
    
    _actions = {
        'hold': 0,
        'buy': 1,
        'sell': 2
    }

    _positions = {
        'flat': 0,
        'long': 1,
        'short': -1
    }
    
    def __init__(self, history_length=64, episode_length=20*81, commission=2,
                order_penalty=0, time_penalty=0):
        
        tick_data = pd.read_feather('../data/processed/ES_tick.feather')
        tick_data = tick_data[tick_data['date'] > '2017-07-29']
        #Create Index from date column
        tick_data.index = tick_data['date']
        tick_data.drop(labels=['date'],axis=1,inplace=True)
        
        """Initialisation function"""
        self._tick_data = tick_data
        self._five_min_data = self._make_5min_bars(tick_data)
        self._history_length = history_length
        self._episode_length = episode_length
        
        self._commission = commission
        self._order_penalty = order_penalty
        self._time_penalty = time_penalty
        
        # We can take 3 actions 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        # Observation consists of history_length bars and 8 features
        # First of the 8 features is the position: 0=flat, 1=long, -1=short
        # Next 5 features are ohlc and the value of the 20 day ema
        # Last 2 features are sin_time and cos_time
        self.observation_space = spaces.Box(low=-9999, high=9999, 
                                            shape=(history_length,8), dtype=np.float32)
        
        self._first_render = True
        self._observation = self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        """Reset the trading environment. Reset rewards, data generator...

        Returns:
            observation (numpy.array): observation of the state
        """
        
        self._iteration = 0
        
        #Find indices of the first bars of each day
        i = self._five_min_data[(self._five_min_data['date'].dt.hour == 9) & \
                 (self._five_min_data['date'].dt.minute == 35)].index.tolist()
        
        #Randomly pick a day to start 
        self._start_index = random.choice(i[4:-math.ceil(self._episode_length/81)])
        
        observation = self._get_observation(index=self._start_index, 
                                            history_length=self._history_length,
                                            position=0)
        
        self._action = self._actions['hold']
        self._position = self._positions['flat']
        self._working_order = None
        self._order_price = 0
        self._target_price = 0
        self._stop_price = 0
        
        self._total_reward = 0
        self._total_pnl = 0
        self._winning_trades = 0.0
        self._total_trades = 0.0
        self._win_rate = 0.0
        
        self._done = False
        self._first_render = True
        
        
        return observation
    
    def _get_observation(self, index, history_length, position=0):
        x_end = index + 1
        x_beg = x_end - history_length
        
        df = self._five_min_data.iloc[x_beg:x_end]
        df = df.loc[:,['open','high','low','close','ema','sin_time','cos_time']]
        df.loc[:,'position'] = position
        
        return df.as_matrix()
        
        
    
    
    def step(self, action):
        """Take an action (buy/sell/hold) and computes the immediate reward.

        Args:
            action (numpy.array): Action to be taken, one-hot encoded.

        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """
        self._action = action
        self._iteration += 1
        pnl = 0
        reward = 0
        info = {}
        
        # Let's move forward one step in time
        x_end = self._start_index + 1 + self._iteration
        x_beg = x_end - self._history_length
        price_series = self._five_min_data.iloc[x_beg:x_end]
        
        # Get tick data for the last bar in the price series 
        ts_end = price_series.iloc[-1]['date']
        ts_start = ts_end - pd.Timedelta(minutes=5)
        ticks = self._tick_data[(self._tick_data.index > ts_start) & \
                                (self._tick_data.index <= ts_end)]
        
        
        # If we don't have a position and there is a buy or sell action, we create the order
        # If we have a position, we apply the time penalty and another large penalty if the
        # system has issued another buy or sell action
        if self._position == self._positions['flat']:
            # Create order -- buy at low or sell at high of previous bar
            if self._action == self._actions['buy']:
                self._working_order = self._actions['buy']
                self._order_price = price_series.iloc[-2]['low']
                reward -= self._order_penalty
            elif self._action == self._actions['sell']:
                self._working_order = self._actions['sell']
                self._order_price = price_series.iloc[-2]['high']
                reward -= self._order_penalty
        else:
            reward -= self._time_penalty
            if not self._action == self._actions['hold']:
                reward -= 500
                #print("Position not flat -- action was buy or sell")
            
        # Simulate order execution by processing each tick in the last bar
        for index, row in ticks.iterrows():
            price = row['last']
            if self._position == self._positions['flat']:
                if self._working_order == self._actions['buy']:
                    if price < self._order_price:
                        self._position = self._positions['long']
                        self._target_price = self._order_price + 1
                        self._stop_price = self._order_price - 1
                        self._working_order = None
                        pnl -= self._commission
                        #print("Buy Order Filled: %s" % self._order_price)
                elif self._working_order == self._actions['sell']:
                    if price > self._order_price:
                        self._position = self._positions['short']
                        self._target_price = self._order_price - 1
                        self._stop_price = self._order_price + 1
                        self._working_order = None
                        pnl -= self._commission
                        #print("Sell Order Filled: %s" % self._order_price)
            elif self._position == self._positions['long']:
                if price > self._target_price:
                    #print("Target Hit: %s" % self._target_price)
                    self._position = self._positions['flat']
                    self._target_price, self._stop_price, self._order_price = (0, ) * 3
                    pnl += 50 - self._commission
                    self._winning_trades += 1
                    self._total_trades += 1
                    self._win_rate = (self._winning_trades / self._total_trades) * 100
                elif price <= self._stop_price:
                    #print("Stop Hit: %s" % self._stop_price)
                    self._position = self._positions['flat']
                    self._target_price, self._stop_price, self._order_price = (0, ) * 3
                    pnl += -50 - self._commission
                    self._total_trades += 1
                    self._win_rate = (self._winning_trades / self._total_trades) * 100
            elif self._position == self._positions['short']:
                if price < self._target_price:
                    #print("Target Hit: %s" % self._target_price)
                    self._position = self._positions['flat']
                    self._target_price, self._stop_price, self._order_price = (0, ) * 3
                    pnl += 50 - self._commission
                    self._winning_trades += 1
                    self._total_trades += 1
                    self._win_rate = (self._winning_trades / self._total_trades) * 100
                elif price >= self._stop_price:
                    #print("Stop Hit: %s" % self._stop_price)
                    self._position = self._positions['flat']
                    self._target_price, self._stop_price, self._order_price = (0, ) * 3
                    pnl += -50 - self._commission
                    self._total_trades += 1
                    self._win_rate = (self._winning_trades / self._total_trades) * 100
        
        reward += pnl
        self._total_reward += reward
        self._total_pnl += pnl
        
        # End of episode logic
        if self._iteration >= self._episode_length:
            self._done = True
        elif self._total_pnl < -500:
            self._done = True
        
        
        observation = self._get_observation(index=self._start_index + self._iteration, 
                                            history_length=self._history_length,
                                            position=self._position)
        self._observation = observation
        
        return observation, reward, self._done, info
        
    
    def render(self):
        """Matlplotlib rendering of each step.
        """
        if self._first_render:
            self._f, self._ax = plt.subplots(figsize=(16,8))
            self._first_render = False
        
        #Format xaxis
        def format_hour(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, len(self._five_min_data.index) - 1)
            return self._five_min_data['date'][thisind].strftime('%b %-d %I:%M')
        self._ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_hour))
        
        curr_index = self._start_index + self._iteration
        curr_bar = self._five_min_data.iloc[curr_index]
        prev_bar = self._five_min_data.iloc[curr_index-1]
        curr_bar_ohlc = curr_bar[['open','high','low','close']]
        candle_data = [(curr_index,) + tuple(curr_bar_ohlc.values)]
        candlestick_ohlc(self._ax, candle_data, width=.5, colorup='g', colordown='r', alpha=1)
        
        # Adjust axes
        i = curr_index - self._start_index
        i_min = max(0,i-100) + self._start_index
        i_max = i + self._start_index + 1 if i < 100 else i_min + 101
        self._ax.set_xlim(i_min - 0.5, i_max + 0.5)
        y_max = self._five_min_data.loc[i_min:i_max]['high'].max()
        y_min = self._five_min_data.loc[i_min:i_max]['low'].min()
        self._ax.set_ylim(y_min - 1, y_max + 1)
        
        # Plot vertical lines indicating new trading day
        ts = curr_bar['date']
        if (ts.hour == 9) and (ts.minute == 35):
            self._ax.axvline(curr_index - 0.5, color='black', lw=0.5)
        
        # Plot ema
        self._ax.plot(self._five_min_data.loc[i_min:i_max].index.tolist(), 
                self._five_min_data.loc[i_min:i_max]['ema'].tolist(), 
                color='blue', lw=0.5) 
        
        # Plot action
        if self._action == self._actions['buy']:
            #self._ax.scatter(curr_index + 1, curr_bar['low'], 
                             #color='lawngreen', marker='^', zorder=100)
            self._ax.plot([curr_index - 1, curr_index], [prev_bar['low'], prev_bar['low']],
                         color='black', ls="-", zorder=100)
        elif self._action == self._actions['sell']:
            self._ax.plot([curr_index - 1, curr_index], [prev_bar['high'], prev_bar['high']],
                         color='black', ls="-", zorder=100)
            
        # Plot stats
        plt.suptitle('Episode Length: ' + "%.0f" % self._iteration + ' ~ ' +
                     'Total Reward: ' + "%.2f" % self._total_reward + ' ~ ' +
                     'Total PnL: ' + "%.2f" % self._total_pnl + ' ~ ' +
                     'Total Trades: ' + "%.0f" % self._total_trades + ' ~ ' +
                     'Win Rate: ' + "%.2f" % self._win_rate + ' ~ ' +
                     'Position: ' + "%.0f" % self._position)
        
        plt.pause(.01)
        
    
    def _make_5min_bars(self, tick_data):
        #Resample to get 5min bars
        five_min_data = pd.DataFrame(
            tick_data['last'].resample('5Min', loffset=datetime.timedelta(minutes=5)).ohlc())
        
        #Create RTH Calendar
        
        #We hack the NYSE Calendar extending the close until 4:15
        class CMERTHCalendar(mcal.exchange_calendar_nyse.NYSEExchangeCalendar):
            @property
            def close_time(self):
                return datetime.time(16, 15)
        
        nyse = CMERTHCalendar()
        schedule = nyse.schedule(start_date=five_min_data.index.min(), 
                                 end_date=five_min_data.index.max())
        
        #Filter out those bars that occur during RTH
        five_min_data['dates'] = pd.to_datetime(five_min_data.index.to_datetime().date)
        five_min_data['valid_date'] = five_min_data['dates'].isin(schedule.index)
        five_min_data['valid_time'] = False
        during_rth = five_min_data['valid_date'] & \
                (five_min_data.index > schedule.loc[five_min_data['dates'],'market_open']) & \
                (five_min_data.index <= schedule.loc[five_min_data['dates'],'market_close'])
        five_min_data.loc[during_rth, 'valid_time'] = True
        five_min_data = five_min_data[five_min_data['valid_time'] == True]
        five_min_data.drop(['dates','valid_date','valid_time'], axis=1, inplace=True)
        
        #Add ema
        five_min_data['ema'] = five_min_data['close'].ewm(span=20, min_periods=20).mean()

        #Reset index
        five_min_data.reset_index(inplace=True)
        
        #Add column for number of seconds elapsed in trading day
        five_min_data['sec'] = (five_min_data['date'].values 
                                - five_min_data['date'].values.astype('datetime64[D]')) / np.timedelta64(1,'s')

        #Calculate sin & cos time
        #24hr time is a cyclical continuous feature
        seconds_in_day = 24*60*60
        five_min_data['sin_time'] = np.sin(2*np.pi*five_min_data['sec']/seconds_in_day)
        five_min_data['cos_time'] = np.cos(2*np.pi*five_min_data['sec']/seconds_in_day)

        five_min_data.drop('sec', axis=1, inplace=True)
        
        return five_min_data