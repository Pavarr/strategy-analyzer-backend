from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import math
from collections import defaultdict
import numpy as np
from scipy import stats
import statsmodels.api as sm

class StrategyParser:
    def __init__(self, html_path):
        self.html_path = html_path
        self.trades = []
        self.metrics = {}
        self.original_balance = None
        self.original_lot_size = None
        self.strategy_name = None  # Sarà impostato dal backend
        
    def parse_html(self):
        """Estrae la tabella Affari dall'HTML"""
        with open(self.html_path, 'r', encoding='utf-16') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Trova tutte le righe della tabella
        trades_data = []
        
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            
            # Controlla se è una riga di trade (13 celle e ha bgcolor)
            if len(cells) == 13 and row.get('bgcolor'):
                first_cell = cells[0].get_text().strip()
                
                # Verifica che sia un timestamp valido
                if '.' in first_cell and ' ' in first_cell and ':' in first_cell:
                    try:
                        trade = {
                            'datetime': cells[0].get_text().strip(),
                            'order': cells[1].get_text().strip(),
                            'symbol': cells[2].get_text().strip(),
                            'type': cells[3].get_text().strip(),
                            'direction': cells[4].get_text().strip(),
                            'volume': float(cells[5].get_text().strip()),
                            'price': float(cells[6].get_text().strip()),
                            'order_id': cells[7].get_text().strip(),
                            'commission': float(cells[8].get_text().strip()),
                            'swap': float(cells[9].get_text().strip()),
                            'profit': float(cells[10].get_text().strip().replace(' ', '')),
                            'balance': float(cells[11].get_text().strip().replace(' ', '')),
                            'comment': cells[12].get_text().strip()
                        }
                        trades_data.append(trade)
                    except (ValueError, IndexError):
                        continue
        
        if not trades_data:
            raise ValueError("Nessun trade trovato nell'HTML")
        
        self.trades = trades_data
        
        # Salva balance originale (prima del primo trade)
        closes = [t for t in self.trades if t['direction'] == 'out']
        if closes:
            self.original_balance = closes[0]['balance'] - closes[0]['profit']
        else:
            # Fallback: usa primo trade o default
            if self.trades:
                self.original_balance = self.trades[0].get('balance', 10000)
            else:
                self.original_balance = 10000
        
        return self.trades
    
    def check_uniform_lot_size(self):
        """Verifica se tutti i trade hanno lo stesso lottaggio"""
        lot_sizes = set([t['volume'] for t in self.trades])
        is_uniform = len(lot_sizes) == 1
        lot_size = list(lot_sizes)[0] if is_uniform else None
        
        if is_uniform:
            self.original_lot_size = lot_size
        
        return {
            'is_uniform': is_uniform,
            'lot_size': lot_size,
            'unique_lots': list(lot_sizes)
        }
    
    def calculate_commission_swap_totals(self):
        """Calcola totali commission e swap"""
        closes = [t for t in self.trades if t['direction'] == 'out']
        
        total_commission = sum([t['commission'] for t in closes])
        total_swap = sum([t['swap'] for t in closes])
        total_profit = sum([t['profit'] for t in closes])
        
        # Percentuale sul profitto (se profitto > 0)
        commission_pct = (abs(total_commission) / total_profit * 100) if total_profit > 0 else 0
        swap_pct = (abs(total_swap) / total_profit * 100) if total_profit > 0 else 0
        
        return {
            'total_commission': round(total_commission, 2),
            'total_swap': round(total_swap, 2),
            'commission_pct_on_profit': round(commission_pct, 2),
            'swap_pct_on_profit': round(swap_pct, 2)
        }
    
    def recalculate_with_balance(self, new_balance):
        """Ricalcola metriche con nuovo balance iniziale (profitti assoluti invariati)"""
        # Genera output con balance modificato
        return self.generate_output(risk_free_rate=0, custom_balance=new_balance)
    
    def recalculate_with_lot_multiplier(self, lot_multiplier):
        """Ricalcola metriche con nuovo moltiplicatore lottaggio"""
        # Salva trades originali
        original_trades = self.trades.copy()
        
        # Moltiplica profit, commission, swap
        for trade in self.trades:
            trade['profit'] = trade['profit'] * lot_multiplier
            trade['commission'] = trade['commission'] * lot_multiplier
            trade['swap'] = trade['swap'] * lot_multiplier
            trade['volume'] = trade['volume'] * lot_multiplier
        
        # Ricalcola balance
        self._recalculate_balance()
        
        # Genera output
        result = self.generate_output(risk_free_rate=0)
        
        # Ripristina trades originali
        self.trades = original_trades
        
        return result
    
    def _recalculate_balance(self):
        """Ricalcola balance dopo modifica profit/commission/swap"""
        closes = [t for t in self.trades if t['direction'] == 'out']
        starting_balance = self.original_balance
        
        running_balance = starting_balance
        for i, trade in enumerate(self.trades):
            if trade['direction'] == 'out':
                running_balance += trade['profit']
                trade['balance'] = running_balance
    
    def _calculate_duration_no_weekends(self, dt_start, dt_end):
        """Calcola durata in secondi tra due datetime ESCLUDENDO weekend"""
        from datetime import timedelta
        
        total_seconds = 0
        current = dt_start
        
        while current < dt_end:
            next_day_start = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            segment_end = min(next_day_start, dt_end)
            
            if current.weekday() < 5:  # Lunedì-Venerdì
                segment_duration = (segment_end - current).total_seconds()
                total_seconds += segment_duration
            
            current = next_day_start
        
        return total_seconds
    
    def calculate_trade_duration(self):
        """Calcola durata media dei trade escludendo weekend"""
        durations = []
        
        for i in range(len(self.trades) - 1):
            current = self.trades[i]
            next_trade = self.trades[i + 1]
            
            if current['direction'] == 'in' and next_trade['direction'] == 'out':
                dt1 = datetime.strptime(current['datetime'], '%Y.%m.%d %H:%M:%S')
                dt2 = datetime.strptime(next_trade['datetime'], '%Y.%m.%d %H:%M:%S')
                
                duration_seconds = self._calculate_duration_no_weekends(dt1, dt2)
                durations.append(duration_seconds)
        
        if not durations:
            return {
                'avg_duration_hours': 0,
                'avg_duration_minutes': 0,
                'avg_duration_formatted': '0h 0m',
                'min_duration_hours': 0,
                'min_duration_minutes': 0,
                'min_duration_formatted': '0h 0m',
                'max_duration_hours': 0,
                'max_duration_minutes': 0,
                'max_duration_formatted': '0h 0m'
            }
        
        avg_duration_seconds = sum(durations) / len(durations)
        avg_hours = int(avg_duration_seconds // 3600)
        avg_minutes = int((avg_duration_seconds % 3600) // 60)
        
        min_duration_seconds = min(durations)
        min_hours = int(min_duration_seconds // 3600)
        min_minutes = int((min_duration_seconds % 3600) // 60)
        
        max_duration_seconds = max(durations)
        max_hours = int(max_duration_seconds // 3600)
        max_minutes = int((max_duration_seconds % 3600) // 60)
        
        return {
            'avg_duration_hours': avg_hours,
            'avg_duration_minutes': avg_minutes,
            'avg_duration_formatted': f'{avg_hours}h {avg_minutes}m',
            'min_duration_hours': min_hours,
            'min_duration_minutes': min_minutes,
            'min_duration_formatted': f'{min_hours}h {min_minutes}m',
            'max_duration_hours': max_hours,
            'max_duration_minutes': max_minutes,
            'max_duration_formatted': f'{max_hours}h {max_minutes}m'
        }
    
    def create_equity_line(self, custom_balance=None, filtered_trades=None):
        """Crea equity line usando solo le chiusure
        
        Args:
            custom_balance: Balance iniziale custom
            filtered_trades: Lista trade filtrati (per long/short only)
        """
        equity_line = {
            'dates': [],
            'values': []
        }
        
        # Usa trade filtrati se forniti, altrimenti tutti i trade
        trades_to_use = filtered_trades if filtered_trades is not None else self.trades
        closes = [t for t in trades_to_use if t['direction'] == 'out']
        
        if not closes:
            return equity_line
        
        # Determina balance iniziale
        if custom_balance:
            starting_balance = custom_balance
        else:
            # Usa balance originale salvato
            starting_balance = self.original_balance if self.original_balance else (closes[0]['balance'] - closes[0]['profit'])
        
        # Costruisci equity line partendo dal balance iniziale
        running_balance = starting_balance
        for trade in closes:
            running_balance += trade['profit']
            equity_line['dates'].append(trade['datetime'])
            equity_line['values'].append(running_balance)
        
        return equity_line
    
    def calculate_drawdowns(self, equity_line=None):
        """Calcola tabella drawdown"""
        if equity_line is None:
            equity_line = self.create_equity_line()
        
        equity = equity_line['values']
        dates = equity_line['dates']
        
        if not equity:
            return []
        
        drawdowns = []
        current_peak = equity[0]
        current_peak_idx = 0
        
        for i in range(len(equity)):
            if equity[i] > current_peak:
                if current_peak_idx < i - 1:
                    trough_idx = current_peak_idx
                    trough = equity[current_peak_idx]
                    
                    for j in range(current_peak_idx, i):
                        if equity[j] < trough:
                            trough = equity[j]
                            trough_idx = j
                    
                    dd_pct = ((trough - current_peak) / current_peak) * 100
                    
                    recovery_idx = None
                    for j in range(trough_idx + 1, len(equity)):
                        if equity[j] >= current_peak:
                            recovery_idx = j
                            break
                    
                    if recovery_idx is not None:
                        dt1 = datetime.strptime(dates[trough_idx].split()[0], '%Y.%m.%d')
                        dt2 = datetime.strptime(dates[recovery_idx].split()[0], '%Y.%m.%d')
                        recovery_days = (dt2 - dt1).days
                    else:
                        recovery_days = 'In corso'
                    
                    drawdowns.append({
                        'peak_date': dates[current_peak_idx].split()[0],
                        'trough_date': dates[trough_idx].split()[0],
                        'recovery_date': dates[recovery_idx].split()[0] if recovery_idx else 'In corso',
                        'peak': round(current_peak, 2),
                        'trough': round(trough, 2),
                        'drawdown_pct': round(dd_pct, 2),
                        'recovery_days': recovery_days
                    })
                
                current_peak = equity[i]
                current_peak_idx = i
        
        if current_peak_idx < len(equity) - 1:
            trough_idx = current_peak_idx
            trough = equity[current_peak_idx]
            
            for j in range(current_peak_idx, len(equity)):
                if equity[j] < trough:
                    trough = equity[j]
                    trough_idx = j
            
            dd_pct = ((trough - current_peak) / current_peak) * 100
            
            drawdowns.append({
                'peak_date': dates[current_peak_idx].split()[0],
                'trough_date': dates[trough_idx].split()[0],
                'recovery_date': 'In corso',
                'peak': round(current_peak, 2),
                'trough': round(trough, 2),
                'drawdown_pct': round(dd_pct, 2),
                'recovery_days': 'In corso'
            })
        
        drawdowns.sort(key=lambda x: x['drawdown_pct'])
        
        return drawdowns
    
    def calculate_drawdown_series(self, equity_line=None):
        """Calcola serie temporale del drawdown per grafico"""
        if equity_line is None:
            equity_line = self.create_equity_line()
        
        equity = equity_line['values']
        dates = equity_line['dates']
        
        dd_series = {
            'dates': dates,
            'values': [],
            'values_pct': []
        }
        
        running_max = equity[0]
        
        for e in equity:
            if e > running_max:
                running_max = e
            
            dd_abs = e - running_max
            dd_pct = (dd_abs / running_max) * 100
            
            dd_series['values'].append(round(dd_abs, 2))
            dd_series['values_pct'].append(round(dd_pct, 2))
        
        return dd_series
    
    def calculate_mae_mfe(self):
        """Calcola MAE/MFE proxy usando equity curve"""
        closes = [t for t in self.trades if t['direction'] == 'out']
        
        if len(closes) < 2:
            return {
                'avg_mae': 0,
                'avg_mae_pct': 0,
                'avg_mfe': 0,
                'avg_mfe_pct': 0,
                'mae_mfe_ratio': 0
            }
        
        mae_list = []
        mfe_list = []
        
        for i in range(len(closes) - 1):
            entry_equity = closes[i]['balance']
            exit_equity = closes[i + 1]['balance']
            
            entry_idx = self.trades.index(next(t for t in self.trades if t == closes[i]))
            exit_idx = self.trades.index(next(t for t in self.trades if t == closes[i + 1]))
            
            equity_slice = [self.trades[j]['balance'] for j in range(entry_idx, exit_idx + 1)]
            
            if equity_slice:
                mae = min(equity_slice) - entry_equity
                mae_pct = (mae / entry_equity * 100) if entry_equity != 0 else 0
                
                mfe = max(equity_slice) - entry_equity
                mfe_pct = (mfe / entry_equity * 100) if entry_equity != 0 else 0
                
                mae_list.append(mae)
                mfe_list.append(mfe)
        
        avg_mae = sum(mae_list) / len(mae_list) if mae_list else 0
        avg_mfe = sum(mfe_list) / len(mfe_list) if mfe_list else 0
        
        entry_equities = [closes[i]['balance'] for i in range(len(closes) - 1)]
        avg_entry = sum(entry_equities) / len(entry_equities) if entry_equities else 1
        
        avg_mae_pct = (avg_mae / avg_entry * 100) if avg_entry != 0 else 0
        avg_mfe_pct = (avg_mfe / avg_entry * 100) if avg_entry != 0 else 0
        
        mae_mfe_ratio = abs(avg_mfe / avg_mae) if avg_mae != 0 else 0
        
        return {
            'avg_mae': round(avg_mae, 2),
            'avg_mae_pct': round(avg_mae_pct, 2),
            'avg_mfe': round(avg_mfe, 2),
            'avg_mfe_pct': round(avg_mfe_pct, 2),
            'mae_mfe_ratio': round(mae_mfe_ratio, 2)
        }
    
    def calculate_advanced_metrics(self, risk_free_rate=0, custom_balance=None, trade_type='all'):
        """Calcola metriche avanzate con supporto filtro trade type"""
        # Applica filtro trade
        filtered_trades = self.filter_trades_by_type(trade_type)
        closes = [t for t in filtered_trades if t['direction'] == 'out']
        
        if not closes or len(closes) < 2:
            return self._empty_advanced_metrics()
        
        # Profit Factor
        total_wins = sum([t['profit'] for t in closes if t['profit'] > 0])
        total_losses = abs(sum([t['profit'] for t in closes if t['profit'] < 0]))
        profit_factor = (total_wins / total_losses) if total_losses != 0 else 0
        
        # Win/Loss Ratio
        wins = [t['profit'] for t in closes if t['profit'] > 0]
        losses = [t['profit'] for t in closes if t['profit'] < 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Kelly Criterion
        win_rate = len(wins) / len(closes) if closes else 0
        kelly_pct = (win_rate - ((1 - win_rate) / win_loss_ratio)) * 100 if win_loss_ratio != 0 else 0
        
        # Calcola returns per Sharpe/Sortino - usa equity filtrata
        equity_line = self.create_equity_line(custom_balance, filtered_trades)
        equity_values = equity_line['values']
        
        if not equity_values:
            return self._empty_advanced_metrics()
        
        # Starting balance
        starting_balance = custom_balance if custom_balance else (self.original_balance if self.original_balance else 10000)
        
        # Primo return: dal capitale iniziale al primo trade
        first_return = closes[0]['profit'] / starting_balance if starting_balance != 0 else 0
        
        # Altri returns
        returns = [first_return]
        for i in range(1, len(equity_values)):
            ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
            returns.append(ret)
        
        if not returns:
            return self._empty_advanced_metrics()
        
        # Statistiche returns
        avg_return = sum(returns) / len(returns)
        
        variance = sum([(r - avg_return) ** 2 for r in returns]) / len(returns)
        volatility = math.sqrt(variance)
        
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_variance = sum([r ** 2 for r in negative_returns]) / len(negative_returns)
            downside_deviation = math.sqrt(downside_variance)
        else:
            downside_deviation = 0.0001
        
        # Sharpe Ratio
        rf_per_trade = risk_free_rate / 100 / len(returns)
        sharpe_ratio = ((avg_return - rf_per_trade) / volatility) if volatility != 0 else 0
        
        # Annualizza
        first_date = datetime.strptime(equity_line['dates'][0].split()[0], '%Y.%m.%d')
        last_date = datetime.strptime(equity_line['dates'][-1].split()[0], '%Y.%m.%d')
        days_total = (last_date - first_date).days or 1
        
        trades_per_year = (len(closes) / days_total) * 365
        sharpe_annualized = sharpe_ratio * math.sqrt(trades_per_year)
        
        # Sortino Ratio
        sortino_ratio = ((avg_return - rf_per_trade) / downside_deviation) if downside_deviation != 0 else 0
        sortino_annualized = sortino_ratio * math.sqrt(trades_per_year)
        
        # Calmar Ratio
        ending_balance = equity_values[-1]
        total_return = (ending_balance - starting_balance) / starting_balance
        
        years = days_total / 365
        annualized_return = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Max Drawdown % - CORRETTO per Recovery Factor
        max_dd = 0
        running_max = equity_values[0]
        equity_at_max_dd_peak = equity_values[0]
        
        for e in equity_values:
            if e > running_max:
                running_max = e
            drawdown = (e - running_max) / running_max * 100
            if drawdown < max_dd:
                max_dd = drawdown
                equity_at_max_dd_peak = running_max
        
        calmar_ratio = (annualized_return / abs(max_dd)) if max_dd != 0 else 0
        
        # Recovery Factor - CORRETTO
        net_profit = ending_balance - starting_balance
        max_dd_abs = equity_at_max_dd_peak * (abs(max_dd) / 100)
        recovery_factor = (net_profit / max_dd_abs) if max_dd_abs != 0 else 0
        
        # === NUOVE STATISTICHE ===
        
        # SQN (System Quality Number)
        profits = [t['profit'] for t in closes]
        sqn = self.calculate_sqn(profits)
        
        # P-value IID (T-test)
        p_value_iid = self.calculate_p_value_ttest(returns)
        
        # P-value HAC (Newey-West)
        p_value_hac = self.calculate_p_value_newey_west(returns)
        
        return {
            'profit_factor': round(profit_factor, 2),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'kelly_pct': round(kelly_pct, 2),
            'sharpe_ratio': round(sharpe_annualized, 2),
            'sortino_ratio': round(sortino_annualized, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'recovery_factor': round(recovery_factor, 3),
            'annualized_return': round(annualized_return, 2),
            'volatility_annual': round(volatility * math.sqrt(trades_per_year) * 100, 2),
            'sqn': round(sqn, 2),
            'p_value_iid': round(p_value_iid, 4),
            'p_value_hac': round(p_value_hac, 4)
        }
    
    def _empty_advanced_metrics(self):
        """Ritorna metriche vuote"""
        return {
            'profit_factor': 0,
            'win_loss_ratio': 0,
            'kelly_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'recovery_factor': 0,
            'annualized_return': 0,
            'volatility_annual': 0,
            'sqn': 0,
            'p_value_iid': 1.0,
            'p_value_hac': 1.0
        }
    
    def filter_trades_by_type(self, trade_type='all'):
        """
        Filtra i trade per tipo
        
        Args:
            trade_type: 'all', 'long' (buy), o 'short' (sell)
            
        Returns:
            Lista di trade filtrati
        """
        if trade_type == 'all':
            return self.trades
        elif trade_type == 'long':
            # Filtra solo trade buy
            filtered = []
            for i in range(len(self.trades)):
                if self.trades[i]['direction'] == 'in' and self.trades[i]['type'] == 'buy':
                    filtered.append(self.trades[i])
                    # Aggiungi anche l'uscita corrispondente
                    if i + 1 < len(self.trades) and self.trades[i + 1]['direction'] == 'out':
                        filtered.append(self.trades[i + 1])
            return filtered
        elif trade_type == 'short':
            # Filtra solo trade sell
            filtered = []
            for i in range(len(self.trades)):
                if self.trades[i]['direction'] == 'in' and self.trades[i]['type'] == 'sell':
                    filtered.append(self.trades[i])
                    # Aggiungi anche l'uscita corrispondente
                    if i + 1 < len(self.trades) and self.trades[i + 1]['direction'] == 'out':
                        filtered.append(self.trades[i + 1])
            return filtered
        else:
            return self.trades
    
    def calculate_metrics(self, custom_balance=None, trade_type='all'):
        """Calcola metriche principali con supporto filtro trade type"""
        # Applica filtro trade
        filtered_trades = self.filter_trades_by_type(trade_type)
        closes = [t for t in filtered_trades if t['direction'] == 'out']
        
        if not closes:
            return {}
        
        # Win rate
        wins = len([t for t in closes if t['profit'] > 0])
        losses = len([t for t in closes if t['profit'] < 0])
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Profit/Loss statistics
        profits = [t['profit'] for t in closes]
        avg_profit = sum(profits) / len(profits) if profits else 0
        total_profit = sum(profits)
        
        win_profits = [t['profit'] for t in closes if t['profit'] > 0]
        avg_win = sum(win_profits) / len(win_profits) if win_profits else 0
        
        loss_profits = [t for t in closes if t['profit'] < 0]
        avg_loss = sum(loss_profits) / len(loss_profits) if loss_profits else 0
        
        # Max DD - usa equity line filtrata
        equity_line = self.create_equity_line(custom_balance, filtered_trades)
        equity = equity_line['values']
        
        if not equity:
            max_dd = 0
            starting_balance = custom_balance if custom_balance else (self.original_balance if self.original_balance else 10000)
            ending_balance = starting_balance
        else:
            max_dd = 0
            running_max = equity[0]
            
            for e in equity:
                if e > running_max:
                    running_max = e
                drawdown = (e - running_max) / running_max * 100
                if drawdown < max_dd:
                    max_dd = drawdown
            
            # Starting and ending balance
            starting_balance = custom_balance if custom_balance else (self.original_balance if self.original_balance else 10000)
            ending_balance = equity[-1]
        
        # Total return
        total_return_pct = ((ending_balance - starting_balance) / starting_balance) * 100 if starting_balance != 0 else 0
        
        self.metrics = {
            'total_trades': int(total_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate_pct': round(win_rate, 2),
            'avg_profit': round(avg_profit, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'total_profit': round(total_profit, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'starting_balance': round(starting_balance, 2),
            'ending_balance': round(ending_balance, 2),
            'total_return_pct': round(total_return_pct, 2)
        }
        
        return self.metrics
    
    def calculate_sqn(self, profits):
        """
        Calcola System Quality Number (Van Tharp)
        SQN = sqrt(N) * (mean_profit / std_profit)
        
        Args:
            profits: Lista profitti trade
            
        Returns:
            float: SQN value
        """
        if len(profits) < 2:
            return 0
        
        n = len(profits)
        mean_profit = np.mean(profits)
        std_profit = np.std(profits, ddof=1)
        
        if std_profit == 0:
            return 0
        
        sqn = np.sqrt(n) * (mean_profit / std_profit)
        return sqn
    
    def calculate_p_value_ttest(self, returns):
        """
        1-sample T-test per verificare se mean(returns) != 0
        
        H0: mean return = 0 (nessun edge)
        H1: mean return != 0 (edge presente)
        
        Args:
            returns: Lista rendimenti
            
        Returns:
            float: p-value
        """
        if len(returns) < 2:
            return 1.0  # Non possiamo rigettare H0
        
        try:
            # T-test contro mean = 0
            t_statistic, p_value = stats.ttest_1samp(returns, 0)
            return p_value
        except:
            return 1.0
    
    def calculate_p_value_newey_west(self, returns):
        """
        OLS regression con Newey-West HAC standard errors
        Testa se mean return è significativamente diverso da 0
        considerando autocorrelazione ed eteroschedasticità
        
        Args:
            returns: Lista rendimenti
            
        Returns:
            float: p-value con correzione HAC
        """
        if len(returns) < 10:  # Serve un minimo di dati
            return 1.0
        
        try:
            # Regressione semplice: returns = alpha + epsilon
            y = np.array(returns)
            X = np.ones(len(returns))  # Solo intercetta
            
            # OLS con Newey-West
            model = sm.OLS(y, X)
            
            # Maxlags per Newey-West (regola empirica: n^0.25)
            maxlags = int(len(returns) ** 0.25)
            
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
            
            # P-value per l'intercetta (alpha)
            p_value = results.pvalues[0]
            
            return p_value
            
        except Exception as e:
            # Fallback a T-test se Newey-West fallisce
            return self.calculate_p_value_ttest(returns)
    
    def calculate_weekday_statistics(self, last_6_months_only=False, filtered_trades=None):
        """
        Calcola statistiche per giorno della settimana (basato su data di entrata)
        
        Args:
            last_6_months_only: Se True, considera solo ultimi 6 mesi DEI DATI
            filtered_trades: Lista trade filtrati (per long/short only)
            
        Returns:
            dict con statistiche per ogni giorno (0=Lunedì, 6=Domenica)
        """
        # Usa trade filtrati se forniti, altrimenti tutti i trade
        trades_to_use = filtered_trades if filtered_trades is not None else self.trades
        closes = [t for t in trades_to_use if t['direction'] == 'out']
        
        if not closes:
            return {}
        
        # Trova ultimo trade nei dati
        all_dates = []
        for i in range(len(trades_to_use) - 1):
            if trades_to_use[i]['direction'] == 'in':
                entry_dt = datetime.strptime(trades_to_use[i]['datetime'].split(' ')[0], '%Y.%m.%d')
                all_dates.append(entry_dt)
        
        # Calcola cutoff per ultimi 6 mesi DEI DATI
        six_months_cutoff = None
        if last_6_months_only and all_dates:
            last_trade_date = max(all_dates)
            six_months_cutoff = last_trade_date - timedelta(days=180)
            
        # Raggruppa trade per giorno della settimana (basato su entrata)
        weekday_trades = defaultdict(list)
        
        for i in range(len(trades_to_use) - 1):
            if trades_to_use[i]['direction'] == 'in' and trades_to_use[i+1]['direction'] == 'out':
                entry = trades_to_use[i]
                exit_trade = trades_to_use[i+1]
                
                # Parse data di entrata
                entry_dt = datetime.strptime(entry['datetime'].split(' ')[0], '%Y.%m.%d')
                
                # Filtra per 6 mesi se richiesto
                if last_6_months_only and six_months_cutoff and entry_dt < six_months_cutoff:
                    continue
                
                weekday = entry_dt.weekday()  # 0=Lunedì, 6=Domenica
                weekday_trades[weekday].append(exit_trade['profit'])
        
        # Calcola statistiche per ogni giorno
        weekday_stats = {}
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in range(7):
            if day not in weekday_trades or len(weekday_trades[day]) == 0:
                weekday_stats[weekday_names[day]] = {
                    'num_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_return': 0,
                    'volatility': 0,
                    'win_loss_ratio': 0,
                    'returns': []
                }
                continue
            
            profits = weekday_trades[day]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]
            
            num_trades = len(profits)
            win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
            
            # Avg return e volatilità
            avg_return = sum(profits) / num_trades if num_trades > 0 else 0
            volatility = np.std(profits) if len(profits) > 1 else 0
            
            # Win/Loss ratio
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            weekday_stats[weekday_names[day]] = {
                'num_trades': num_trades,
                'win_rate': round(win_rate, 2),
                'profit_factor': round(profit_factor, 2),
                'avg_return': round(avg_return, 2),
                'volatility': round(volatility, 2),
                'win_loss_ratio': round(win_loss_ratio, 2),
                'returns': [round(p, 2) for p in profits]  # Per box plot
            }
        
        return weekday_stats
    
    def calculate_monthly_statistics(self, filtered_trades=None):
        """
        Calcola statistiche per mese/anno (basato su data di entrata)
        
        Args:
            filtered_trades: Lista trade filtrati (per long/short only)
        
        Returns:
            dict con anni come chiavi e mesi come sotto-chiavi
        """
        # Usa trade filtrati se forniti, altrimenti tutti i trade
        trades_to_use = filtered_trades if filtered_trades is not None else self.trades
        closes = [t for t in trades_to_use if t['direction'] == 'out']
        
        if not closes:
            return {}
        
        # Raggruppa trade per mese/anno
        monthly_trades = defaultdict(lambda: defaultdict(list))
        
        for i in range(len(trades_to_use) - 1):
            if trades_to_use[i]['direction'] == 'in' and trades_to_use[i+1]['direction'] == 'out':
                entry = trades_to_use[i]
                exit_trade = trades_to_use[i+1]
                
                # Parse data di entrata
                entry_dt = datetime.strptime(entry['datetime'].split(' ')[0], '%Y.%m.%d')
                year = entry_dt.year
                month = entry_dt.month
                
                monthly_trades[year][month].append({
                    'profit': exit_trade['profit'],
                    'balance': exit_trade['balance']
                })
        
        # Calcola statistiche per ogni mese
        monthly_stats = {}
        
        for year in sorted(monthly_trades.keys()):
            monthly_stats[year] = {}
            
            # Raccogli tutti i profit dell'anno per calcolare statistiche annuali
            year_profits = []
            year_trades_count = 0
            
            for month in range(1, 13):
                if month not in monthly_trades[year] or len(monthly_trades[year][month]) == 0:
                    monthly_stats[year][month] = {
                        'num_trades': 0,
                        'return_pct': None,
                        'profit_factor': 0,
                        'volatility': 0
                    }
                    continue
                
                trades = monthly_trades[year][month]
                profits = [t['profit'] for t in trades]
                wins = [p for p in profits if p > 0]
                losses = [p for p in profits if p < 0]
                
                num_trades = len(profits)
                year_trades_count += num_trades
                year_profits.extend(profits)
                
                # Rendimento % del mese (calcolato sul balance iniziale del mese)
                first_balance_before = trades[0]['balance'] - trades[0]['profit']
                total_profit_month = sum(profits)
                return_pct = (total_profit_month / first_balance_before * 100) if first_balance_before > 0 else 0
                
                # Profit factor
                gross_profit = sum(wins) if wins else 0
                gross_loss = abs(sum(losses)) if losses else 0
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
                
                # Volatilità
                volatility = np.std(profits) if len(profits) > 1 else 0
                
                monthly_stats[year][month] = {
                    'num_trades': num_trades,
                    'return_pct': round(return_pct, 2),
                    'profit_factor': round(profit_factor, 2),
                    'volatility': round(volatility, 2)
                }
            
            # Calcola statistiche annuali
            if year_profits:
                year_wins = [p for p in year_profits if p > 0]
                year_losses = [p for p in year_profits if p < 0]
                
                year_gross_profit = sum(year_wins) if year_wins else 0
                year_gross_loss = abs(sum(year_losses)) if year_losses else 0
                year_profit_factor = (year_gross_profit / year_gross_loss) if year_gross_loss > 0 else 0
                
                year_total_profit = sum(year_profits)
                year_volatility = np.std(year_profits) if len(year_profits) > 1 else 0
                year_win_rate = (len(year_wins) / len(year_profits) * 100) if year_profits else 0
                
                # Calcola return % annuale (somma di tutti i profit dell'anno)
                # Prendo balance all'inizio dell'anno
                first_year_trade = monthly_trades[year][min(monthly_trades[year].keys())][0]
                year_start_balance = first_year_trade['balance'] - first_year_trade['profit']
                year_return_pct = (year_total_profit / year_start_balance * 100) if year_start_balance > 0 else 0
                
                monthly_stats[year]['yearly_total'] = {
                    'num_trades': year_trades_count,
                    'return_pct': round(year_return_pct, 2),
                    'profit_factor': round(year_profit_factor, 2),
                    'volatility': round(year_volatility, 2),
                    'win_rate': round(year_win_rate, 2)
                }
            else:
                monthly_stats[year]['yearly_total'] = {
                    'num_trades': 0,
                    'return_pct': 0,
                    'profit_factor': 0,
                    'volatility': 0,
                    'win_rate': 0
                }
        
        return monthly_stats
    
    def calculate_returns_distribution(self, filtered_trades=None):
        """
        Calcola distribuzione rendimenti (istogramma) con curtosi e asimmetria
        
        Args:
            filtered_trades: Lista trade filtrati (per long/short only)
        
        Returns:
            dict con bins, frequenze, curtosi e asimmetria
        """
        # Usa trade filtrati se forniti, altrimenti tutti i trade
        trades_to_use = filtered_trades if filtered_trades is not None else self.trades
        closes = [t for t in trades_to_use if t['direction'] == 'out']
        
        if not closes:
            return {}
        
        profits = [t['profit'] for t in closes]
        
        if len(profits) < 3:
            return {}
        
        # Trova min e max
        min_profit = min(profits)
        max_profit = max(profits)
        
        # Crea bins equidistanti (30 bins)
        num_bins = min(30, len(profits))  # Max 30 bins o numero trade se minore
        bin_edges = np.linspace(min_profit, max_profit, num_bins + 1)
        
        # Calcola frequenze
        hist, edges = np.histogram(profits, bins=bin_edges)
        
        # Prepara bin labels (centri dei bin)
        bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
        
        # Calcola curtosi e asimmetria
        # Curtosi: misura "code pesanti" della distribuzione
        # Valori normali: ~3 (distribuzione normale)
        # >3 = code pesanti (più estremi), <3 = code leggere
        mean_profit = np.mean(profits)
        std_profit = np.std(profits, ddof=1)  # sample std
        
        if std_profit > 0:
            # Curtosi (kurtosis) - usando formula standard
            n = len(profits)
            fourth_moment = sum([(p - mean_profit)**4 for p in profits]) / n
            kurtosis = (fourth_moment / (std_profit**4)) - 3  # Excess kurtosis
            
            # Asimmetria (skewness)
            third_moment = sum([(p - mean_profit)**3 for p in profits]) / n
            skewness = third_moment / (std_profit**3)
        else:
            kurtosis = 0
            skewness = 0
        
        return {
            'bin_centers': [round(x, 2) for x in bin_centers],
            'frequencies': hist.tolist(),
            'bin_edges': [round(x, 2) for x in edges.tolist()],
            'kurtosis': round(kurtosis, 3),
            'skewness': round(skewness, 3),
            'min_profit': round(min_profit, 2),
            'max_profit': round(max_profit, 2),
            'mean_profit': round(mean_profit, 2),
            'std_profit': round(std_profit, 2)
        }
    
    def generate_output(self, risk_free_rate=0, custom_balance=None, trade_type='all'):
        """Genera output JSON completo con supporto filtro trade type"""
        # Parse (se non già fatto)
        if not self.trades:
            self.parse_html()
        
        # Applica filtro trade
        filtered_trades = self.filter_trades_by_type(trade_type)
        
        # Calcola tutto con trade filtrati
        duration_stats = self.calculate_trade_duration()  # Mantiene tutti i trade per durata
        equity_line = self.create_equity_line(custom_balance, filtered_trades)
        drawdowns = self.calculate_drawdowns(equity_line)
        dd_series = self.calculate_drawdown_series(equity_line)
        metrics = self.calculate_metrics(custom_balance, trade_type)
        advanced_metrics = self.calculate_advanced_metrics(risk_free_rate, custom_balance, trade_type)
        mae_mfe = self.calculate_mae_mfe()  # Mantiene tutti i trade
        commission_swap = self.calculate_commission_swap_totals()  # Mantiene tutti i trade
        lot_check = self.check_uniform_lot_size()
        
        # Statistiche temporali con trade filtrati
        weekday_stats_full = self.calculate_weekday_statistics(last_6_months_only=False, filtered_trades=filtered_trades)
        weekday_stats_6m = self.calculate_weekday_statistics(last_6_months_only=True, filtered_trades=filtered_trades)
        monthly_stats = self.calculate_monthly_statistics(filtered_trades=filtered_trades)
        returns_dist = self.calculate_returns_distribution(filtered_trades=filtered_trades)
        
        # Combina tutte le metriche
        metrics.update(duration_stats)
        metrics.update(advanced_metrics)
        metrics.update(mae_mfe)
        metrics.update(commission_swap)
        
        # Prepara tabella raw con tutte le informazioni
        raw_data = []
        
        # Trova coppie di trade in/out
        for i in range(len(self.trades) - 1):
            if self.trades[i]['direction'] == 'in' and self.trades[i+1]['direction'] == 'out':
                entry = self.trades[i]
                exit = self.trades[i+1]
                
                # Separa data e ora
                entry_datetime = entry['datetime'].split(' ')
                exit_datetime = exit['datetime'].split(' ')
                
                raw_data.append({
                    'entry_date': entry_datetime[0] if len(entry_datetime) > 0 else entry['datetime'],
                    'entry_time': entry_datetime[1] if len(entry_datetime) > 1 else '',
                    'exit_date': exit_datetime[0] if len(exit_datetime) > 0 else exit['datetime'],
                    'exit_time': exit_datetime[1] if len(exit_datetime) > 1 else '',
                    'symbol': exit['symbol'],
                    'type': entry['type'],  # buy o sell
                    'volume': entry['volume'],  # lot size
                    'entry_price': entry['price'],
                    'exit_price': exit['price'],
                    'commission': exit['commission'],
                    'swap': exit['swap'],
                    'profit': exit['profit'],
                    'balance': exit['balance'],
                    'comment': entry['comment']  # Commento dell'ENTRATA
                })
        
        output = {
            'strategy_name': self.strategy_name if self.strategy_name else self.html_path.split('/')[-1].replace('.html', ''),
            'metrics': metrics,
            'equity_line': equity_line,
            'drawdown_series': dd_series,
            'drawdowns': drawdowns,
            'raw_data': raw_data,
            'lot_check': lot_check,
            'original_balance': self.original_balance,
            'weekday_stats': {
                'full_history': weekday_stats_full,
                'last_6_months': weekday_stats_6m
            },
            'monthly_stats': monthly_stats,
            'returns_distribution': returns_dist
        }
        
        return output


# Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = StrategyParser(sys.argv[1])
        result = parser.generate_output()
        print(json.dumps(result, indent=2))
