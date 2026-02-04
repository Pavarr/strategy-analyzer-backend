from bs4 import BeautifulSoup
from datetime import datetime
import json
import math

class StrategyParser:
    def __init__(self, html_path):
        self.html_path = html_path
        self.trades = []
        self.metrics = {}
        self.original_balance = None
        self.original_lot_size = None
        
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
    
    def create_equity_line(self, custom_balance=None):
        """Crea equity line usando solo le chiusure"""
        equity_line = {
            'dates': [],
            'values': []
        }
        
        closes = [t for t in self.trades if t['direction'] == 'out']
        
        if custom_balance:
            # Usa balance custom
            running_balance = custom_balance
            for trade in closes:
                running_balance += trade['profit']
                equity_line['dates'].append(trade['datetime'])
                equity_line['values'].append(running_balance)
        else:
            # Usa balance originale
            for trade in closes:
                equity_line['dates'].append(trade['datetime'])
                equity_line['values'].append(trade['balance'])
        
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
    
    def calculate_advanced_metrics(self, risk_free_rate=0, custom_balance=None):
        """Calcola metriche avanzate"""
        closes = [t for t in self.trades if t['direction'] == 'out']
        
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
        
        # Calcola returns per Sharpe/Sortino
        equity_line = self.create_equity_line(custom_balance)
        equity_values = equity_line['values']
        
        # Starting balance
        starting_balance = custom_balance if custom_balance else (closes[0]['balance'] - closes[0]['profit'])
        
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
        
        return {
            'profit_factor': round(profit_factor, 2),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'kelly_pct': round(kelly_pct, 2),
            'sharpe_ratio': round(sharpe_annualized, 2),
            'sortino_ratio': round(sortino_annualized, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'recovery_factor': round(recovery_factor, 3),
            'annualized_return': round(annualized_return, 2),
            'volatility_annual': round(volatility * math.sqrt(trades_per_year) * 100, 2)
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
            'volatility_annual': 0
        }
    
    def calculate_metrics(self, custom_balance=None):
        """Calcola metriche principali"""
        closes = [t for t in self.trades if t['direction'] == 'out']
        
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
        
        loss_profits = [t['profit'] for t in closes if t['profit'] < 0]
        avg_loss = sum(loss_profits) / len(loss_profits) if loss_profits else 0
        
        # Max DD
        equity_line = self.create_equity_line(custom_balance)
        equity = equity_line['values']
        
        max_dd = 0
        running_max = equity[0]
        
        for e in equity:
            if e > running_max:
                running_max = e
            drawdown = (e - running_max) / running_max * 100
            if drawdown < max_dd:
                max_dd = drawdown
        
        # Starting and ending balance
        starting_balance = custom_balance if custom_balance else (closes[0]['balance'] - closes[0]['profit'])
        ending_balance = equity[-1] if equity else starting_balance
        
        # Total return
        total_return_pct = ((ending_balance - starting_balance) / starting_balance) * 100
        
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
    
    def generate_output(self, risk_free_rate=0, custom_balance=None):
        """Genera output JSON completo"""
        # Parse (se non già fatto)
        if not self.trades:
            self.parse_html()
        
        # Calcola tutto
        duration_stats = self.calculate_trade_duration()
        equity_line = self.create_equity_line(custom_balance)
        drawdowns = self.calculate_drawdowns(equity_line)
        dd_series = self.calculate_drawdown_series(equity_line)
        metrics = self.calculate_metrics(custom_balance)
        advanced_metrics = self.calculate_advanced_metrics(risk_free_rate, custom_balance)
        mae_mfe = self.calculate_mae_mfe()
        commission_swap = self.calculate_commission_swap_totals()
        lot_check = self.check_uniform_lot_size()
        
        # Combina tutte le metriche
        metrics.update(duration_stats)
        metrics.update(advanced_metrics)
        metrics.update(mae_mfe)
        metrics.update(commission_swap)
        
        # Prepara tabella raw
        raw_data = []
        for trade in self.trades:
            if trade['direction'] == 'out':
                raw_data.append({
                    'date': trade['datetime'],
                    'asset': trade['symbol'],
                    'equity': trade['balance']
                })
        
        output = {
            'strategy_name': self.html_path.split('/')[-1].replace('.html', ''),
            'metrics': metrics,
            'equity_line': equity_line,
            'drawdown_series': dd_series,
            'drawdowns': drawdowns,
            'raw_data': raw_data,
            'lot_check': lot_check,
            'original_balance': self.original_balance
        }
        
        return output


# Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = StrategyParser(sys.argv[1])
        result = parser.generate_output()
        print(json.dumps(result, indent=2))
