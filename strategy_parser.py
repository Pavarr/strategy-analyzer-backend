from bs4 import BeautifulSoup
from datetime import datetime
import json

class StrategyParser:
    def __init__(self, html_path):
        self.html_path = html_path
        self.trades = []
        self.metrics = {}
        
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
        return self.trades
    
    def calculate_trade_duration(self):
        """Calcola durata media dei trade (coppia in/out sequenziali)"""
        durations = []
        
        # I trade in/out sono sequenziali
        for i in range(len(self.trades) - 1):
            current = self.trades[i]
            next_trade = self.trades[i + 1]
            
            if current['direction'] == 'in' and next_trade['direction'] == 'out':
                dt1 = datetime.strptime(current['datetime'], '%Y.%m.%d %H:%M:%S')
                dt2 = datetime.strptime(next_trade['datetime'], '%Y.%m.%d %H:%M:%S')
                duration_seconds = (dt2 - dt1).total_seconds()
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
    
    def create_equity_line(self):
        """Crea equity line usando solo le chiusure (out)"""
        equity_line = {
            'dates': [],
            'values': []
        }
        
        for trade in self.trades:
            if trade['direction'] == 'out':
                equity_line['dates'].append(trade['datetime'])
                equity_line['values'].append(trade['balance'])
        
        return equity_line
    
    def calculate_drawdowns(self):
        """Calcola tabella drawdown"""
        # Ottieni equity line
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
                # Nuovo peak raggiunto
                if current_peak_idx < i - 1:
                    # C'era un drawdown precedente
                    trough_idx = current_peak_idx
                    trough = equity[current_peak_idx]
                    
                    for j in range(current_peak_idx, i):
                        if equity[j] < trough:
                            trough = equity[j]
                            trough_idx = j
                    
                    dd_pct = ((trough - current_peak) / current_peak) * 100
                    
                    # Trova recovery
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
        
        # Controlla drawdown in corso
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
        
        # Ordina per drawdown %
        drawdowns.sort(key=lambda x: x['drawdown_pct'])
        
        return drawdowns
    
    def calculate_metrics(self):
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
        equity_line = self.create_equity_line()
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
        starting_balance = closes[0]['balance'] - sum(profits)
        ending_balance = closes[-1]['balance']
        
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
    
    def generate_output(self):
        """Genera output JSON completo"""
        # Parse
        self.parse_html()
        
        # Calcola tutto
        duration_stats = self.calculate_trade_duration()
        equity_line = self.create_equity_line()
        drawdowns = self.calculate_drawdowns()
        metrics = self.calculate_metrics()
        
        # Combina metriche con durata
        metrics.update(duration_stats)
        
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
            'drawdowns': drawdowns,
            'raw_data': raw_data
        }
        
        return output


# Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = StrategyParser(sys.argv[1])
        result = parser.generate_output()
        print(json.dumps(result, indent=2))
