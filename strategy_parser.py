import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder per numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class StrategyParser:
    def __init__(self, html_path):
        self.html_path = html_path
        self.trades_df = None
        self.equity_df = None
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
        
        self.trades_df = pd.DataFrame(trades_data)
        self.trades_df['datetime'] = pd.to_datetime(self.trades_df['datetime'], format='%Y.%m.%d %H:%M:%S')
        
        return self.trades_df
    
    def calculate_trade_duration(self):
        """Calcola durata media dei trade (coppia in/out sequenziali)"""
        durations = []
        
        # I trade in/out sono sequenziali: in (order N) → out (order N+1)
        for i in range(len(self.trades_df) - 1):
            current = self.trades_df.iloc[i]
            next_trade = self.trades_df.iloc[i + 1]
            
            # Se current è "in" e next è "out", è una coppia
            if current['direction'] == 'in' and next_trade['direction'] == 'out':
                duration_seconds = (next_trade['datetime'] - current['datetime']).total_seconds()
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
        
        avg_duration_seconds = np.mean(durations)
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
        closes = self.trades_df[self.trades_df['direction'] == 'out'].copy()
        
        self.equity_df = closes[['datetime', 'balance']].reset_index(drop=True)
        self.equity_df.columns = ['date', 'equity']
        
        return self.equity_df
    
    def calculate_drawdowns(self):
        """Calcola tabella drawdown con peak, trough, recovery"""
        equity = self.equity_df['equity'].values
        dates = self.equity_df['date'].values
        
        # Trova tutti i peak (massimi locali)
        peaks = []
        current_peak = equity[0]
        current_peak_idx = 0
        
        drawdowns = []
        
        for i in range(len(equity)):
            if equity[i] > current_peak:
                # Nuovo peak raggiunto
                if current_peak_idx < i - 1:
                    # C'era un drawdown precedente, calcoliamolo
                    trough_idx = np.argmin(equity[current_peak_idx:i])
                    trough_idx += current_peak_idx
                    trough = equity[trough_idx]
                    
                    dd_pct = ((trough - current_peak) / current_peak) * 100
                    
                    # Trova recovery (se esiste)
                    recovery_idx = None
                    for j in range(trough_idx + 1, len(equity)):
                        if equity[j] >= current_peak:
                            recovery_idx = j
                            break
                    
                    if recovery_idx is not None:
                        recovery_days = (dates[recovery_idx] - dates[trough_idx]).astype('timedelta64[D]').astype(int)
                    else:
                        recovery_days = None
                    
                    drawdowns.append({
                        'peak_date': str(dates[current_peak_idx])[:10],
                        'trough_date': str(dates[trough_idx])[:10],
                        'recovery_date': str(dates[recovery_idx])[:10] if recovery_idx else 'In corso',
                        'peak': round(current_peak, 2),
                        'trough': round(trough, 2),
                        'drawdown_pct': round(dd_pct, 2),
                        'recovery_days': recovery_days if recovery_days else 'In corso'
                    })
                
                current_peak = equity[i]
                current_peak_idx = i
        
        # Controlla se c'è un drawdown in corso alla fine
        if current_peak_idx < len(equity) - 1:
            trough_idx = np.argmin(equity[current_peak_idx:])
            trough_idx += current_peak_idx
            trough = equity[trough_idx]
            dd_pct = ((trough - current_peak) / current_peak) * 100
            
            drawdowns.append({
                'peak_date': str(dates[current_peak_idx])[:10],
                'trough_date': str(dates[trough_idx])[:10],
                'recovery_date': 'In corso',
                'peak': round(current_peak, 2),
                'trough': round(trough, 2),
                'drawdown_pct': round(dd_pct, 2),
                'recovery_days': 'In corso'
            })
        
        return sorted(drawdowns, key=lambda x: x['drawdown_pct'])
    
    def calculate_metrics(self):
        """Calcola metriche principali"""
        closes = self.trades_df[self.trades_df['direction'] == 'out']
        
        # Win rate
        wins = len(closes[closes['profit'] > 0])
        losses = len(closes[closes['profit'] < 0])
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Profit/Loss statistics
        avg_profit = closes['profit'].mean()
        total_profit = closes['profit'].sum()
        
        avg_win = closes[closes['profit'] > 0]['profit'].mean() if wins > 0 else 0
        avg_loss = closes[closes['profit'] < 0]['profit'].mean() if losses > 0 else 0
        
        # Max DD
        equity = self.equity_df['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_dd = drawdown.min()
        
        # Starting and ending balance
        starting_balance = self.equity_df['equity'].iloc[0] - closes['profit'].sum()
        ending_balance = self.equity_df['equity'].iloc[-1]
        
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
        self.create_equity_line()
        drawdowns = self.calculate_drawdowns()
        metrics = self.calculate_metrics()
        
        # Combina metriche con durata
        metrics.update(duration_stats)
        
        # Prepara tabella raw
        raw_table = self.equity_df.copy()
        raw_table['asset'] = self.trades_df[self.trades_df['direction'] == 'out']['symbol'].values
        raw_table = raw_table[['date', 'asset', 'equity']]
        raw_table['date'] = raw_table['date'].astype(str)
        raw_table['equity'] = raw_table['equity'].astype(float)
        
        # Converti equity_line values in liste Python native
        equity_values = [float(v) for v in self.equity_df['equity'].tolist()]
        
        output = {
            'strategy_name': self.html_path.split('/')[-1].replace('.html', ''),
            'metrics': metrics,
            'equity_line': {
                'dates': self.equity_df['date'].astype(str).tolist(),
                'values': equity_values
            },
            'drawdowns': drawdowns,
            'raw_data': raw_table.to_dict('records')
        }
        
        return output


# Test con il file caricato
if __name__ == "__main__":
    parser = StrategyParser('/mnt/user-data/uploads/012_QQQ_D_WZQ_OOS1.html')
    result = parser.generate_output()
    
    # Salva JSON
    with open('/home/claude/output.json', 'w') as f:
        json.dump(result, indent=2, fp=f, cls=NumpyEncoder)
    
    print("✅ Parsing completato")
    print(f"\nMetriche principali:")
    print(f"  Total trades: {result['metrics']['total_trades']}")
    print(f"  Win rate: {result['metrics']['win_rate_pct']}%")
    print(f"  Total profit: ${result['metrics']['total_profit']}")
    print(f"  Max DD: {result['metrics']['max_drawdown_pct']}%")
    print(f"  Avg duration: {result['metrics']['avg_duration_formatted']}")
    print(f"\nDrawdown trovati: {len(result['drawdowns'])}")
    print(f"Equity points: {len(result['equity_line']['dates'])}")
