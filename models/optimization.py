import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptimizationEngine:
    """
    Power Usage Optimization Engine
    Provides suggestions for reducing power consumption and costs
    """
    
    def __init__(self):
        # Default electricity rates (can be customized)
        self.peak_rate = 0.15  # $/kWh during peak hours
        self.off_peak_rate = 0.08  # $/kWh during off-peak hours
        self.standard_rate = 0.12  # $/kWh during standard hours
        
        # Peak hours definition (6 PM - 10 PM)
        self.peak_hours = list(range(18, 22))
        self.off_peak_hours = list(range(22, 6)) + list(range(0, 6))
        
        # Optimization thresholds
        self.high_usage_threshold = 4.0  # kWh
        self.cost_savings_threshold = 0.10  # 10% minimum savings to suggest
        
    def get_optimization_suggestions(self, data):
        """
        Generate optimization suggestions based on usage patterns
        """
        suggestions = []
        
        # Validate input data
        if data is None or len(data) == 0:
            return suggestions
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            return suggestions
        
        # Check if required columns exist
        if 'usage' not in data.columns:
            return suggestions
        
        # Analyze usage patterns
        usage_patterns = self.analyze_usage_patterns(data)
        
        # Check for high usage periods
        high_usage_periods = self.identify_high_usage_periods(data)
        if not high_usage_periods.empty:
            suggestions.append({
                'type': 'high_usage',
                'title': 'High Usage Periods Detected',
                'description': f'Usage exceeds {self.high_usage_threshold} kWh during {len(high_usage_periods)} hours',
                'impact': 'high',
                'potential_savings': self.calculate_potential_savings(high_usage_periods),
                'actions': [
                    'Identify appliances running during peak hours',
                    'Consider shifting non-essential loads to off-peak hours',
                    'Check for malfunctioning equipment'
                ]
            })
        
        # Check for peak hour usage
        peak_usage = self.analyze_peak_hour_usage(data)
        if peak_usage['percentage'] > 0.3:  # More than 30% usage during peak hours
            suggestions.append({
                'type': 'peak_hour_usage',
                'title': 'High Peak Hour Usage',
                'description': f'{peak_usage["percentage"]:.1%} of usage occurs during peak hours (6 PM - 10 PM)',
                'impact': 'medium',
                'potential_savings': peak_usage['potential_savings'],
                'actions': [
                    'Schedule laundry and dishwashing for off-peak hours',
                    'Use programmable thermostats to reduce HVAC during peak hours',
                    'Consider time-of-use rate plans'
                ]
            })
        
        # Check for weekend vs weekday patterns
        weekend_analysis = self.analyze_weekend_patterns(data)
        if weekend_analysis['difference'] > 1.0:  # More than 1 kWh difference
            suggestions.append({
                'type': 'weekend_pattern',
                'title': 'Weekend Usage Pattern',
                'description': f'Weekend usage is {weekend_analysis["difference"]:.1f} kWh higher than weekdays',
                'impact': 'low',
                'potential_savings': weekend_analysis['potential_savings'],
                'actions': [
                    'Review weekend appliance usage',
                    'Consider energy-efficient weekend activities',
                    'Optimize weekend schedule for energy savings'
                ]
            })
        
        # Check for temperature correlation
        temp_analysis = self.analyze_temperature_correlation(data)
        if temp_analysis['correlation'] > 0.7:  # Strong temperature correlation
            suggestions.append({
                'type': 'temperature_correlation',
                'title': 'Temperature-Dependent Usage',
                'description': f'Usage strongly correlates with temperature (r={temp_analysis["correlation"]:.2f})',
                'impact': 'medium',
                'potential_savings': temp_analysis['potential_savings'],
                'actions': [
                    'Improve home insulation',
                    'Use programmable thermostats',
                    'Consider energy-efficient HVAC systems',
                    'Optimize temperature settings'
                ]
            })
        
        # Check for unusual patterns
        unusual_patterns = self.detect_unusual_patterns(data)
        if unusual_patterns:
            suggestions.append({
                'type': 'unusual_patterns',
                'title': 'Unusual Usage Patterns',
                'description': f'Detected {len(unusual_patterns)} unusual usage patterns',
                'impact': 'high',
                'potential_savings': sum([p['potential_savings'] for p in unusual_patterns]),
                'actions': [
                    'Investigate unusual usage spikes',
                    'Check for equipment malfunctions',
                    'Review recent changes in usage patterns'
                ]
            })
        
        return suggestions
    
    def analyze_usage_patterns(self, data):
        """
        Analyze overall usage patterns
        """
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        patterns = {
            'total_usage': df['usage'].sum(),
            'average_usage': df['usage'].mean(),
            'peak_usage': df['usage'].max(),
            'usage_std': df['usage'].std(),
            'total_hours': len(df),
            'peak_hour_usage': df[df['hour'].isin(self.peak_hours)]['usage'].sum() if 'hour' in df.columns else 0,
            'off_peak_usage': df[df['hour'].isin(self.off_peak_hours)]['usage'].sum() if 'hour' in df.columns else 0
        }
        
        return patterns
    
    def calculate_potential_savings(self, high_usage_data):
        """
        Calculate potential savings from reducing high usage periods
        """
        if high_usage_data.empty:
            return 0
        
        # Calculate excess usage above threshold
        excess_usage = high_usage_data['usage'].sum() - (self.high_usage_threshold * len(high_usage_data))
        
        # Assume 50% of excess usage can be reduced
        reducible_usage = excess_usage * 0.5
        
        # Calculate savings at standard rate
        potential_savings = reducible_usage * self.standard_rate
        
        return potential_savings
    
    def identify_high_usage_periods(self, data):
        """
        Identify periods with unusually high usage
        """
        high_usage = data[data['usage'] > self.high_usage_threshold]
        return high_usage
    
    def analyze_peak_hour_usage(self, data):
        """
        Analyze usage during peak hours
        """
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
        
        if 'hour' not in df.columns:
            return {'percentage': 0, 'potential_savings': 0}
        
        peak_usage = df[df['hour'].isin(self.peak_hours)]['usage'].sum()
        total_usage = df['usage'].sum()
        peak_percentage = peak_usage / total_usage if total_usage > 0 else 0
        
        # Calculate potential savings by shifting 20% of peak usage to off-peak
        shiftable_usage = peak_usage * 0.2
        potential_savings = shiftable_usage * (self.peak_rate - self.off_peak_rate)
        
        return {
            'percentage': peak_percentage,
            'peak_usage': peak_usage,
            'potential_savings': potential_savings
        }
    
    def analyze_weekend_patterns(self, data):
        """
        Compare weekend vs weekday usage patterns
        """
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        if 'is_weekend' not in df.columns:
            return {'difference': 0, 'potential_savings': 0}
        
        weekday_avg = df[df['is_weekend'] == 0]['usage'].mean()
        weekend_avg = df[df['is_weekend'] == 1]['usage'].mean()
        
        difference = weekend_avg - weekday_avg if not pd.isna(weekend_avg) and not pd.isna(weekday_avg) else 0
        
        # Potential savings by reducing weekend usage by 10%
        weekend_hours = len(df[df['is_weekend'] == 1])
        potential_savings = weekend_hours * difference * 0.1 * self.standard_rate
        
        return {
            'difference': difference,
            'weekday_avg': weekday_avg,
            'weekend_avg': weekend_avg,
            'potential_savings': potential_savings
        }
    
    def analyze_temperature_correlation(self, data):
        """
        Analyze correlation between temperature and usage
        """
        df = data.copy()
        
        if 'temperature' not in df.columns:
            return {'correlation': 0, 'potential_savings': 0}
        
        # Calculate correlation
        correlation = df['usage'].corr(df['temperature'])
        
        if pd.isna(correlation):
            correlation = 0
        
        # Estimate potential savings from temperature optimization
        # Assume 10% reduction in temperature-dependent usage
        temp_dependent_usage = abs(correlation) * df['usage'].sum()
        potential_savings = temp_dependent_usage * 0.1 * self.standard_rate
        
        return {
            'correlation': correlation,
            'temp_dependent_usage': temp_dependent_usage,
            'potential_savings': potential_savings
        }
    
    def detect_unusual_patterns(self, data):
        """
        Detect unusual usage patterns that might indicate problems
        """
        unusual_patterns = []
        df = data.copy()
        
        # Detect sudden spikes (more than 2 standard deviations above mean)
        mean_usage = df['usage'].mean()
        std_usage = df['usage'].std()
        threshold = mean_usage + 2 * std_usage
        
        spikes = df[df['usage'] > threshold]
        
        for _, row in spikes.iterrows():
            unusual_patterns.append({
                'type': 'usage_spike',
                'timestamp': row.get('timestamp', 'Unknown'),
                'usage': row['usage'],
                'threshold': threshold,
                'potential_savings': (row['usage'] - mean_usage) * self.standard_rate
            })
        
        # Detect continuous high usage (more than 3 hours above threshold)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            high_usage_mask = df['usage'] > self.high_usage_threshold
            high_usage_groups = (high_usage_mask != high_usage_mask.shift()).cumsum()
            
            for group_id in high_usage_groups.unique():
                group_data = df[high_usage_groups == group_id]
                if len(group_data) >= 3 and group_data['usage'].mean() > self.high_usage_threshold:
                    unusual_patterns.append({
                        'type': 'continuous_high_usage',
                        'start_time': group_data['timestamp'].iloc[0],
                        'end_time': group_data['timestamp'].iloc[-1],
                        'duration_hours': len(group_data),
                        'avg_usage': group_data['usage'].mean(),
                        'potential_savings': len(group_data) * (group_data['usage'].mean() - mean_usage) * self.standard_rate
                    })
        
        return unusual_patterns
    
    def calculate_cost(self, data):
        """
        Calculate current electricity cost
        """
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
        
        if 'hour' not in df.columns:
            return df['usage'].sum() * self.standard_rate
        
        # Calculate costs for different time periods
        peak_usage = df[df['hour'].isin(self.peak_hours)]['usage'].sum()
        off_peak_usage = df[df['hour'].isin(self.off_peak_hours)]['usage'].sum()
        standard_usage = df['usage'].sum() - peak_usage - off_peak_usage
        
        total_cost = (peak_usage * self.peak_rate + 
                     off_peak_usage * self.off_peak_rate + 
                     standard_usage * self.standard_rate)
        
        return total_cost
    
    def calculate_optimized_cost(self, data):
        """
        Calculate potential cost with optimizations applied
        """
        current_cost = self.calculate_cost(data)
        
        # Apply optimization strategies
        suggestions = self.get_optimization_suggestions(data)
        
        total_savings = sum([s.get('potential_savings', 0) for s in suggestions])
        
        optimized_cost = current_cost - total_savings
        
        return max(optimized_cost, 0)  # Ensure cost doesn't go negative
    
    def get_detailed_analysis(self, data):
        """
        Get detailed analysis of usage patterns and optimization opportunities
        """
        analysis = {
            'usage_summary': self.analyze_usage_patterns(data),
            'cost_analysis': {
                'current_cost': self.calculate_cost(data),
                'optimized_cost': self.calculate_optimized_cost(data),
                'potential_savings': self.calculate_cost(data) - self.calculate_optimized_cost(data)
            },
            'peak_analysis': self.analyze_peak_hour_usage(data),
            'weekend_analysis': self.analyze_weekend_patterns(data),
            'temperature_analysis': self.analyze_temperature_correlation(data),
            'unusual_patterns': self.detect_unusual_patterns(data),
            'optimization_suggestions': self.get_optimization_suggestions(data)
        }
        
        return analysis
