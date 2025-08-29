// Power Usage Forecasting Tool - Charts JavaScript

// Chart.js configuration
Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#6c757d';

// Color schemes
const chartColors = {
    primary: '#007bff',
    secondary: '#6c757d',
    success: '#28a745',
    warning: '#ffc107',
    danger: '#dc3545',
    info: '#17a2b8',
    light: '#f8f9fa',
    dark: '#343a40',
    purple: '#6f42c1',
    pink: '#e83e8c',
    teal: '#20c997',
    orange: '#fd7e14'
};

// Usage chart configuration
function createUsageChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels || [],
            datasets: [{
                label: 'Power Usage (kWh)',
                data: data.usage || [],
                borderColor: chartColors.primary,
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: chartColors.primary,
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: chartColors.primary,
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Power Usage (kWh)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Forecast chart with confidence intervals
function createForecastChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const datasets = [
        {
            label: 'Historical Usage',
            data: data.historical_usage || [],
            borderColor: chartColors.primary,
            backgroundColor: 'rgba(0, 123, 255, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.4,
            pointRadius: 3
        },
        {
            label: 'Forecast',
            data: data.forecast_usage || [],
            borderColor: chartColors.success,
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.4,
            pointRadius: 3,
            borderDash: [5, 5]
        }
    ];
    
    // Add confidence intervals if available
    if (data.confidence_upper && data.confidence_lower) {
        datasets.push({
            label: 'Confidence Interval',
            data: data.confidence_upper,
            borderColor: 'rgba(255, 193, 7, 0.3)',
            backgroundColor: 'rgba(255, 193, 7, 0.1)',
            borderWidth: 1,
            fill: '+1',
            tension: 0.4,
            pointRadius: 0
        });
        
        datasets.push({
            label: 'Confidence Lower',
            data: data.confidence_lower,
            borderColor: 'rgba(255, 193, 7, 0.3)',
            backgroundColor: 'rgba(255, 193, 7, 0.1)',
            borderWidth: 1,
            fill: false,
            tension: 0.4,
            pointRadius: 0
        });
    }
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [...(data.historical_timestamps || []), ...(data.forecast_timestamps || [])],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: chartColors.primary,
                    borderWidth: 1,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Power Usage (kWh)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Usage distribution pie chart
function createDistributionChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.labels || ['Low Usage', 'Normal Usage', 'High Usage'],
            datasets: [{
                data: data.values || [30, 50, 20],
                backgroundColor: [
                    chartColors.success,
                    chartColors.primary,
                    chartColors.danger
                ],
                borderColor: '#ffffff',
                borderWidth: 2,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: chartColors.primary,
                    borderWidth: 1,
                    cornerRadius: 8
                }
            }
        }
    });
}

// Hourly usage pattern chart
function createHourlyPatternChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.hours || Array.from({length: 24}, (_, i) => `${i}:00`),
            datasets: [{
                label: 'Average Usage (kWh)',
                data: data.usage || [],
                backgroundColor: chartColors.primary,
                borderColor: chartColors.primary,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: chartColors.primary,
                    borderWidth: 1,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Hour of Day',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Average Usage (kWh)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Cost analysis chart
function createCostChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels || ['Current Cost', 'Optimized Cost', 'Potential Savings'],
            datasets: [{
                label: 'Cost ($)',
                data: data.values || [],
                backgroundColor: [
                    chartColors.danger,
                    chartColors.success,
                    chartColors.warning
                ],
                borderColor: [
                    chartColors.danger,
                    chartColors.success,
                    chartColors.warning
                ],
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: chartColors.primary,
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return `$${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Cost ($)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Temperature correlation chart
function createTemperatureChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Usage vs Temperature',
                data: data.points || [],
                backgroundColor: chartColors.info,
                borderColor: chartColors.info,
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: chartColors.primary,
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return `Usage: ${context.parsed.y.toFixed(2)} kWh, Temp: ${context.parsed.x.toFixed(1)}°C`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Temperature (°C)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Power Usage (kWh)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Weekly pattern chart
function createWeeklyPatternChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: days,
            datasets: [{
                label: 'Average Daily Usage (kWh)',
                data: data.values || [],
                backgroundColor: chartColors.purple,
                borderColor: chartColors.purple,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: chartColors.primary,
                    borderWidth: 1,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Day of Week',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Average Usage (kWh)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Update chart data
function updateChartData(chart, newData) {
    if (!chart) return;
    
    chart.data.labels = newData.labels || chart.data.labels;
    chart.data.datasets.forEach((dataset, index) => {
        if (newData.datasets && newData.datasets[index]) {
            dataset.data = newData.datasets[index].data || dataset.data;
        }
    });
    
    chart.update('active');
}

// Export chart as image
function exportChartAsImage(chart, filename = 'chart.png') {
    if (!chart) return;
    
    const link = document.createElement('a');
    link.download = filename;
    link.href = chart.toBase64Image();
    link.click();
}

// Chart utility functions
function calculateChartStatistics(data) {
    if (!data || data.length === 0) return {};
    
    const sum = data.reduce((a, b) => a + b, 0);
    const avg = sum / data.length;
    const max = Math.max(...data);
    const min = Math.min(...data);
    
    return {
        total: sum,
        average: avg,
        maximum: max,
        minimum: min,
        range: max - min
    };
}

// Export chart functions
window.PowerForecastCharts = {
    createUsageChart,
    createForecastChart,
    createDistributionChart,
    createHourlyPatternChart,
    createCostChart,
    createTemperatureChart,
    createWeeklyPatternChart,
    updateChartData,
    exportChartAsImage,
    calculateChartStatistics,
    chartColors
}; 