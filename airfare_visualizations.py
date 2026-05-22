"""
Airfare Data Visualizations
Author: Prashant Anand
Description: Generate charts and plots from cleaned airfare data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='darkgrid', palette='muted')
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.6,
    'font.family':      'serif',
    'figure.titlesize': 15,
})

ACCENT  = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'
ACCENT4 = '#d2a8ff'


def load_data():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from airfare_pipeline import (
        generate_raw_airfare_data, clean_airfare_data, engineer_features
    )
    raw     = generate_raw_airfare_data(n=500)
    cleaned = clean_airfare_data(raw)
    return engineer_features(cleaned)


def plot_all(df):
    fig = plt.figure(figsize=(18, 14), facecolor='#0d1117')
    fig.suptitle('✈  Airfare Data Analysis Dashboard', fontsize=17,
                 color='#e6edf3', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1. Price distribution ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['price_usd'], bins=30, color=ACCENT, alpha=0.85, edgecolor='#0d1117')
    ax1.axvline(df['price_usd'].mean(), color=ACCENT2, lw=1.8, linestyle='--', label=f"Mean ${df['price_usd'].mean():.0f}")
    ax1.axvline(df['price_usd'].median(), color=ACCENT3, lw=1.8, linestyle='--', label=f"Median ${df['price_usd'].median():.0f}")
    ax1.set_title('Price Distribution', color='#e6edf3')
    ax1.set_xlabel('Price (USD)')
    ax1.set_ylabel('Count')
    ax1.legend(fontsize=8)

    # ── 2. Avg price by airline ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    airline_avg = df.groupby('airline')['price_usd'].mean().sort_values(ascending=True).tail(7)
    bars = ax2.barh(airline_avg.index, airline_avg.values, color=ACCENT4, alpha=0.85)
    for bar, val in zip(bars, airline_avg.values):
        ax2.text(val + 5, bar.get_y() + bar.get_height()/2,
                 f'${val:.0f}', va='center', fontsize=8, color='#c9d1d9')
    ax2.set_title('Avg Price by Airline', color='#e6edf3')
    ax2.set_xlabel('Avg Price (USD)')

    # ── 3. Price by stops ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    stops_avg = df.groupby('stops')['price_usd'].mean()
    colors = [ACCENT3, ACCENT2, '#e3b341']
    ax3.bar([f"{s} stop{'s' if s!=1 else ''}" for s in stops_avg.index],
            stops_avg.values, color=colors[:len(stops_avg)], alpha=0.85)
    for i, val in enumerate(stops_avg.values):
        ax3.text(i, val + 10, f'${val:.0f}', ha='center', fontsize=9, color='#c9d1d9')
    ax3.set_title('Avg Price by Stops', color='#e6edf3')
    ax3.set_ylabel('Avg Price (USD)')

    # ── 4. Price by booking window ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    bw_order = ['Last-min', '1-2wk', '2-4wk', '1-2mo', '2mo+']
    bw_avg   = df.groupby('booking_window', observed=True)['price_usd'].mean().reindex(bw_order)
    ax4.plot(bw_order, bw_avg.values, color=ACCENT, marker='o', lw=2, ms=7)
    ax4.fill_between(range(len(bw_order)), bw_avg.values, alpha=0.15, color=ACCENT)
    ax4.set_xticks(range(len(bw_order)))
    ax4.set_xticklabels(bw_order, fontsize=8)
    ax4.set_title('Price vs Booking Window', color='#e6edf3')
    ax4.set_ylabel('Avg Price (USD)')

    # ── 5. Price by cabin class ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    cabin_avg = df.groupby('cabin_class')['price_usd'].mean().sort_values(ascending=False)
    cabin_colors = [ACCENT2, ACCENT4, ACCENT, ACCENT3]
    wedges, texts, autotexts = ax5.pie(
        cabin_avg.values,
        labels=cabin_avg.index,
        autopct='%1.0f%%',
        colors=cabin_colors[:len(cabin_avg)],
        startangle=90,
        textprops={'color': '#c9d1d9', 'fontsize': 8}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color('#0d1117')
    ax5.set_title('Price Share by Cabin Class', color='#e6edf3')

    # ── 6. Top routes ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    route_avg = df.groupby('route')['price_usd'].mean().sort_values(ascending=False).head(6)
    ax6.barh(route_avg.index, route_avg.values, color=ACCENT, alpha=0.85)
    for i, val in enumerate(route_avg.values):
        ax6.text(val + 5, i, f'${val:.0f}', va='center', fontsize=8, color='#c9d1d9')
    ax6.set_title('Top 6 Most Expensive Routes', color='#e6edf3')
    ax6.set_xlabel('Avg Price (USD)')
    ax6.invert_yaxis()

    # ── 7. Price by month ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    month_avg = df.groupby('month')['price_usd'].mean()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    ax7.bar([month_names[m-1] for m in month_avg.index],
            month_avg.values, color=ACCENT3, alpha=0.8)
    ax7.set_title('Avg Price by Month', color='#e6edf3')
    ax7.set_ylabel('Avg Price (USD)')
    ax7.tick_params(axis='x', labelsize=7)

    # ── 8. Correlation heatmap ────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    corr_cols = ['price_usd', 'duration_hrs', 'stops', 'days_before_departure',
                 'is_weekend_flight', 'is_holiday_season', 'cabin_rank']
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax8, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                linewidths=0.3, linecolor='#0d1117',
                cbar_kws={'shrink': 0.8})
    ax8.set_title('Feature Correlation', color='#e6edf3')
    ax8.tick_params(axis='x', labelsize=7, rotation=30)
    ax8.tick_params(axis='y', labelsize=7, rotation=0)

    # ── 9. Price vs duration scatter ──────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    sc = ax9.scatter(df['duration_hrs'], df['price_usd'],
                     c=df['stops'], cmap='plasma', alpha=0.5, s=18)
    plt.colorbar(sc, ax=ax9, label='Stops', shrink=0.8)
    ax9.set_title('Price vs Duration (color=stops)', color='#e6edf3')
    ax9.set_xlabel('Duration (hrs)')
    ax9.set_ylabel('Price (USD)')

    plt.savefig('/home/claude/airfare-project/airfare_dashboard.png',
                dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print("✅  Dashboard saved → airfare_dashboard.png")
    plt.close()


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════╗")
    print("║   AIRFARE VISUALIZATIONS                     ║")
    print("║   Author: Prashant Anand                     ║")
    print("╚══════════════════════════════════════════════╝")

    df = load_data()
    print(f"\nGenerating 9-panel dashboard from {len(df)} records...")
    plot_all(df)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   VISUALIZATIONS COMPLETE ✓                  ║")
    print("╚══════════════════════════════════════════════╝")
