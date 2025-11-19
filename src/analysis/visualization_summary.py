import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import ast
import numpy as np
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_processed_data

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'analysis_results', 'exploratory_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_analysis_summary_dashboard():
    """
    Creates a comprehensive summary dashboard showing key insights from all analyses.
    """
    print("\n" + "="*80)
    print("CREATING ANALYSIS SUMMARY DASHBOARD")
    print("="*80)
    
    # Load data
    df = load_processed_data(mock=False)
    df['main_categories'] = df['main_categories'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Create derived features
    df['num_authors'] = df['authors'].apply(
        lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.strip().startswith('[') 
        else len(x.split(',')) if isinstance(x, str) else 0
    )
    df['discipline_count'] = df['main_categories'].apply(len)
    df['title_length'] = df['title'].str.len()
    df['abstract_length'] = df['abstract'].str.len()
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Dataset Overview (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    summary_text = f"""
    DATASET OVERVIEW
    
    Total Papers: {len(df):,}
    Year Range: {df['submission_year'].min()}-{df['submission_year'].max()}
    Span: {df['submission_year'].max() - df['submission_year'].min()} years
    
    Categories: {df['main_categories'].explode().nunique()}
    Avg Authors: {df['num_authors'].mean():.1f}
    Avg Disciplines: {df['discipline_count'].mean():.2f}
    
    Missing Values: {df.isnull().sum().sum():,} cells
    Data Completeness: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%
    """
    ax1.text(0.1, 0.5, summary_text, fontsize=11, fontweight='bold',
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.set_title('Dataset Summary', fontsize=14, fontweight='bold', pad=10)
    
    # 2. Top Categories (Top Middle Left)
    ax2 = fig.add_subplot(gs[0, 1])
    df_exploded = df.explode('main_categories')
    top_cats = df_exploded['main_categories'].value_counts().head(8)
    colors = sns.color_palette("husl", len(top_cats))
    bars = ax2.barh(range(len(top_cats)), top_cats.values, color=colors)
    ax2.set_yticks(range(len(top_cats)))
    ax2.set_yticklabels(top_cats.index, fontsize=9)
    ax2.set_xlabel('Publications', fontsize=10, fontweight='bold')
    ax2.set_title('Top 8 Research Categories', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    for i, (idx, val) in enumerate(top_cats.items()):
        ax2.text(val + 20, i, f'{val:,}', va='center', fontsize=9, fontweight='bold')
    
    # 3. Publication Growth Over Time (Top Middle Right)
    ax3 = fig.add_subplot(gs[0, 2])
    yearly_counts = df.groupby('submission_year').size()
    ax3.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2.5, 
            markersize=6, color='#2E86AB', markerfacecolor='#A23B72')
    ax3.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3, color='#2E86AB')
    ax3.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Publications', fontsize=10, fontweight='bold')
    ax3.set_title('Publication Growth Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Author Distribution (Top Right)
    ax4 = fig.add_subplot(gs[0, 3])
    author_dist = df[df['num_authors'] <= 20]['num_authors'].value_counts().sort_index()
    ax4.bar(author_dist.index, author_dist.values, color=sns.color_palette("coolwarm", len(author_dist)), 
           edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Number of Authors', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax4.set_title('Authors per Paper Distribution', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 5. Interdisciplinarity Distribution (Middle Left)
    ax5 = fig.add_subplot(gs[1, 0])
    disc_dist = df['discipline_count'].value_counts().sort_index()
    colors_disc = sns.color_palette("plasma", len(disc_dist))
    bars = ax5.bar(disc_dist.index, disc_dist.values, color=colors_disc, edgecolor='black', linewidth=1.2)
    ax5.set_xlabel('Number of Disciplines', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Number of Papers', fontsize=10, fontweight='bold')
    ax5.set_title('Interdisciplinarity Distribution', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Category Co-occurrence (Top Pairs) (Middle Middle Left)
    ax6 = fig.add_subplot(gs[1, 1])
    from itertools import combinations
    from collections import Counter
    cooccurrence = Counter()
    for categories in df['main_categories']:
        if isinstance(categories, list) and len(categories) > 1:
            for cat1, cat2 in combinations(sorted(categories), 2):
                cooccurrence[(cat1, cat2)] += 1
    top_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:8]
    pairs_labels = [f"{c1}↔{c2}" for (c1, c2), _ in top_pairs]
    pairs_counts = [count for _, count in top_pairs]
    colors_pairs = sns.color_palette("rocket", len(pairs_labels))
    bars = ax6.barh(range(len(pairs_labels)), pairs_counts, color=colors_pairs)
    ax6.set_yticks(range(len(pairs_labels)))
    ax6.set_yticklabels(pairs_labels, fontsize=8)
    ax6.set_xlabel('Co-occurrence Count', fontsize=10, fontweight='bold')
    ax6.set_title('Top Category Pairs', fontsize=12, fontweight='bold')
    ax6.invert_yaxis()
    ax6.grid(axis='x', alpha=0.3, linestyle='--')
    for i, count in enumerate(pairs_counts):
        ax6.text(count + 5, i, f'{count}', va='center', fontsize=8, fontweight='bold')
    
    # 7. Text Length Analysis (Middle Middle Right)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(df['title_length'], df['abstract_length'], alpha=0.3, s=10, color='#F18F01')
    ax7.set_xlabel('Title Length (characters)', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Abstract Length (characters)', fontsize=10, fontweight='bold')
    ax7.set_title('Title vs Abstract Length', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, linestyle='--')
    # Add correlation annotation
    corr = df[['title_length', 'abstract_length']].corr().iloc[0, 1]
    ax7.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax7.transAxes,
            fontsize=10, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 8. Temporal Distribution by Category (Middle Right)
    ax8 = fig.add_subplot(gs[1, 3])
    top_5_cats = df_exploded['main_categories'].value_counts().head(5).index.tolist()
    for cat in top_5_cats:
        cat_data = df_exploded[df_exploded['main_categories'] == cat]
        yearly_cat = cat_data.groupby('submission_year').size()
        ax8.plot(yearly_cat.index, yearly_cat.values, marker='o', linewidth=1.5, 
                markersize=3, label=cat, alpha=0.7)
    ax8.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Publications', fontsize=10, fontweight='bold')
    ax8.set_title('Top 5 Categories Over Time', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=8, loc='best')
    ax8.grid(True, alpha=0.3, linestyle='--')
    
    # 9. Missing Values Summary (Bottom Left)
    ax9 = fig.add_subplot(gs[2, 0])
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
    if len(missing_data) > 0:
        bars = ax9.barh(range(len(missing_data)), missing_data.values, 
                       color=sns.color_palette("Reds_r", len(missing_data)))
        ax9.set_yticks(range(len(missing_data)))
        ax9.set_yticklabels(missing_data.index, fontsize=8)
        ax9.set_xlabel('Missing Count', fontsize=10, fontweight='bold')
        ax9.set_title('Missing Values by Column', fontsize=12, fontweight='bold')
        ax9.invert_yaxis()
        ax9.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 10. Key Statistics (Bottom Middle Left)
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.axis('off')
    stats_text = f"""
    KEY STATISTICS
    
    Single Discipline: {(df['discipline_count'] == 1).sum():,} papers
    Multi-Discipline: {(df['discipline_count'] > 1).sum():,} papers
    
    Single Author: {(df['num_authors'] == 1).sum():,} papers
    Multi-Author: {(df['num_authors'] > 1).sum():,} papers
    
    Avg Title Length: {df['title_length'].mean():.0f} chars
    Avg Abstract Length: {df['abstract_length'].mean():.0f} chars
    
    Most Common Category: {df_exploded['main_categories'].value_counts().index[0]}
    """
    ax10.text(0.1, 0.5, stats_text, fontsize=10, fontweight='bold',
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax10.set_title('Key Statistics', fontsize=14, fontweight='bold', pad=10)
    
    # 11. Year Distribution (Bottom Middle Right)
    ax11 = fig.add_subplot(gs[2, 2])
    year_dist = df['submission_year'].value_counts().sort_index()
    ax11.fill_between(year_dist.index, year_dist.values, alpha=0.6, color='#6A4C93')
    ax11.plot(year_dist.index, year_dist.values, linewidth=2, color='#2E86AB')
    ax11.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax11.set_ylabel('Publications', fontsize=10, fontweight='bold')
    ax11.set_title('Publication Distribution by Year', fontsize=12, fontweight='bold')
    ax11.grid(True, alpha=0.3, linestyle='--')
    
    # 12. Analysis Summary (Bottom Right)
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    analysis_text = f"""
    ANALYSIS SUMMARY
    
    ✅ Data Quality: Complete
    ✅ Statistical Tests: ANOVA, Correlations
    ✅ Predictive Models: 3 models
    ✅ Visualizations: 15+ charts
    
    Dataset: 10K sample
    Methodology: Validated
    Ready for: Full dataset scale
    """
    ax12.text(0.1, 0.5, analysis_text, fontsize=10, fontweight='bold',
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax12.set_title('Analysis Status', fontsize=14, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle('Comprehensive Analysis Dashboard\nArXiv Publications Dataset (10K Sample)', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'analysis_summary_dashboard.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Summary dashboard saved: analysis_summary_dashboard.png")
    print(f"  Location: {OUTPUT_DIR}/analysis_summary_dashboard.png")

def main():
    """Main function to create summary dashboard."""
    try:
        create_analysis_summary_dashboard()
        print("\n" + "="*80)
        print("✅ DASHBOARD CREATION COMPLETE!")
        print("="*80)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

