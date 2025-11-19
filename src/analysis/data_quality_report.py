import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import ast
from scipy import stats

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_processed_data

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'analysis_results', 'data_quality')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    print("\n" + "="*80)
    print("MISSING VALUE ANALYSIS")
    print("="*80)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    print("\nColumns with Missing Values:")
    print(missing_df.to_string(index=False))
    
    # Visualize missing values
    if len(missing_df) > 0:
        plt.figure(figsize=(12, 6))
        bars = plt.barh(missing_df['Column'], missing_df['Missing Percentage'], 
                       color=sns.color_palette("Reds_r", len(missing_df)))
        plt.xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
        plt.title('Missing Values by Column', fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (idx, val) in enumerate(missing_df.iterrows()):
            plt.text(val['Missing Percentage'] + 0.5, i, 
                    f"{val['Missing Percentage']:.1f}% ({int(val['Missing Count']):,})",
                    va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'missing_values_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Missing values visualization saved: missing_values_analysis.png")
    
    return missing_df

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def analyze_outliers(df):
    """Analyze outliers in numerical columns."""
    print("\n" + "="*80)
    print("OUTLIER DETECTION ANALYSIS")
    print("="*80)
    
    # Ensure main_categories is a list
    df['main_categories'] = df['main_categories'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Create derived numerical features
    df['num_authors'] = df['authors'].apply(
        lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.strip().startswith('[') 
        else len(x.split(',')) if isinstance(x, str) else 0
    )
    df['discipline_count'] = df['main_categories'].apply(len)
    df['title_length'] = df['title'].str.len()
    df['abstract_length'] = df['abstract'].str.len()
    
    numerical_cols = ['submission_year', 'num_authors', 'discipline_count', 'title_length', 'abstract_length']
    
    outlier_summary = []
    
    for col in numerical_cols:
        if col in df.columns:
            outliers, lower, upper = detect_outliers_iqr(df, col)
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(df)) * 100
            
            outlier_summary.append({
                'Column': col,
                'Outlier Count': outlier_count,
                'Outlier Percentage': outlier_percent,
                'Lower Bound': lower,
                'Upper Bound': upper,
                'Min Value': df[col].min(),
                'Max Value': df[col].max(),
                'Mean': df[col].mean(),
                'Median': df[col].median()
            })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print("\nOutlier Summary:")
    print(outlier_df.to_string(index=False))
    
    # Create outlier visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        if col in df.columns and idx < len(axes):
            ax = axes[idx]
            
            # Box plot
            bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor(sns.color_palette("Set2")[0])
            bp['boxes'][0].set_alpha(0.7)
            
            ax.set_title(f'{col.replace("_", " ").title()}\nOutliers: {outlier_df[outlier_df["Column"]==col]["Outlier Count"].values[0]:,}',
                        fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
    
    # Remove extra subplot
    if len(numerical_cols) < len(axes):
        axes[-1].remove()
    
    plt.suptitle('Outlier Detection: Box Plots for Numerical Features', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'outlier_detection_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Outlier visualization saved: outlier_detection_boxplots.png")
    
    return outlier_df

def analyze_distributions(df):
    """Analyze distributions of key variables."""
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Ensure derived features exist
    if 'num_authors' not in df.columns:
        df['num_authors'] = df['authors'].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.strip().startswith('[') 
            else len(x.split(',')) if isinstance(x, str) else 0
        )
    if 'discipline_count' not in df.columns:
        df['main_categories'] = df['main_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['discipline_count'] = df['main_categories'].apply(len)
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Submission year distribution
    ax = axes[0, 0]
    df['submission_year'].hist(bins=30, ax=ax, color=sns.color_palette("viridis", 1)[0], edgecolor='black')
    ax.set_title('Distribution of Submission Years', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Number of authors distribution
    ax = axes[0, 1]
    author_counts = df['num_authors'].value_counts().sort_index()
    author_counts = author_counts[author_counts.index <= 20]  # Limit for clarity
    ax.bar(author_counts.index, author_counts.values, color=sns.color_palette("coolwarm", len(author_counts)), edgecolor='black')
    ax.set_title('Distribution of Authors per Paper', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Authors', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Discipline count distribution
    ax = axes[0, 2]
    disc_counts = df['discipline_count'].value_counts().sort_index()
    ax.bar(disc_counts.index, disc_counts.values, color=sns.color_palette("plasma", len(disc_counts)), edgecolor='black')
    ax.set_title('Distribution of Disciplines per Paper', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Disciplines', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Title length distribution
    ax = axes[1, 0]
    df['title_length'].hist(bins=50, ax=ax, color=sns.color_palette("rocket", 1)[0], edgecolor='black')
    ax.set_title('Distribution of Title Lengths', fontsize=12, fontweight='bold')
    ax.set_xlabel('Character Count', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Abstract length distribution
    ax = axes[1, 1]
    df['abstract_length'].hist(bins=50, ax=ax, color=sns.color_palette("mako", 1)[0], edgecolor='black')
    ax.set_title('Distribution of Abstract Lengths', fontsize=12, fontweight='bold')
    ax.set_xlabel('Character Count', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Top categories
    ax = axes[1, 2]
    df_exploded = df.explode('main_categories')
    top_cats = df_exploded['main_categories'].value_counts().head(10)
    ax.barh(range(len(top_cats)), top_cats.values, color=sns.color_palette("husl", len(top_cats)))
    ax.set_yticks(range(len(top_cats)))
    ax.set_yticklabels(top_cats.index, fontsize=9)
    ax.set_title('Top 10 Research Categories', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency', fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Distribution visualization saved: data_distributions.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary_cols = ['submission_year', 'num_authors', 'discipline_count', 'title_length', 'abstract_length']
    summary_stats = df[summary_cols].describe()
    print(summary_stats)
    
    return summary_stats

def generate_correlation_analysis(df):
    """Generate correlation analysis between numerical features."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Ensure derived features exist
    if 'num_authors' not in df.columns:
        df['num_authors'] = df['authors'].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.strip().startswith('[') 
            else len(x.split(',')) if isinstance(x, str) else 0
        )
    if 'discipline_count' not in df.columns:
        df['main_categories'] = df['main_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['discipline_count'] = df['main_categories'].apply(len)
    if 'title_length' not in df.columns:
        df['title_length'] = df['title'].str.len()
    if 'abstract_length' not in df.columns:
        df['abstract_length'] = df['abstract'].str.len()
    
    # Select numerical columns
    numerical_cols = ['submission_year', 'num_authors', 'discipline_count', 'title_length', 'abstract_length']
    corr_data = df[numerical_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, mask=mask,
                vmin=-1, vmax=1)
    plt.title('Correlation Matrix: Numerical Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Correlation heatmap saved: correlation_heatmap.png")
    
    print("\nCorrelation Matrix:")
    print(corr_data.round(3))
    
    # Identify strong correlations
    print("\nStrong Correlations (|r| > 0.3):")
    for i in range(len(corr_data.columns)):
        for j in range(i+1, len(corr_data.columns)):
            corr_val = corr_data.iloc[i, j]
            if abs(corr_val) > 0.3:
                print(f"  {corr_data.columns[i]} ↔ {corr_data.columns[j]}: {corr_val:.3f}")
    
    return corr_data

def generate_data_quality_summary(df):
    """Generate a comprehensive data quality summary report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA QUALITY SUMMARY")
    print("="*80)
    
    # Basic info
    print(f"\nDataset Overview:")
    print(f"  • Total Records: {len(df):,}")
    print(f"  • Total Columns: {len(df.columns)}")
    print(f"  • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    print(f"\nData Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  • {dtype}: {count} columns")
    
    # Missing values summary
    total_missing = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_percent = (total_missing / total_cells) * 100
    print(f"\nMissing Values:")
    print(f"  • Total Missing Cells: {total_missing:,}")
    print(f"  • Missing Percentage: {missing_percent:.2f}%")
    
    # Year range
    if 'submission_year' in df.columns:
        print(f"\nTemporal Coverage:")
        print(f"  • Year Range: {df['submission_year'].min()} - {df['submission_year'].max()}")
        print(f"  • Span: {df['submission_year'].max() - df['submission_year'].min()} years")
    
    # Save summary to file
    summary_file = os.path.join(OUTPUT_DIR, 'data_quality_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("DATA QUALITY SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset Overview:\n")
        f.write(f"  Total Records: {len(df):,}\n")
        f.write(f"  Total Columns: {len(df.columns)}\n")
        f.write(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        f.write(f"Missing Values: {total_missing:,} ({missing_percent:.2f}%)\n")
        if 'submission_year' in df.columns:
            f.write(f"Year Range: {df['submission_year'].min()} - {df['submission_year'].max()}\n")
    
    print(f"\n✓ Summary report saved: data_quality_summary.txt")

def main():
    """Main function to run comprehensive data quality analysis."""
    try:
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA QUALITY ANALYSIS")
        print("="*80)
        
        # Load data
        print("\nLoading processed data...")
        df = load_processed_data(mock=False)
        
        # Ensure main_categories is a list
        df['main_categories'] = df['main_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # Run all analyses
        missing_df = analyze_missing_values(df)
        outlier_df = analyze_outliers(df)
        summary_stats = analyze_distributions(df)
        corr_matrix = generate_correlation_analysis(df)
        generate_data_quality_summary(df)
        
        print("\n" + "="*80)
        print("✅ DATA QUALITY ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {OUTPUT_DIR}")
        print("\nGenerated Files:")
        print("  1. missing_values_analysis.png - Missing value visualization")
        print("  2. outlier_detection_boxplots.png - Outlier detection box plots")
        print("  3. data_distributions.png - Distribution analysis")
        print("  4. correlation_heatmap.png - Correlation matrix")
        print("  5. data_quality_summary.txt - Summary report")
        print("\n" + "="*80)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the data acquisition script first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

