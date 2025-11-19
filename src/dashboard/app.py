from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import sys
import ast
import base64
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_processed_data

app = Flask(__name__)

# Load data once when the app starts
try:
    # Load data, using mock=False to use the processed sample
    df = load_processed_data(mock=False)
    # Ensure 'main_categories' is a list of strings
    df['main_categories'] = df['main_categories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Combine titles and abstracts for word cloud
    df['text_content'] = df['title'] + ' ' + df['abstract']
except FileNotFoundError:
    df = pd.DataFrame({'title': [], 'abstract': [], 'text_content': []})
    print("Warning: Could not load data. Dashboard functionality will be limited.")

def generate_word_cloud(text, filtered_df=None):
    """Generates an enhanced word cloud image from text and returns it as a base64 string."""
    if not text:
        return "", {}
        
    # Comprehensive stop words list
    stop_words = set([
        'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'on', 'with', 'this', 
        'paper', 'analysis', 'using', 'which', 'from', 'study', 'research', 'novel', 
        'technique', 'approach', 'we', 'our', 'results', 'show', 'present', 'propose',
        'method', 'methods', 'data', 'model', 'models', 'based', 'different', 'also',
        'can', 'used', 'use', 'one', 'two', 'new', 'time', 'more', 'these', 'their',
        'than', 'when', 'where', 'what', 'how', 'why', 'are', 'was', 'were', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'shall', 'being', 'been', 'become', 'becomes', 'becoming'
    ])
    
    # Generate word cloud with better settings
    wordcloud = WordCloud(
        width=1000, 
        height=500, 
        background_color='white',
        stopwords=stop_words,
        max_words=100,
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10,
        max_font_size=100,
        collocations=True
    ).generate(text)
    
    # Get word frequencies for statistics
    word_freq = wordcloud.words_
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Create enhanced visualization with subplots
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35, figure=fig)
    
    # Main word cloud (larger, left side)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(wordcloud, interpolation='bilinear')
    ax1.axis("off")
    ax1.set_title('Word Cloud Visualization', fontsize=14, fontweight='bold', pad=20)
    
    # Top 20 words bar chart (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if top_words:
        words, freqs = zip(*top_words)
        colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
        bars = ax2.barh(range(len(words)), freqs, color=colors)
        ax2.set_yticks(range(len(words)))
        ax2.set_yticklabels(words, fontsize=9)
        ax2.set_xlabel('Relative Frequency', fontsize=10, fontweight='bold')
        ax2.set_title('Top 20 Keywords', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        # Add value labels
        for i, freq in enumerate(freqs):
            ax2.text(freq + 0.01, i, f'{freq:.2f}', va='center', fontsize=8)
    
    # Statistics panel (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Calculate statistics if filtered_df is provided
    stats_text = "WORD CLOUD STATISTICS\n\n"
    if filtered_df is not None and len(filtered_df) > 0:
        stats_text += f"Papers Analyzed: {len(filtered_df):,}\n"
        stats_text += f"Total Words: {len(text.split()):,}\n"
        stats_text += f"Unique Words: {len(set(text.lower().split())):,}\n"
        stats_text += f"Top Categories:\n"
        try:
            top_cats = filtered_df['main_categories'].explode().value_counts().head(3)
            for i, (cat, count) in enumerate(top_cats.items(), 1):
                stats_text += f"  {i}. {cat}: {count}\n"
        except:
            pass
        stats_text += f"\nYear Range:\n"
        try:
            if 'submission_year' in filtered_df.columns:
                stats_text += f"  {filtered_df['submission_year'].min()}-{filtered_df['submission_year'].max()}\n"
        except:
            pass
    else:
        stats_text += f"Total Words: {len(text.split()):,}\n"
        stats_text += f"Unique Words: {len(set(text.lower().split())):,}\n"
        stats_text += f"Top Keywords: {len(top_words)}\n"
    
    ax3.text(0.1, 0.5, stats_text, fontsize=10, fontweight='bold',
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.35, wspace=0.35)
    
    # Convert to PNG image in memory
    img = BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img.seek(0)
    
    # Encode to base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    
    # Return image and statistics
    stats_dict = {
        'top_words': dict(top_words[:10]),  # Top 10 for JSON
        'total_words': len(text.split()),
        'unique_words': len(set(text.lower().split())),
        'paper_count': len(filtered_df) if filtered_df is not None else 0
    }
    
    return f"data:image/png;base64,{img_base64}", stats_dict

@app.route('/')
def index():
    """Main dashboard page."""
    # In a full dashboard, you would pass summary statistics or initial plots here
    return render_template('index.html')

@app.route('/wordcloud', methods=['POST'])
def wordcloud_endpoint():
    """Enhanced endpoint to generate interactive word cloud with statistics."""
    try:
        w1 = request.form.get('word', '').strip().lower()
        year_min = request.form.get('year_min', '').strip()
        year_max = request.form.get('year_max', '').strip()
        
        if not w1:
            return jsonify({'error': 'Please enter a keyword.'}), 400

        # Filter papers where w1 is present in the title or abstract
        # Handle multi-word phrases: if keyword has spaces, search as phrase; otherwise use word boundaries
        if ' ' in w1:
            # Multi-word phrase: search for exact phrase
            search_pattern = w1
        else:
            # Single word: use word boundaries
            search_pattern = r'\b' + w1 + r'\b'
        
        filtered_df = df[df['text_content'].str.contains(search_pattern, case=False, na=False, regex=(' ' not in w1))].copy()
        
        # Apply year filters if provided
        if year_min:
            try:
                year_min = int(year_min)
                if 'submission_year' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['submission_year'] >= year_min]
            except (ValueError, KeyError, TypeError):
                pass
        
        if year_max:
            try:
                year_max = int(year_max)
                if 'submission_year' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['submission_year'] <= year_max]
            except (ValueError, KeyError, TypeError):
                pass
        
        if filtered_df.empty:
            # Provide helpful error message
            error_msg = f'No papers found containing "{w1}"'
            if year_min or year_max:
                year_range = []
                if year_min:
                    year_range.append(f'from {year_min}')
                if year_max:
                    year_range.append(f'to {year_max}')
                error_msg += f' in the year range {", ".join(year_range)}'
            error_msg += '. Try: removing year filters, using a different keyword, or checking spelling.'
            return jsonify({'error': error_msg}), 404

        # Combine all text content from the filtered papers
        combined_text = " ".join(filtered_df['text_content'].tolist())
        
        # Generate enhanced word cloud with statistics
        img_data, stats = generate_word_cloud(combined_text, filtered_df)
        
        # Get additional statistics
        categories = {}
        year_range = {}
        try:
            categories = filtered_df['main_categories'].explode().value_counts().head(5).to_dict()
        except Exception:
            pass
        
        try:
            if 'submission_year' in filtered_df.columns and not filtered_df['submission_year'].isna().all():
                year_range = {
                    'min': int(filtered_df['submission_year'].min()),
                    'max': int(filtered_df['submission_year'].max())
                }
        except Exception:
            pass
        
        return jsonify({
            'image': img_data, 
            'count': len(filtered_df),
            'stats': stats,
            'categories': categories,
            'year_range': year_range
        })
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/stats', methods=['GET'])
def get_dashboard_stats():
    """Get overall dashboard statistics."""
    try:
        stats = {
            'total_papers': len(df),
            'year_range': {
                'min': int(df['submission_year'].min()) if 'submission_year' in df.columns else None,
                'max': int(df['submission_year'].max()) if 'submission_year' in df.columns else None
            },
            'top_categories': {},
            'avg_authors': 0,
            'total_categories': 0
        }
        
        try:
            df_exploded = df.explode('main_categories')
            stats['top_categories'] = df_exploded['main_categories'].value_counts().head(10).to_dict()
            stats['total_categories'] = df_exploded['main_categories'].nunique()
        except:
            pass
        
        try:
            df['num_authors'] = df['authors'].apply(
                lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.strip().startswith('[') 
                else len(x.split(',')) if isinstance(x, str) else 0
            )
            stats['avg_authors'] = float(df['num_authors'].mean())
        except:
            pass
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local development, use a standard port
    # In the sandbox, we will use gunicorn to run the app
    print("Dashboard application ready. Run with: gunicorn -w 4 -b 0.0.0.0:8080 app:app")
    # app.run(debug=True, port=5000)

