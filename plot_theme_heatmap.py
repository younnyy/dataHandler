import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def plot_heatmap():
    # Data
    data = {
        '主题1': [1.000, 0.738, 0.231, 0.221],
        '主题2': [0.738, 1.000, 0.299, 0.357],
        '主题3': [0.231, 0.299, 1.000, 0.223],
        '主题4': [0.221, 0.357, 0.223, 1.000]
    }
    index = ['主题1', '主题2', '主题3', '主题4']
    df = pd.DataFrame(data, index=index)

    # Font Setup for Chinese on Mac
    # Try to find a suitable Chinese font
    font_path = '/System/Library/Fonts/STHeiti Light.ttc'
    if not os.path.exists(font_path):
        # Fallback to common Mac Chinese fonts
        font_path = '/System/Library/Fonts/PingFang.ttc' 
    
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
    else:
        print("Warning: Chinese font not found, characters may not display correctly.")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    plt.figure(figsize=(8, 6))
    
    # Plot Heatmap
    # annot=True to show numbers
    # cmap='coolwarm' or 'YlGnBu' for good contrast
    # vmin=0, vmax=1 for correlation
    sns.heatmap(df, annot=True, cmap='Blues', vmin=0, vmax=1, fmt='.3f', 
                linewidths=1, linecolor='white')

    plt.title('主题相关性热力图', fontsize=15, fontproperties=prop if os.path.exists(font_path) else None)
    
    # Save
    output_path = 'theme_correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    # plt.show()

if __name__ == "__main__":
    plot_heatmap()
