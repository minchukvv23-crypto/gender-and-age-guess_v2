from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'telegram_author_profile_synthetic.csv'
ASSETS = ROOT / 'report_assets'
ASSETS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
counts = df['class_label'].value_counts().sort_index()
plt.figure(figsize=(9, 4.2))
counts.plot(kind='bar')
plt.title('Distribution of synthetic dataset classes')
plt.xlabel('Class label')
plt.ylabel('Number of texts')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(ASSETS / 'dataset_distribution.png', dpi=180)
plt.close()

# simple pipeline diagram
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis('off')
boxes = [
    ('Synthetic\nTelegram text', 0.06),
    ('Cleaning and\nnormalization', 0.26),
    ('TF-IDF\nvectorization', 0.46),
    ('MLP neural\nnetwork', 0.66),
    ('Class:\ngender + age', 0.84),
]
for text, x in boxes:
    rect = plt.Rectangle((x, 0.35), 0.14, 0.28, fill=False, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.07, 0.49, text, ha='center', va='center', fontsize=9)
for _, x in boxes[:-1]:
    ax.annotate('', xy=(x+0.18, 0.49), xytext=(x+0.14, 0.49), arrowprops=dict(arrowstyle='->', lw=1.4))
plt.tight_layout()
plt.savefig(ASSETS / 'pipeline_scheme.png', dpi=180)
plt.close()
