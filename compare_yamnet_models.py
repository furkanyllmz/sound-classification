"""
YAMNet Model Karşılaştırma Script
Orijinal vs Improved modellerini karşılaştırır
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results(results_path):
    """Results JSON dosyasını yükle"""
    if not os.path.exists(results_path):
        return None
    with open(results_path, 'r') as f:
        return json.load(f)

def compare_models():
    """Tüm YAMNet modellerini karşılaştır"""

    print("\n" + "="*70)
    print("YAMNet Model Karşılaştırması")
    print("="*70 + "\n")

    # Model sonuçlarını topla
    models_data = {}

    # Orijinal YAMNet
    original_path = 'results/yamnet_transfer/results.json'
    if os.path.exists(original_path):
        models_data['Original'] = load_results(original_path)

    # Improved modeller
    architectures = ['shallow', 'medium', 'deep', 'residual', 'ensemble']
    for arch in architectures:
        results_path = f'results/yamnet_improved_{arch}/results.json'
        if os.path.exists(results_path):
            models_data[f'Improved-{arch}'] = load_results(results_path)

    if not models_data:
        print("❌ Hiç model sonucu bulunamadı!")
        print("\n💡 İpucu: Önce modelleri eğitin:")
        print("   python train_yamnet_transfer.py")
        print("   python train_yamnet_improved.py")
        return

    # Sonuçları göster
    print("📊 Model Sonuçları:\n")
    print(f"{'Model':<25} {'Test Accuracy':<15} {'Test Loss':<12}")
    print("-" * 55)

    accuracies = []
    losses = []
    model_names = []

    for model_name, data in models_data.items():
        acc = data['test_accuracy']
        loss = data['test_loss']
        accuracies.append(acc)
        losses.append(loss)
        model_names.append(model_name)
        print(f"{model_name:<25} {acc:>6.2%}          {loss:>8.4f}")

    # En iyi model
    best_idx = np.argmax(accuracies)
    print("\n" + "="*55)
    print(f"🏆 EN İYİ MODEL: {model_names[best_idx]}")
    print(f"   Accuracy: {accuracies[best_idx]:.2%}")
    print(f"   Loss: {losses[best_idx]:.4f}")
    print("="*55)

    # Sınıf bazında karşılaştırma
    print("\n📈 Sınıf Bazında Karşılaştırma:\n")

    # Sadece orijinal ve en iyi modeli karşılaştır
    if 'Original' in models_data:
        original_report = models_data['Original']['classification_report']
        best_report = models_data[model_names[best_idx]]['classification_report']
        class_names = models_data['Original']['class_names']

        print(f"{'Class':<20} {'Original F1':<15} {'Best F1':<15} {'Improvement':<15}")
        print("-" * 70)

        for cls in class_names:
            orig_f1 = original_report[cls]['f1-score']
            best_f1 = best_report[cls]['f1-score']
            improvement = best_f1 - orig_f1
            arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
            print(f"{cls:<20} {orig_f1:>6.2%}          {best_f1:>6.2%}          {arrow} {abs(improvement):>5.2%}")

    # Görselleştirme
    create_comparison_plots(model_names, accuracies, losses, models_data)

    print("\n✓ Karşılaştırma grafikleri kaydedildi: results/comparison/")

def create_comparison_plots(model_names, accuracies, losses, models_data):
    """Karşılaştırma grafikleri oluştur"""

    os.makedirs('results/comparison', exist_ok=True)

    # 1. Accuracy ve Loss karşılaştırması
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
    bars1 = axes[0].bar(range(len(model_names)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Model Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
    axes[0].set_ylim([min(accuracies) * 0.95, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Değerleri bar üzerine yaz
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

    # Loss bar chart
    bars2 = axes[1].bar(range(len(model_names)), losses, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Model Loss Karşılaştırması', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Değerleri bar üzerine yaz
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/comparison/accuracy_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Sınıf bazında F1-score karşılaştırması
    if 'Original' in models_data and len(models_data) > 1:
        class_names = models_data['Original']['class_names']

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(class_names))
        width = 0.8 / len(models_data)

        for i, (model_name, data) in enumerate(models_data.items()):
            f1_scores = [data['classification_report'][cls]['f1-score'] for cls in class_names]
            offset = width * i - (width * len(models_data) / 2) + width / 2
            ax.bar(x + offset, f1_scores, width, label=model_name, alpha=0.8)

        ax.set_xlabel('Class', fontweight='bold', fontsize=12)
        ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
        ax.set_title('Sınıf Bazında F1-Score Karşılaştırması', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig('results/comparison/class_f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Radar chart (en iyi 3 model için)
    if len(models_data) >= 1:
        # En iyi 3 modeli seç
        sorted_models = sorted(zip(model_names, accuracies), key=lambda x: x[1], reverse=True)[:3]
        top_models = [name for name, _ in sorted_models]

        class_names = list(models_data.values())[0]['class_names']

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='polar')

        angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
        angles += angles[:1]

        for model_name in top_models:
            if model_name in models_data:
                data = models_data[model_name]
                f1_scores = [data['classification_report'][cls]['f1-score'] for cls in class_names]
                f1_scores += f1_scores[:1]
                ax.plot(angles, f1_scores, 'o-', linewidth=2, label=model_name)
                ax.fill(angles, f1_scores, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(class_names, size=10)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart\n(F1-Score per Class)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('results/comparison/radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    compare_models()
