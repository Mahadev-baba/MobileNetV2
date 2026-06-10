import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------
epochs = np.arange(1, 11)

# 140K Model Training History
train_acc_140 = [0.9609, 0.9899, 0.9929, 0.9943, 0.9945, 0.9949, 0.9954, 0.9961, 0.9958, 0.9968]
val_acc_140 = [0.9837, 0.9911, 0.9909, 0.9921, 0.9909, 0.9909, 0.9959, 0.9939, 0.9972, 0.9961]
train_loss_140 = [0.3515, 0.3231, 0.3200, 0.3187, 0.3186, 0.3182, 0.3177, 0.3171, 0.3173, 0.3163]
val_loss_140 = [0.3289, 0.3217, 0.3221, 0.3208, 0.3222, 0.3219, 0.3171, 0.3195, 0.3161, 0.3169]

# 200K Model Training History
train_acc_200 = [0.8699, 0.9256, 0.9397, 0.9506, 0.9557, 0.9624, 0.9646, 0.9695, 0.9707, 0.9739]
val_acc_200 = [0.9142, 0.9225, 0.9247, 0.9445, 0.9452, 0.9508, 0.9468, 0.9547, 0.9481, 0.9570]
train_loss_200 = [0.4319, 0.3814, 0.3691, 0.3596, 0.3553, 0.3492, 0.3471, 0.3425, 0.3415, 0.3383]
val_loss_200 = [0.3918, 0.3857, 0.3841, 0.3659, 0.3649, 0.3601, 0.3639, 0.3567, 0.3635, 0.3544]

# Final Evaluation Summary Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Specificity', 'Sensitivity', 'MCC', 'Kappa']
val_140 = [0.996700, 0.996700, 0.996700, 0.996700, 0.999840, 0.996700, 0.996700, 0.993400, 0.993400]
val_200 = [0.958300, 0.950905, 0.966500, 0.958639, 0.990930, 0.950100, 0.966500, 0.916723, 0.916600]

# ---------------------------------------------------------
# 2. PLOT 1: ACCURACY LEARNING CURVES
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(epochs, train_acc_140, color='royalblue', marker='o', linestyle='-', linewidth=2, label='140K Train Acc')
ax1.plot(epochs, val_acc_140, color='royalblue', marker='s', linestyle='--', linewidth=1.5, label='140K Val Acc')
ax1.plot(epochs, train_acc_200, color='darkorange', marker='o', linestyle='-', linewidth=2, label='200K Train Acc')
ax1.plot(epochs, val_acc_200, color='darkorange', marker='s', linestyle='--', linewidth=1.5, label='200K Val Acc')

ax1.set_title('Training and Validation Accuracy Comparison', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Epochs', fontsize=10)
ax1.set_ylabel('Accuracy', fontsize=10)
ax1.set_xticks(epochs)
ax1.set_ylim(0.85, 1.01)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('combined_accuracy_curves.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. PLOT 2: LOSS LEARNING CURVES
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(epochs, train_loss_140, color='royalblue', marker='o', linestyle='-', linewidth=2, label='140K Train Loss')
ax2.plot(epochs, val_loss_140, color='royalblue', marker='s', linestyle='--', linewidth=1.5, label='140K Val Loss')
ax2.plot(epochs, train_loss_200, color='darkorange', marker='o', linestyle='-', linewidth=2, label='200K Train Loss')
ax2.plot(epochs, val_loss_200, color='darkorange', marker='s', linestyle='--', linewidth=1.5, label='200K Val Loss')

ax2.set_title('Training and Validation Loss Comparison', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Epochs', fontsize=10)
ax2.set_ylabel('Loss Value', fontsize=10)
ax2.set_xticks(epochs)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('combined_loss_curves.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 4. PLOT 3: SUMMARY METRICS BAR CHART (SORTED)
# ---------------------------------------------------------
combined = list(zip(metrics, val_140, val_200))
combined_sorted = sorted(combined, key=lambda x: x[1])

sorted_metrics = [x[0] for x in combined_sorted]
sorted_140 = [x[1] for x in combined_sorted]
sorted_200 = [x[2] for x in combined_sorted]

fig3, ax3 = plt.subplots(figsize=(10, 5.5))
x_indexes = np.arange(len(sorted_metrics))
bar_width = 0.35

ax3.bar(x_indexes - bar_width/2, sorted_140, bar_width, label='140K Model', color='royalblue')
ax3.bar(x_indexes + bar_width/2, sorted_200, bar_width, label='200K Model', color='darkorange')

ax3.set_title('Final Evaluation Metrics Comparison (Sorted by 140K Performance)', fontsize=12, fontweight='bold', pad=10)
ax3.set_ylabel('Score Value', fontsize=10)
ax3.set_xticks(x_indexes)
ax3.set_xticklabels(sorted_metrics, rotation=45, ha='right', fontsize=9)
ax3.set_ylim(0.88, 1.01)
ax3.grid(True, axis='y', linestyle=':', alpha=0.6)
ax3.legend(loc='lower left')
plt.tight_layout()
plt.savefig('combined_metrics_comparison.png', dpi=300)
plt.close()

print("All plots created and saved successfully.")
