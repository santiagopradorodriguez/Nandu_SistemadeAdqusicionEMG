import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('EMG_desarrollo/base_de_datos_electrodos/2026-06-10/PCA/PCA_Comp15_SNR0-5/vector_maestro_300d.csv')
features = df.drop(columns=['Vocal', 'Toma']).values
vocal_A = df['Vocal'].values == 'A'
mean_A = np.mean(features[vocal_A], axis=0)

plt.figure()
for ch in range(3):
    plt.plot(mean_A[ch*100:(ch+1)*100], label=f'Ch {ch}')
plt.axvline(x=40, color='r', linestyle='--')
plt.legend()
plt.savefig('test_auditor.png')
print("Mean min/max:", np.min(mean_A), np.max(mean_A))
